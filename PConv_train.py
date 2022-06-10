import os
import gc
import glob
import argparse

from copy import deepcopy
from datetime import datetime

# import these here because of gdal partial import error
# if loaded from libs
import pandas as pd
import rasterio as rio
import numpy as np
import rioxarray
import xarray as xr

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, EarlyStopping
 
from Pconv.libs.utils import MaskGenerator
from Pconv.libs.model_analysis import keras_model_memory_usage_in_bytes
from Pconv.libs.pconv_model import pconv_model


# --------------------------------------------------------------------------------------- #
#                                    Argument Parsing                                     #
# --------------------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
stages = [0,1,2]

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

parser.add_argument('-i', '--input', dest='input', default=None, help='directory containing val and train sets')


parser.add_argument('-c', '--checkpoint', dest='checkpoint', default=None, help='Name of model checkpoint to be loaded')
parser.add_argument('-a', '--analysis', dest='analysis', default=False, type=str2bool, const=True, nargs='?', help='Show model summary and size')
parser.add_argument('-r', '--run', dest='run', default=True, type=str2bool, const=True, nargs='?', help='Run the model')
parser.add_argument('-n', '--name', dest='name', default='pconv_model', type=str, help="Name to be used for outputs (learning_curve, weights etc.)")
parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.0, help="Dropout value, default = 0.0")

parser.add_argument('-vgg_norm', '--vgg_norm', dest='vgg_norm', default=True, type=str2bool, const=True, nargs='?', help='Normalise loss function to imagenet specs')
parser.add_argument('-m', '--mask', dest='mask', default='segmented', type=str, help="Mask used for final evaluation")

parser.add_argument('-p', '--penalty', dest='penalty', default=False, type=str2bool, const=True, nargs='?', help='Modified penalty loss function')
parser.add_argument('-area', '--area', dest='area', default=False, type=str2bool, const=True, nargs='?', help='Modified area scale formula loss function')

parser.add_argument('-resume', '--resume', dest='resume', default=0, type=int, choices=stages, help="Resume the training at a previous stage")

# model calibation 
parser.add_argument('--shape', dest='shape', default=256, type=int, help="Shape of input image, should be 256 or less, 512 required model adjustments")
parser.add_argument('--batch', dest='batch', default=8, type=int, help="Batch size used by model")
parser.add_argument('--step_train', dest='step_train', default=1000, type=int, help="# steps per epoch in train set")
parser.add_argument('--step_val', dest='step_val', default=100, type=int, help="# steps per epoch in val set")
parser.add_argument('--epoch1', dest='epoch1', default=70, type=int, help="# epochs in stage 1")
parser.add_argument('--epoch2', dest='epoch2', default=50, type=int, help="# epochs in stage 2")
parser.add_argument('--epoch1_lr', dest='epoch1_lr', default=0.0002, type=float, help="learning rate in stage 1")
parser.add_argument('--epoch2_lr', dest='epoch2_lr', default=0.00005, type=float, help="learning rate in stage 2")


VGG16_WEIGHTS   = "Pconv/data/vgg16_pytorch2keras.h5"

TB_DIR          = "Pconv/callbacks/tensorboard/"
CSV_DIR         = "Pconv/callbacks/csvlogger/"
WEIGHTS_DIR     = "Pconv/callbacks/weights/"
BEST_DIR        = "Pconv/callbacks/best_model/"

STAGE_1         = "initial/"
STAGE_2         = "fine_tuning/"
SEED            = 42

def main():
    args = parser.parse_args()
    # --------------------------------------------------------------------------------------- #
    #                                         Config                                          #
    # --------------------------------------------------------------------------------------- #
    name       = args.name
    shape      = args.shape
    batch_size = args.batch
    step_train = args.step_train
    step_val   = args.step_val
    epoch1     = args.epoch1
    epoch2     = args.epoch2
    epoch1_lr  = args.epoch1_lr
    epoch2_lr  = args.epoch2_lr
    
    

    if args.checkpoint != None:
        if os.path.isfile(WEIGHTS_DIR + args.checkpoint):
            model_weights = WEIGHTS_DIR + args.checkpoint
        else:
            print("Cannot find file {} in the directory {}".format(args.checkpoint, WEIGHTS_DIR))
            exit()

    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    class MemoryCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            K.clear_session()

    # --------------------------------------------------------------------------------------- #
    #                                      Data Loader                                        #
    # --------------------------------------------------------------------------------------- #

    train_paths = []
    train_mask_paths = []
    val_paths = []
    val_mask_paths = []


    for filename in glob.iglob(args.input + 'train/images/*'):
        train_paths.append(filename)
        filename = filename.replace('images', 'masks')
        train_mask_paths.append(filename.replace('.tif', '_mask.tif'))

    for filename in glob.iglob(args.input + 'val/images/*'):
        val_paths.append(filename)
        filename = filename.replace('images', 'masks')
        val_mask_paths.append(filename.replace('.tif', '_mask.tif'))

    train_df = pd.DataFrame({'image_path': train_paths, 'mask_path': train_mask_paths})
    val_df   = pd.DataFrame({'image_path': val_paths,   'mask_path': val_mask_paths})

    if args.mask == 'segmented':

        class AugmentedDataGenerator(ImageDataGenerator):
            def flow_from_dataframe(self, directory, *args, **kwargs):
                generator = super().flow_from_dataframe(directory, class_mode=None, color_mode='grayscale', *args, **kwargs)
                seed = None if 'seed' not in kwargs else kwargs['seed']
                while True:
                    
                    img = next(generator)
                    
                    gc.collect()
                    yield img

        # Create training generator
        train_datagen = AugmentedDataGenerator(
            horizontal_flip=True
        )
        train_generator_image = train_datagen.flow_from_dataframe(
            train_df, x_col = 'image_path',
            target_size = (shape, shape),
            batch_size = batch_size,
            seed = SEED
        )
        train_generator_mask = train_datagen.flow_from_dataframe(
            train_df, x_col = 'mask_path',
            target_size = (shape, shape),
            batch_size = batch_size,
            seed = SEED
        )

        # Create validation generator
        val_datagen = AugmentedDataGenerator()
        val_generator_image = val_datagen.flow_from_dataframe(
            val_df, x_col = 'image_path', 
            target_size = (shape, shape), 
            batch_size = batch_size, 
            seed = SEED 
        )
        val_generator_mask = val_datagen.flow_from_dataframe(
            val_df, x_col = 'mask_path',  
            target_size = (shape, shape), 
            batch_size = batch_size, 
            seed = SEED
        )

        def image_and_mask_generator(image_generator, mask_generator):
            generator = zip(image_generator, mask_generator)
            for (img, mask) in generator:

                img_max = np.amax(img)
                img_min = np.amin(img)
                img = (img - img_min) / (img_max - img_min)

                masked = deepcopy(img)
                masked[mask==1] = 1
                mask = 1 - mask

                gc.collect()
                yield [masked, mask, img], img


        train_generator = image_and_mask_generator(train_generator_image, train_generator_mask)        
        val_generator   = image_and_mask_generator(val_generator_image, val_generator_mask)

    elif args.mask == 'box':

        class AugmentedDataGenerator(ImageDataGenerator):
            def flow_from_dataframe(self, directory, mask_generator, *args, **kwargs):
                generator = super().flow_from_dataframe(directory, class_mode=None, color_mode='grayscale', *args, **kwargs)
                seed = None if 'seed' not in kwargs else kwargs['seed']
                while True:
                    
                    img = next(generator)
                    img_max = np.amax(img)
                    img_min = np.amin(img)
                    img = (img - img_min) / (img_max - img_min)

                    mask = np.stack([
                        mask_generator.sample()
                        for _ in range(img.shape[0])], axis=0
                    )

                    masked = deepcopy(img)
                    masked[mask==0] = 1
                    
                    gc.collect()
                    yield [masked, mask, img], img
        
        train_datagen = AugmentedDataGenerator(
            horizontal_flip=True
        )
        train_generator = train_datagen.flow_from_dataframe(
            train_df, 
            MaskGenerator(shape, shape, 128, 128),
            x_col = 'image_path',
            target_size=(shape, shape),
            batch_size=batch_size
        )

        val_datagen = AugmentedDataGenerator(
            horizontal_flip=True
        )
        val_generator = val_datagen.flow_from_dataframe(
            val_df, 
            MaskGenerator(shape, shape, 128, 128),
            x_col = 'image_path',
            target_size=(shape, shape),
            batch_size=batch_size,
            seed = SEED
        )

    else:
        print("Invalid mask algorithm choosen, options: (segmented, box)")
        exit()

    # get time of program start
    start = datetime.now()
    start = start.strftime("%d/%m/%Y %H:%M:%S")


    # define and compile model
    model = pconv_model(fine_tuning=False, lr=epoch1_lr, image_size=(shape, shape), 
                        drop=args.dropout, vgg16_weights=VGG16_WEIGHTS, vgg_norm=args.vgg_norm,
                        loss_penalty=args.penalty, loss_area=args.area)
    if args.resume == 1:
        if args.checkpoint != None:
            model.load_weights(model_weights)
            print("Successfully loaded model weights from {}\n".format(model_weights))
        else:
            print("Checkpoint was not provided, can not resume training at stage 1")

    if args.run and (args.resume == 0 or args.resume == 1):
    # --------------------------------------------------------------------------------------- #
    #                                        Training                                         #
    # --------------------------------------------------------------------------------------- #
        history = model.fit(
            train_generator,
            steps_per_epoch=step_train,
            epochs=epoch1,
            validation_data=val_generator,
            validation_steps=step_val,
            callbacks=[
                CSVLogger(CSV_DIR + STAGE_1 + name +"_log.csv", append=True),
                TensorBoard(log_dir=TB_DIR + STAGE_1, write_graph=True),
                EarlyStopping(patience=10, verbose=1),
                MemoryCallback(),
                ModelCheckpoint(WEIGHTS_DIR + STAGE_1 + name + "_{epoch:02d}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
                ModelCheckpoint(BEST_DIR + name + "_Stage1.hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
            ]
        )


        # get time of program end
        end_stage_1 = datetime.now()
        end_stage_1 = end_stage_1.strftime("%d/%m/%Y %H:%M:%S")

        print("Training started at: ", start)
        print("Training stage 1 ended at:   ", end_stage_1)


    # define and compile model
    model = pconv_model(fine_tuning=True, lr=epoch2_lr, image_size=(shape, shape), drop=0.0, 
                        vgg16_weights=VGG16_WEIGHTS, vgg_norm=args.vgg_norm,
                        loss_penalty=args.penalty, loss_area=args.area)
    if args.resume == 2:
        print("Resuming training at stage 2")
        # resume from checkpoint
        if args.checkpoint != None:
            model.load_weights(model_weights)
            print("Successfully loaded model weights from {}\n".format(model_weights))
        else:
            # if no checkpoint provided, try best model from stage 1 of same name
            if os.path.isfile(BEST_DIR + name + "_Stage1.hdf5"):
                model.load_weights(BEST_DIR + name + "_Stage1.hdf5")
                print("Successfully loaded model weights from {}".format(BEST_DIR + name + "_Stage1.hdf5"))
            else:
                print("Failed to find the best model epoch, exitting.....")
                exit()
    else:
        # if no checkpoint provided, try best model from stage 1 of same name
        if os.path.isfile(BEST_DIR + name + "_Stage1.hdf5"):
            model.load_weights(BEST_DIR + name + "_Stage1.hdf5")
            print("Successfully loaded model weights from {}".format(BEST_DIR + name + "_Stage1.hdf5"))
        else:
            print("Failed to find the best model epoch, exitting.....")
            exit()

    if args.run:
    # --------------------------------------------------------------------------------------- #
    #                                        Training                                         #
    # --------------------------------------------------------------------------------------- #
        history = model.fit(
            train_generator,
            steps_per_epoch=step_train,
            epochs=epoch2,
            validation_data=val_generator,
            validation_steps=step_val,
            callbacks=[
                CSVLogger(CSV_DIR + STAGE_2 + name +"_log.csv", append=True),
                TensorBoard(log_dir=TB_DIR + STAGE_2, write_graph=True),
                EarlyStopping(patience=10, verbose=1),
                MemoryCallback(),
                ModelCheckpoint(WEIGHTS_DIR + STAGE_2 + name + "_{epoch:02d}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
                ModelCheckpoint(BEST_DIR + name + "_Stage2.hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
            ]
        )


        # get time of program end
        end_stage_2 = datetime.now()
        end_stage_2 = end_stage_2.strftime("%d/%m/%Y %H:%M:%S")

        print("Training started at: ", start)
        if args.resume != 2:
            print("Training stage 1 ended at:   ", end_stage_1)
        print("Training stage 2 ended at:   ", end_stage_2)


    if args.analysis:
        model = pconv_model(fine_tuning=False, lr=epoch1_lr, image_size=(shape, shape), drop=args.dropout, 
                            vgg16_weights=VGG16_WEIGHTS, vgg_norm=args.vgg_norm,
                            loss_penalty=args.penalty, loss_area=args.area)
        model.summary()

        size = keras_model_memory_usage_in_bytes(model, batch_size=batch_size)

        def format_bytes(b):
            power = 2**1024
            n = 0
            power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
            while b > power:
                b /= power
                n += 1
            
            print("model size: {}{}B".format(b, power_labels[n]))
        
        format_bytes(size)



if __name__ == '__main__':
    main()
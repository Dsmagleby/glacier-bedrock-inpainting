# using PCovn python env

import os
import gc
import glob
import argparse
import numpy as np
import pandas as pd


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Dropout, UpSampling2D, Input
from tensorflow.keras.layers import Conv2D, Concatenate, MaxPooling2D, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split

from Mask_segmentation.libs.unet_model import UNet


parser = argparse.ArgumentParser()

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

parser.add_argument('-i', '--input', dest='input', default=None, help='directory containing test images and full masks')
parser.add_argument('-n', '--name', dest='name', default='mask_segmentation', help='name to save model as')
parser.add_argument('-s', '--shape', dest='shape', type=int, default=256, help='mask shape ex. 256')

# calibrate model behavior
parser.add_argument('-b', '--batch', dest='batch', type=int, default=6, help='model batch size, defualt = 6')
parser.add_argument('-e', '--epochs', dest='epochs', type=int, default=50, help='model # of epochs, defualt = 50')
parser.add_argument('-sv', '--step_val', dest='step_val', type=int, default=15, help='model # of val steps * batch, defualt = 15')
parser.add_argument('-st', '--step_train', dest='step_train', type=int, default=150, help='model # of train steps * batch, defualt = 150')
parser.add_argument('-lr', '--lr', dest='lr', type=float, default=0.0002, help='model optimizer (ADAM) learning rate, default = 0.0002')
parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.5, help='model dropout rate, default = 0.5')
parser.add_argument('-bn', '--batchnorm', dest='batchnorm', default=True, type=str2bool, const=True, nargs='?', help='Use batchnorm, default = True')


WEIGHTS_DIR = "Mask_segmentation/callbacks/weights/"
TB_DIR      = "Mask_segmentation/callbacks/tensorboard/"
CSV_DIR     = "Mask_segmentation/callbacks/csvlogger/"
BEST_DIR    = "Mask_segmentation/callbacks/best_model/"
SEED = 42

def main():
    args = parser.parse_args()

    steps_val = args.step_val * args.batch
    steps_train = args.step_train * args.batch
    lr = args.lr

    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)


    image_paths = []
    mask_paths = []

    for filename in glob.iglob(args.input + 'images/*'):
        image_paths.append(filename)
    
    for filename in glob.iglob(args.input + 'masks_full/*'):
        mask_paths.append(filename)

    df = pd.DataFrame({'image_path': image_paths, 'mask_path': mask_paths})
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    print(len(image_paths), len(mask_paths))


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
        target_size = (args.shape, args.shape),
        batch_size = args.batch,
        seed = SEED
    )
    train_generator_mask = train_datagen.flow_from_dataframe(
        train_df, x_col = 'mask_path',
        target_size = (args.shape, args.shape),
        batch_size = args.batch,
        seed = SEED
    )

    # Create validation generator
    val_datagen = AugmentedDataGenerator()
    val_generator_image = val_datagen.flow_from_dataframe(
        val_df, x_col = 'image_path', 
        target_size = (args.shape, args.shape), 
        batch_size = args.batch,
        seed = SEED 
    )
    val_generator_mask = val_datagen.flow_from_dataframe(
        val_df, x_col = 'mask_path',  
        target_size = (args.shape, args.shape), 
        batch_size = args.batch,
        seed = SEED
    )

    def image_and_mask_generator(image_generator, mask_generator):
        generator = zip(image_generator, mask_generator)
        for (img, mask) in generator:
            
            img[img < 0] = 0
            img_max = np.amax(img)
            img_min = np.amin(img)
            img = (img - img_min) / (img_max - img_min)

            gc.collect()
            yield img, mask


    train_generator = image_and_mask_generator(train_generator_image, train_generator_mask)        
    val_generator   = image_and_mask_generator(val_generator_image, val_generator_mask)


    model = UNet((args.shape, args.shape, 1), start_ch=64, dropout=args.dropout, batchnorm=args.batchnorm)
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics='accuracy')


    class MemoryCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            gc.collect()
            K.clear_session()


    model.fit(
        train_generator,
        steps_per_epoch=steps_train,
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=steps_val,
        callbacks=[
            CSVLogger(CSV_DIR + args.name + "_log.csv", append=True),
            TensorBoard(log_dir=TB_DIR, write_graph=True),
            EarlyStopping(patience=10, verbose=1),
            ModelCheckpoint(BEST_DIR + args.name + ".hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True),
            ModelCheckpoint(WEIGHTS_DIR + "weights_"+ args.name + ".{epoch:02d}-{val_loss:.2f}-{loss:.2f}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
            MemoryCallback()
        ]
    )


    print("Best version of the model have been saved to Mask_segmentation/callbacks/best_model/")



if __name__ == '__main__':
    main()


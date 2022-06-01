import os
import gc
import cv2
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from copy import deepcopy
from datetime import datetime
from sklearn import preprocessing
from skimage.morphology import erosion

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, EarlyStopping

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
 
from ..libs.model_analysis import keras_model_memory_usage_in_bytes
from ..libs.PConv_base import decoder_block, encoder_block
from ..libs.loss_functions import IrregularLoss, SSIM, PSNR, vgg16_feature_model
from ..libs.utils import GlacierMaskGenerator, MaskGenerator


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

parser.add_argument('-c', '--checkpoint', dest='checkpoint', default=None, help='Name of model checkpoint to be loaded')
parser.add_argument('-a', '--analysis', dest='analysis', default=False, type=str2bool, const=True, nargs='?', help='Show model summary and size')
parser.add_argument('-r', '--run', dest='run', default=True, type=str2bool, const=True, nargs='?', help='Run the model')
parser.add_argument('-n', '--name', dest='name', default='pconv_model', type=str, help="Name to be used for outputs (learning_curve, weights etc.)")
parser.add_argument('-d', '--dropout', dest='dropout', type=float, default=0.0, help="Dropout value, default = 0.0")

parser.add_argument('-vgg_norm', '--vgg_norm', dest='vgg_norm', default=False, type=str2bool, const=True, nargs='?', help='Normalise loss function to imagenet specs')
parser.add_argument('-m', '--mask', dest='mask', default='segmented', type=str, help="Mask used for final evaluation")

parser.add_argument('-v', '--val', dest='val', default=False, type=str2bool, const=True, nargs='?', help='Run model on validation subset')
parser.add_argument('-t', '--test', dest='test', default=False, type=str2bool, const=True, nargs='?', help='Run model on test subset')
parser.add_argument('-w', '--weights', dest='weights', default=None, help='Weights to be loaded')
parser.add_argument('-augment_target', '--augment_target', dest='augment_target', default=False, type=str2bool, const=True, nargs='?', help='Augment the target image')
parser.add_argument('-augment_linear', '--augment_linear', dest='augment_linear', default=False, type=str2bool, const=True, nargs='?', help='Augment the target image')
parser.add_argument('-normal_box', '--normal_box', dest='normal_box', default=False, type=str2bool, const=True, nargs='?', help='Augment the target image')

parser.add_argument('-ext', '--extended', dest='ext', default='', type=str, help='dataset selection (256, extended)')
parser.add_argument('-p', '--penalty', dest='penalty', default=False, type=str2bool, const=True, nargs='?', help='Modified penalty loss function')
parser.add_argument('-area', '--area', dest='area', default=False, type=str2bool, const=True, nargs='?', help='Modified area scale formula loss function')

parser.add_argument('-resume', '--resume', dest='resume', default=0, type=int, choices=stages, help="Resume the training at a previous stage")
args = parser.parse_args()


# --------------------------------------------------------------------------------------- #
#                                         Config                                          #
# --------------------------------------------------------------------------------------- #
BATCH_SIZE      = 8
SHAPE           = 256
STEPS_PER_EPOCH = 1000


EPOCHS_STAGE_1  = 70
LR_STAGE_1      = 0.0002
WEIGHTS_DIR     = "callbacks/inpainting/weights/"
STAGE_1         = "initial/"
EPOCHS_STAGE_2  = 50
LR_STAGE_2      = 0.00005
STAGE_2         = "fine_tuning/"

BEST_DIR        = "callbacks/inpainting/best_model/"

STEPS_VAL       = 100
SEED            = 42
DEM_MAX         = 4804
DROPOUT         = args.dropout
WEIGHTS         = args.weights

VGG16_WEIGHTS   = "data/vgg16_pytorch2keras.h5"
NAME            = args.name

TB_DIR          = "callbacks/inpainting/tensorboard/"
CSV_DIR         = "callbacks/inpainting/csvlogger/"
TRAIN_DIR       = "aster_dataset/inpainting/train/"
VAL_DIR         = "aster_dataset/inpainting/val/"
TEST_DIR        = "aster_dataset/inpainting/test/"




if args.checkpoint != None:
    if os.path.isfile(WEIGHTS_DIR + args.checkpoint):
        MODEL_WEIGHTS = WEIGHTS_DIR + args.checkpoint
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


def get_distance_mask(patch):
    new_patch = np.ones_like(patch).astype(float)
    new_patch = np.negative(new_patch)
    #np.seterr(invalid='ignore')

    fill_value = 0
    while True:
        if np.sum(patch) == 0:   
            break

        patch_erosion = erosion(patch)
        patch -= patch_erosion
        new_patch[np.where(patch == 1)] = fill_value
        patch = patch_erosion
        fill_value += 1

    new_patch[new_patch == 0] = 0.01
    new_patch = new_patch/(fill_value-1)
    return new_patch

def box_augment(patch):
    backup = deepcopy(patch)
    distance_mask = get_distance_mask(patch)
    func = lambda x: 26.145*x + 43.075
    target_height = func(distance_mask)
    outside = np.where(backup == 0)
    target_height[outside] = 0
    return target_height
# --------------------------------------------------------------------------------------- #
#                                      Data Loader                                        #
# --------------------------------------------------------------------------------------- #

train_paths = []
train_mask_paths = []
val_paths = []
val_mask_paths = []

target_name = '*_v2'
if args.augment_target:
    if args.normal_box:
        target_name = '*_v2_augmented_box'
    else:
        target_name = '*_v2_augmented'
elif args.augment_linear:
    if args.normal_box:
        target_name = '*_v2_linear_box'
    else:
        target_name = '*_v2_linear_v2'

for filename in glob.iglob('aster_dataset/inpainting/train/images_extended/' + target_name + '/*'):
    train_paths.append(filename)
    filename = filename.replace('images', 'masks')
    if args.augment_target:
        if args.normal_box:
            filename = filename.replace('v2_augmented_box', 'v2_box')
        else:
            filename = filename.replace('v2_augmented', 'v2')
    if args.augment_linear:
        if args.normal_box:
            filename = filename.replace('v2_linear_box', 'v2_box')
        else:
            filename = filename.replace('v2_linear_v2', 'v2')
    train_mask_paths.append(filename.replace('.tif', '_mask.tif'))

for filename in glob.iglob('aster_dataset/inpainting/val/images_extended/' + target_name + '/*'):
    val_paths.append(filename)
    filename = filename.replace('images', 'masks')
    if args.augment_target:
        if args.normal_box:
            filename = filename.replace('v2_augmented_box', 'v2_box')
        else:
            filename = filename.replace('v2_augmented', 'v2')
    if args.augment_linear:
        if args.normal_box:
            filename = filename.replace('v2_linear_box', 'v2_box')
        else:
            filename = filename.replace('v2_linear_v2', 'v2')
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
        target_size = (SHAPE, SHAPE),
        batch_size = BATCH_SIZE,
        seed = SEED
    )
    train_generator_mask = train_datagen.flow_from_dataframe(
        train_df, x_col = 'mask_path',
        target_size = (SHAPE, SHAPE),
        batch_size = BATCH_SIZE,
        seed = SEED
    )

    # Create validation generator
    val_datagen = AugmentedDataGenerator()
    val_generator_image = val_datagen.flow_from_dataframe(
        val_df, x_col = 'image_path', 
        target_size = (SHAPE, SHAPE), 
        batch_size = BATCH_SIZE, 
        seed = SEED 
    )
    val_generator_mask = val_datagen.flow_from_dataframe(
        val_df, x_col = 'mask_path',  
        target_size = (SHAPE, SHAPE), 
        batch_size = BATCH_SIZE, 
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
        MaskGenerator(SHAPE, SHAPE, 128, 128),
        x_col = 'image_path',
        target_size=(SHAPE, SHAPE),
        batch_size=BATCH_SIZE
    )

    val_datagen = AugmentedDataGenerator(
        horizontal_flip=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        val_df, 
        MaskGenerator(SHAPE, SHAPE, 128, 128),
        x_col = 'image_path',
        target_size=(SHAPE, SHAPE),
        batch_size=BATCH_SIZE,
        seed = SEED
    )

else:
    print("Invalid mask algorithm choosen, options: (segmented, box)")
    exit()

# --------------------------------------------------------------------------------------- #
#                                   Data Loader Test                                      #
# --------------------------------------------------------------------------------------- #
val_data = next(val_generator)
(masked, mask, _), ori = val_data

_, axes = plt.subplots(ori.shape[0], 3, figsize=(20, 20))
for i in range(len(ori)):
    axes[i,1].imshow(mask[i,:,:,:],  cmap='terrain')
    axes[i,2].imshow(masked[i,:,:,:], cmap='terrain', vmin=0)
    axes[i,0].imshow(ori[i,:,:,:],   cmap='terrain', vmin=0)
    axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[i,2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    
axes[0,0].set_title('Original image')
axes[0,1].set_title('Mask')
axes[0,2].set_title('Masked image')
    
plt.tight_layout()
plt.savefig("figures/" + NAME + "_dataloader_test.png")


# --------------------------------------------------------------------------------------- #
#                                         Model                                           #
# --------------------------------------------------------------------------------------- #
def pconv_model(fine_tuning=False, lr=0.0002, predict_only=False, image_size=(256, 256), drop=0, vgg16_weights='imagenet'):
    """Inpainting model."""

    img_input  = Input(shape=(image_size[0], image_size[1], 1), name='input_img')
    mask_input = Input(shape=(image_size[0], image_size[1], 1), name='input_mask')
    y_true     = Input(shape=(image_size[0], image_size[1], 1), name='y_true')
    
    # Encoder:
    # --------
    e_img_1, e_mask_1 = encoder_block(img_input, mask_input, 32, 7, drop=0, batch_norm=False, count='1')
    e_img_2, e_mask_2 = encoder_block(e_img_1, e_mask_1, 64, 5, drop=drop, freeze_bn=fine_tuning, count='2')
    e_img_3, e_mask_3 = encoder_block(e_img_2, e_mask_2, 128, 5, drop=drop, freeze_bn=fine_tuning, count='3')
    e_img_4, e_mask_4 = encoder_block(e_img_3, e_mask_3, 256, 3, drop=drop, freeze_bn=fine_tuning, count='4')
    e_img_5, e_mask_5 = encoder_block(e_img_4, e_mask_4, 256, 3, drop=drop, freeze_bn=fine_tuning, count='5')
    e_img_6, e_mask_6 = encoder_block(e_img_5, e_mask_5, 256, 3, drop=drop, freeze_bn=fine_tuning, count='6')
    e_img_7, e_mask_7 = encoder_block(e_img_6, e_mask_6, 256, 3, drop=drop, freeze_bn=fine_tuning, count='7')
    e_img_8, e_mask_8 = encoder_block(e_img_7, e_mask_7, 256, 3, drop=drop, freeze_bn=fine_tuning, count='8')

    # Decoder:
    # --------
    d_img_9, d_mask_9   = decoder_block(e_img_8, e_mask_8, e_img_7, e_mask_7, 256, drop=drop, count='9')
    d_img_10, d_mask_10 = decoder_block(d_img_9, d_mask_9, e_img_6, e_mask_6, 256, drop=drop, count='10')
    d_img_11, d_mask_11 = decoder_block(d_img_10, d_mask_10, e_img_5, e_mask_5, 256, drop=drop, count='11')
    d_img_12, d_mask_12 = decoder_block(d_img_11, d_mask_11, e_img_4, e_mask_4, 256, drop=drop, count='12')
    d_img_13, d_mask_13 = decoder_block(d_img_12, d_mask_12, e_img_3, e_mask_3, 128, drop=drop, count='13')
    d_img_14, d_mask_14 = decoder_block(d_img_13, d_mask_13, e_img_2, e_mask_2, 64, drop=drop, count='14')
    d_img_15, d_mask_15 = decoder_block(d_img_14, d_mask_14, e_img_1, e_mask_1, 32, drop=drop, count='15')
    d_img_16 = decoder_block(d_img_15, d_mask_15, img_input, mask_input, 1, last_layer=True, count='16')
    
    model = Model(inputs=[img_input, mask_input, y_true], outputs=d_img_16)

    # This will also freeze bn parameters `beta` and `gamma`: 
    if fine_tuning:
        for l in model.layers:
            if 'bn_enc' in l.name:
                l.trainable = False
    
    if predict_only:
        return model

    vgg_model = vgg16_feature_model(['block1_pool', 'block2_pool', 'block3_pool'], weights=vgg16_weights)
    model.add_loss(IrregularLoss(mask_input, vgg_model, args.vgg_norm, args.penalty, args.area)(y_true, d_img_16))
    #model.add_metric(positive_ratio(mask_input, y_true, d_img_16), name='PositiveRatio')
    model.compile(Adam(learning_rate=lr), metrics=[SSIM, PSNR])

    return model

# get time of program start
start = datetime.now()
start = start.strftime("%d/%m/%Y %H:%M:%S")


# define and compile model
model = pconv_model(fine_tuning=False, lr=LR_STAGE_1, image_size=(SHAPE, SHAPE), drop=DROPOUT, vgg16_weights=VGG16_WEIGHTS)
if args.resume == 1:
    if args.checkpoint != None:
        model.load_weights(MODEL_WEIGHTS)
        print("Successfully loaded model weights from {}\n".format(MODEL_WEIGHTS))
    else:
        print("Checkpoint was not provided, can not resume training at stage 1")

if args.run and (args.resume == 0 or args.resume == 1):
# --------------------------------------------------------------------------------------- #
#                                        Training                                         #
# --------------------------------------------------------------------------------------- #
    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS_STAGE_1,
        validation_data=val_generator,
        validation_steps=STEPS_VAL,
        callbacks=[
            CSVLogger(CSV_DIR + STAGE_1 + NAME +"_log.csv", append=True),
            TensorBoard(log_dir=TB_DIR + STAGE_1, write_graph=True),
            EarlyStopping(patience=10, verbose=1),
            MemoryCallback(),
            ModelCheckpoint(WEIGHTS_DIR + STAGE_1 + NAME + "_{epoch:02d}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
            ModelCheckpoint(BEST_DIR + NAME + "_Stage1.hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
        ]
    )

    plt.figure(figsize=(8,8))
    plt.title("Learning curve")
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="Best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('data/learning_curve/'+ NAME +'stage_1_learning_curve.png')


    (input_img, mask, _), orig_img = next(val_generator)
    output_img = model.predict([input_img, mask, orig_img])
    output_img[mask == 1] = input_img[mask == 1]


    # Show side by side
    _, axes = plt.subplots(input_img.shape[0], 3, figsize=(20, 20))
    for i in range(len(orig_img)):
        axes[i,1].imshow(input_img[i,:,:,:],  cmap='terrain', vmin=0)
        axes[i,2].imshow(output_img[i,:,:,:], cmap='terrain', vmin=0)
        axes[i,0].imshow(orig_img[i,:,:,:],   cmap='terrain', vmin=0)
        axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[i,2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('Masked image')
    axes[0,2].set_title('Prediction')
        
    plt.tight_layout()
    plt.savefig("figures/" + NAME + "_stage_1.png")

    # get time of program end
    end_stage_1 = datetime.now()
    end_stage_1 = end_stage_1.strftime("%d/%m/%Y %H:%M:%S")

    print("Training started at: ", start)
    print("Training stage 1 ended at:   ", end_stage_1)


# define and compile model
model = pconv_model(fine_tuning=True, lr=LR_STAGE_2, image_size=(SHAPE, SHAPE), drop=0.0, vgg16_weights=VGG16_WEIGHTS)
if args.resume == 2:
    print("Resuming training at stage 2")
    # resume from checkpoint
    if args.checkpoint != None:
        model.load_weights(MODEL_WEIGHTS)
        print("Successfully loaded model weights from {}\n".format(MODEL_WEIGHTS))
    else:
        # if no checkpoint provided, try best model from stage 1 of same name
        if os.path.isfile(BEST_DIR + NAME + "_Stage1.hdf5"):
            model.load_weights(BEST_DIR + NAME + "_Stage1.hdf5")
            print("Successfully loaded model weights from {}".format(BEST_DIR + NAME + "_Stage1.hdf5"))
        else:
            print("Failed to find the best model epoch, exitting.....")
            exit()
else:
    # if no checkpoint provided, try best model from stage 1 of same name
    if os.path.isfile(BEST_DIR + NAME + "_Stage1.hdf5"):
        model.load_weights(BEST_DIR + NAME + "_Stage1.hdf5")
        print("Successfully loaded model weights from {}".format(BEST_DIR + NAME + "_Stage1.hdf5"))
    else:
        print("Failed to find the best model epoch, exitting.....")
        exit()

if args.run:
# --------------------------------------------------------------------------------------- #
#                                        Training                                         #
# --------------------------------------------------------------------------------------- #
    history = model.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=EPOCHS_STAGE_2,
        validation_data=val_generator,
        validation_steps=STEPS_VAL,
        callbacks=[
            CSVLogger(CSV_DIR + STAGE_2 + NAME +"_log.csv", append=True),
            TensorBoard(log_dir=TB_DIR + STAGE_2, write_graph=True),
            EarlyStopping(patience=10, verbose=1),
            MemoryCallback(),
            ModelCheckpoint(WEIGHTS_DIR + STAGE_2 + NAME + "_{epoch:02d}.hdf5", monitor="val_loss", verbose=1, save_weights_only=True),
            ModelCheckpoint(BEST_DIR + NAME + "_Stage2.hdf5", monitor="val_loss", verbose=1, save_weights_only=True, save_best_only=True)
        ]
    )

    plt.figure(figsize=(8,8))
    plt.title("Learning curve")
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r", label="Best model")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('data/learning_curve/'+ NAME +'stage_2_learning_curve.png')


    (input_img, mask, _), orig_img = next(val_generator)
    output_img = model.predict([input_img, mask, orig_img])
    output_img[mask == 1] = input_img[mask == 1]


    # Show side by side
    _, axes = plt.subplots(input_img.shape[0], 3, figsize=(20, 20))
    for i in range(len(orig_img)):
        axes[i,1].imshow(input_img[i,:,:,:],  cmap='terrain', vmin=0)
        axes[i,2].imshow(output_img[i,:,:,:], cmap='terrain', vmin=0)
        axes[i,0].imshow(orig_img[i,:,:,:],   cmap='terrain', vmin=0)
        axes[i,0].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[i,1].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        axes[i,2].tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
        
    axes[0,0].set_title('Original image')
    axes[0,1].set_title('Masked image')
    axes[0,2].set_title('Prediction')
        
    plt.tight_layout()
    plt.savefig("figures/" + NAME + "_stage_2.png")

    # get time of program end
    end_stage_2 = datetime.now()
    end_stage_2 = end_stage_2.strftime("%d/%m/%Y %H:%M:%S")

    print("Training started at: ", start)
    if args.resume != 2:
        print("Training stage 1 ended at:   ", end_stage_1)
    print("Training stage 2 ended at:   ", end_stage_2)






if args.val or args.test:
    model = pconv_model(fine_tuning=True, lr=LR_STAGE_2, image_size=(SHAPE, SHAPE), drop=0.0, vgg16_weights=VGG16_WEIGHTS)
    if os.path.isfile(WEIGHTS):
        model.load_weights(WEIGHTS)
        print("\nSuccessfully loaded model weights from {}\n".format(WEIGHTS))
    else:
        print("\nFailed to loaded model weights from {}\n".format(WEIGHTS))
        exit()


    if args.test:

        RGI_burned = ['RGI60-11.00562', 'RGI60-11.00590', 'RGI60-11.00603', 'RGI60-11.00638', 'RGI60-11.00647', 
                      'RGI60-11.00689', 'RGI60-11.00695', 'RGI60-11.00846', 'RGI60-11.00950', 'RGI60-11.01024', 
                      'RGI60-11.01041', 'RGI60-11.01067', 'RGI60-11.01144', 'RGI60-11.01199', 'RGI60-11.01296', 
                      'RGI60-11.01344', 'RGI60-11.01367', 'RGI60-11.01376', 'RGI60-11.01473', 'RGI60-11.01509', 
                      'RGI60-11.01576', 'RGI60-11.01604', 'RGI60-11.01698', 'RGI60-11.01776', 'RGI60-11.01786', 
                      'RGI60-11.01790', 'RGI60-11.01791', 'RGI60-11.01806', 'RGI60-11.01813', 'RGI60-11.01840', 
                      'RGI60-11.01857', 'RGI60-11.01894', 'RGI60-11.01928', 'RGI60-11.01962', 'RGI60-11.01986', 
                      'RGI60-11.02006', 'RGI60-11.02024', 'RGI60-11.02027', 'RGI60-11.02244', 'RGI60-11.02249', 
                      'RGI60-11.02261', 'RGI60-11.02448', 'RGI60-11.02490', 'RGI60-11.02507', 'RGI60-11.02549', 
                      'RGI60-11.02558', 'RGI60-11.02583', 'RGI60-11.02584', 'RGI60-11.02596', 'RGI60-11.02600', 
                      'RGI60-11.02624', 'RGI60-11.02673', 'RGI60-11.02679', 'RGI60-11.02704', 'RGI60-11.02709', 
                      'RGI60-11.02715', 'RGI60-11.02740', 'RGI60-11.02745', 'RGI60-11.02755', 'RGI60-11.02774', 
                      'RGI60-11.02775', 'RGI60-11.02787', 'RGI60-11.02796', 'RGI60-11.02864', 'RGI60-11.02884', 
                      'RGI60-11.02890', 'RGI60-11.02909', 'RGI60-11.03249']

        for i in range(len(RGI_burned)):
            orig_img = cv2.imread('aster_dataset/inpainting/test/images/' + RGI_burned[i] + '.tif', cv2.IMREAD_UNCHANGED)
            mask = cv2.imread('aster_dataset/inpainting/test/masks/' + RGI_burned[i] + '_mask.tif', cv2.IMREAD_UNCHANGED)

            img_max = np.amax(orig_img)
            img_min = np.amin(orig_img)
            orig_img = (orig_img - img_min) / (img_max - img_min)

            input_img = deepcopy(orig_img)
            input_img[mask==1] = 0
            mask = 1 - mask
            
            input_img = input_img[np.newaxis, ...]
            mask = mask[np.newaxis, ...]
            orig_img = orig_img[np.newaxis, ...]
                    
            output_img = model.predict([input_img, mask, orig_img])

            output_denorm = (output_img * (img_max - img_min) + img_min)

            cv2.imwrite('aster_dataset/inpainting/prediction/test/' + RGI_burned[i] + '_' + NAME + '.tif', output_denorm.squeeze().astype(np.uint16))


    if args.val:

        val_images = []
        for filename in glob.iglob('aster_dataset/inpainting/val/image_256/' + '*'):
            val_images.append(filename)
        
        validation_images = val_images[:50]
        print(validation_images[39:-4])

        mask_generator = GlacierMaskGenerator(SHAPE, SHAPE, 1)

        for i in range(len(validation_images)):
            orig_img = cv2.imread('aster_dataset/inpainting/val/image_256/' + validation_images[i][39:-4] + '.tif', cv2.IMREAD_UNCHANGED)
            if args.mask == 'segmented':
                mask = cv2.imread('aster_dataset/inpainting/val/mask_256/' + validation_images[i][39:-4] + '_mask.tif', cv2.IMREAD_UNCHANGED)
            else:
                mask = mask_generator.sample(
                    0.075, 196, 
                    np.random.randint(4,8),
                    np.random.uniform(0.1, 0.9),
                    np.random.uniform(0.01, 0.5), 
                    SEED
                ).squeeze()

            img_max = np.amax(orig_img)
            img_min = np.amin(orig_img)
            orig_img = (orig_img - img_min) / (img_max - img_min)

            input_img = deepcopy(orig_img)
            if args.mask == 'segmented':
                input_img[mask==1] = 0
                mask = 1 - mask
            else:
                input_img[mask==0] = 0
            
            input_img = input_img[np.newaxis, ...]
            mask = mask[np.newaxis, ...]
            orig_img = orig_img[np.newaxis, ...]
                    
            output_img = model.predict([input_img, mask, orig_img])
            output_denorm = (output_img * (img_max - img_min) + img_min)

            cv2.imwrite('aster_dataset/inpainting/prediction/val/' + validation_images[i][39:-4] + '_' + NAME + '_' + args.mask + '.tif', output_denorm.squeeze().astype(np.uint16))
            if args.mask != 'segmented':
                cv2.imwrite('aster_dataset/inpainting/prediction/bezier_masks/' + validation_images[i][39:-4] + '_' + NAME + '_bezier_mask.tif', output_denorm.squeeze().astype(np.uint16))


if args.analysis:
    model = pconv_model(fine_tuning=False, lr=LR_STAGE_1, image_size=(SHAPE, SHAPE), drop=DROPOUT, vgg16_weights=VGG16_WEIGHTS)
    model.summary()

    size = keras_model_memory_usage_in_bytes(model, batch_size=BATCH_SIZE)

    def format_bytes(b):
        power = 2**1024
        n = 0
        power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
        while b > power:
            b /= power
            n += 1
        
        print("model size: {}{}B".format(b, power_labels[n]))
    
    format_bytes(size)
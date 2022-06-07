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
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger, Callback, EarlyStopping

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
 
from ..libs.PConv_base import decoder_block, encoder_block
from ..libs.loss_functions import IrregularLoss, SSIM, PSNR, vgg16_feature_model


# --------------------------------------------------------------------------------------- #
#                                    Argument Parsing                                     #
# --------------------------------------------------------------------------------------- #
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

parser.add_argument('-n', '--name', dest='name', default='pconv_model', type=str, help="Name to be used for outputs (learning_curve, weights etc.)")
parser.add_argument('-vgg_norm', '--vgg_norm', dest='vgg_norm', default=False, type=str2bool, const=True, nargs='?', help='Normalise loss function to imagenet specs')
parser.add_argument('-w', '--weights', dest='weights', default=None, help='Weights to be loaded')
parser.add_argument('-i', '--img', dest='img', default=None, type=str, help='input image')
parser.add_argument('-m', '--mask', dest='mask', default=None, type=str, help='input mask')
parser.add_argument('-o', '--output_dir', dest='output_dir', default=None, type=str, help='output dir if --all=True')
parser.add_argument('-b', '--burn', dest='burn', default=False, type=str2bool, const=True, nargs='?', help='Run all burned glaciers')
parser.add_argument('-a', '--all', dest='all', default=False, type=str2bool, const=True, nargs='?', help='Run all glaciers')
args = parser.parse_args()


# --------------------------------------------------------------------------------------- #
#                                         Config                                          #
# --------------------------------------------------------------------------------------- #
BATCH_SIZE      = 8
SHAPE           = 256
STEPS_PER_EPOCH = 1000
WEIGHTS         = args.weights
EPOCHS_STAGE_2  = 50
LR_STAGE_2      = 0.00005
VGG16_WEIGHTS   = "data/vgg16_pytorch2keras.h5"
NAME            = args.name

IMG             = args.img
MASK            = args.mask
if args.all:
    if IMG[:-1] != '/':
        IMG = ''.join((IMG, '/'))
    if MASK[:-1] != '/':
        MASK = ''.join((MASK, '/'))
    
    OUTDIR          = args.output_dir
    if OUTDIR[:-1] != '/':
        OUTDIR = ''.join((OUTDIR, '/'))


# enable memory growth
physical_devices = tf.config.list_physical_devices('GPU') 
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        K.clear_session()


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
    model.add_loss(IrregularLoss(mask_input, vgg_model, args.vgg_norm, False, False)(y_true, d_img_16))
    model.compile(Adam(learning_rate=lr), metrics=[SSIM, PSNR])

    return model



model = pconv_model(fine_tuning=True, lr=LR_STAGE_2, image_size=(SHAPE, SHAPE), drop=0.0, vgg16_weights=VGG16_WEIGHTS)
if os.path.isfile(WEIGHTS):
    model.load_weights(WEIGHTS)
    print("\nSuccessfully loaded model weights from {}\n".format(WEIGHTS))
else:
    print("\nFailed to loaded model weights from {}\n".format(WEIGHTS))
    exit()

if args.burn:

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


elif args.all:

    image_paths, mask_paths = [], []
    for filename in glob.iglob(IMG + '*'):
        image_paths.append(filename)
    for filename in glob.iglob(MASK + '*'):
        mask_paths.append(filename)
    print("# images: {} and # masks: {}".format(len(image_paths), len(mask_paths)))


    for i in tqdm(range(len(image_paths))):
        orig_img = cv2.imread(image_paths[i], cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_UNCHANGED)

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

        cv2.imwrite(OUTDIR + image_paths[i][-18:], output_denorm.squeeze().astype(np.uint16))


else:
    if os.path.isfile(IMG):
        orig_img = cv2.imread(IMG, cv2.IMREAD_UNCHANGED)
    else:
        print(IMG, " does not exist")

    if os.path.isfile(MASK):
        mask = cv2.imread(MASK, cv2.IMREAD_UNCHANGED)
    else:
        print(MASK, " does not exist")

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

    cv2.imwrite('aster_dataset/inpainting/prediction/single_prediction/' +  NAME + '.tif', output_denorm.squeeze().astype(np.uint16))
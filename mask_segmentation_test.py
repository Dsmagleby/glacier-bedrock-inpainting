import os
import gc
import cv2
import glob
import argparse

# import these here because of gdal partial import error
# if loaded from libs
import pandas as pd
import rasterio as rio
import numpy as np
import rioxarray
import xarray as xr

from tqdm import tqdm
from tensorflow.keras.optimizers import Adam


from Pconv.libs.utils import find_maximum
from Mask_segmentation.libs.unet_model import UNet


parser = argparse.ArgumentParser(description='Create segmented training masks')
def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

parser.add_argument("--input",  type=str, default=None, help="path to the training images")
parser.add_argument("--outdir", type=str, default="dataset/", help="path for the output file")
parser.add_argument('--shape', dest='shape', type=int, default=256, help='mask shape ex. 256')
parser.add_argument('--weights', dest='weights', default=None, help='Weights to be loaded')

parser.add_argument('--dropout', dest='dropout', type=float, default=0.5, help='model dropout rate, default = 0.5')
parser.add_argument('--batchnorm', dest='batchnorm', default=True, type=str2bool, const=True, nargs='?', help='Use batchnorm, default = True')


def create_masks(model, image_paths, shape): 
    remove = []

    for image in tqdm(image_paths):
        try:
            patch = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            patch = patch[np.newaxis,..., np.newaxis]

            ridges = find_maximum(patch.squeeze(), shape, iter=3)
            ridges = ridges[...,np.newaxis]

            patch[patch < 0] = 0
            img_max = np.amax(patch)
            img_min = np.amin(patch)
            patch = (patch - img_min) / (img_max - img_min)
            mask_orig = model.predict(patch)
            
            multiplier = 0.5
            
            while True:
                save = True
                mask = np.where(mask_orig > np.max(mask_orig)*multiplier, 1, 0)
                mask = mask - ridges
                mask = np.where(mask < 0, 0, mask)           
                mask = mask[0].squeeze()
                

                #check that not too much of the border is masked
                border = np.concatenate((mask[:8].ravel(), mask[-8:].ravel(), mask[:,-8:].ravel(), mask[:,:8].ravel()))
                if np.sum(border) > 2000:
                    save = False
            
                #remove close border
                mask[:8] = 0
                mask[-8:] = 0
                mask[:,-8:] = 0
                mask[:,:8] = 0

                if np.sum(mask) < 50:
                    save = False   

                elif (np.sum(mask) / (shape*shape)) > 0.50:
                    save = False
                
                mask = mask[..., np.newaxis]
                
                if save:
                    cv2.imwrite(image.replace('images', 'masks')[:-4] + "_mask.tif", mask.astype(np.float32))
                    break
                
                if multiplier < 0.05:
                    remove.append(image)
                    break
                    
                multiplier -= 0.005
        except:
            remove.append(image)
   # list of images, where no reasonable mask can be made
    return remove


def main():
    args = parser.parse_args()

    shape = args.shape
    weights = args.weights
    dropout = args.dropout
    batchnorm = args.batchnorm

    train_path = args.outdir + 'train/'
    val_path = args.outdir + 'val/'

    if os.path.isdir(train_path):
        if os.path.isdir(train_path + 'masks/'):
            None
        else:
            os.mkdir(train_path + 'masks/')


    if os.path.isdir(val_path):
        if os.path.isdir(val_path + 'masks/'):
            None
        else:
            os.mkdir(val_path + 'masks/')


    # we'll only create 1 channel masks, if we need 3 channels, that is handled when needed.
    mask_model = UNet((shape, shape, 1), start_ch=64, dropout=dropout, batchnorm=batchnorm)
    mask_model.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics='accuracy')
    mask_model.load_weights(weights)


    image_paths = []
    for filename in glob.iglob(train_path + 'images/*'):
        image_paths.append(filename)
    train_remove = create_masks(mask_model, image_paths, shape)

    image_paths = []
    for filename in glob.iglob(val_path + 'images/*'):
        image_paths.append(filename)
    val_remove = create_masks(mask_model, image_paths, shape)


    for file in train_remove:
        if os.path.exists(file):
            os.remove(file)
    
    for file in val_remove:
        if os.path.exists(file):
            os.remove(file)



if __name__ == '__main__':
    main()
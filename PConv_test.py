import os
import gc
import cv2
import glob
import argparse
import numpy as np

from copy import deepcopy
from tqdm import tqdm

import tensorflow as tf
 
from Pconv.libs.pconv_model import pconv_model


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
parser.add_argument('-vgg_norm', '--vgg_norm', dest='vgg_norm', default=True, type=str2bool, const=True, nargs='?', help='Normalise loss function to imagenet specs')
parser.add_argument('-w', '--weights', dest='weights', default=None, help='Weights to be loaded')
parser.add_argument('-i', '--img', dest='img', default=None, type=str, help='input image')
parser.add_argument('-m', '--mask', dest='mask', default=None, type=str, help='input mask')
parser.add_argument('-o', '--output_dir', dest='output_dir', default=None, type=str, help='output dir if --all=True')
parser.add_argument('-b', '--burn', dest='burn', default=False, type=str2bool, const=True, nargs='?', help='Run all burned glaciers')
parser.add_argument('-a', '--all', dest='all', default=False, type=str2bool, const=True, nargs='?', help='Run all glaciers')

# model calibration
parser.add_argument('--shape', dest='shape', default=256, type=int, help='input shape')
parser.add_argument('--epoch_lr', dest='epoch_lr', default=0.00005, type=float, help='model learning rate')

VGG16_WEIGHTS   = "Pconv/data/vgg16_pytorch2keras.h5"


def main():
    args = parser.parse_args()
    # --------------------------------------------------------------------------------------- #
    #                                         Config                                          #
    # --------------------------------------------------------------------------------------- #
    shape           = args.shape
    weights         = args.weights
    epoch_lr        = args.epoch_lr
    name            = args.name

    img             = args.img
    mask            = args.mask
    if args.all:
        if img[:-1] != '/':
            img = ''.join((img, '/'))
        if mask[:-1] != '/':
            mask = ''.join((mask, '/'))
        
        outdir          = args.output_dir
        if outdir[:-1] != '/':
            outdir = ''.join((outdir, '/'))
    
    if os.path.isdir(outdir):
        None
    else:
        os.mkdir(outdir)


    # enable memory growth
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # --------------------------------------------------------------------------------------- #
    #                                         Model                                           #
    # --------------------------------------------------------------------------------------- #
    model = pconv_model(fine_tuning=True, lr=epoch_lr, image_size=(shape, shape), drop=0.0, vgg16_weights=VGG16_WEIGHTS)
    if os.path.isfile(weights):
        model.load_weights(weights)
        print("\nSuccessfully loaded model weights from {}\n".format(weights))
    else:
        print("\nFailed to loaded model weights from {}\n".format(weights))
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
            orig_img = cv2.imread(img + RGI_burned[i] + '.tif', cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(img + RGI_burned[i] + '_mask.tif', cv2.IMREAD_UNCHANGED)

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

            cv2.imwrite(outdir + RGI_burned[i] + '_' + name + '.tif', output_denorm.squeeze().astype(np.uint16))


    elif args.all:

        image_paths, mask_paths = [], []
        for filename in glob.iglob(img + '*'):
            image_paths.append(filename)
        for filename in glob.iglob(mask + '*'):
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

            cv2.imwrite(outdir + image_paths[i][-18:], output_denorm.squeeze().astype(np.uint16))


    else:
        if os.path.isfile(img):
            orig_img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        else:
            print(img, " does not exist")

        if os.path.isfile(mask):
            mask = cv2.imread(mask, cv2.IMREAD_UNCHANGED)
        else:
            print(mask, " does not exist")

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

        cv2.imwrite(outdir + img[-18:] + '_' + name + '.tif', output_denorm.squeeze().astype(np.uint16))




if __name__ == '__main__':
    main()
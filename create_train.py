import os
import argparse

import rasterio as rio

from oggm import utils
from tqdm import tqdm

<<<<<<< HEAD:create_train.py
from Pconv.libs.set_generation import  flow_train_dataset
=======
from Pconv.libs.utils import coords_to_xy, contains_glacier
from Pconv.libs.set_generation import create_test_images_from_glacier_center, flow_train_dataset
>>>>>>> ff235f8e07649933f6b15782f11857c141ef0ab7:dataset/create_train.py



parser = argparse.ArgumentParser(description='Create training images')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM tile folder")
parser.add_argument("--outdir", type=str, default="dataset/", help="path for the output file")

# currently not used
parser.add_argument("--region",  type=int, default=11, help="RGI region")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")

# adjust naming and sampling behavior 
parser.add_argument("--postfix",  type=str, default='a', help="postfix added behind samples")
parser.add_argument("--max_height",  type=int, default=4800, help="Max desired height of training samples")
parser.add_argument("--average_height",  type=int, default=2187, help="Sampling average 128x128 center box condition")
parser.add_argument("--samples",  type=int, default=25000, help="Number of samples to attempt to create")


def main():

    args = parser.parse_args()
    
    train_path = args.outdir + 'train/'
    val_path = args.outdir + 'val/'

    if os.path.isdir(train_path):
        if os.path.isdir(train_path + 'images/'):
            None
        else:
            os.mkdir(train_path + 'images/')
    else:
        os.mkdir(train_path)
        os.mkdir(train_path + 'images/')

    if os.path.isdir(val_path):
        if os.path.isdir(val_path + 'images/'):
            None
        else:
            os.mkdir(val_path + 'images/')
    else:
        os.mkdir(val_path)
        os.mkdir(val_path + 'images/')

    # load DEM
    dem = rio.open(args.input, 'r').read(1)
    empty = rio.open(args.input.replace('.tif', '_mask.tif'), 'r').read(1)

    print("Attempting to create {} samples".format(args.samples))

    _ = flow_train_dataset(dem, empty, (args.shape, args.shape), train_path + 'images/', val_path + 'images/', args.max_height, 
                           args.average_height, 'average', args.samples, args.postfix)




if __name__ == '__main__':
    main()
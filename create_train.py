import os
import glob
import cv2
import argparse

import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np

from oggm import utils
from tqdm import tqdm

from ..libs.utils import coords_to_xy, contains_glacier
from ..libs.set_generation import create_test_images_from_glacier_center, flow_train_dataset



parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM tile folder")
parser.add_argument("--outdir", type=str, default="DEM_files/mosaic/mosaic.tif", help="path for the output file")

parser.add_argument("--region",  type=int, default=11, help="RGI region")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")

parser.add_argument("--postfix",  type=str, default='a', help="postfix added behind samples")
parser.add_argument("--max_height",  type=int, default=4800, help="Max desired height of training samples")
parser.add_argument("--average_height",  type=int, default=2187, help="Sampling average 128x128 center box condition")
parser.add_argument("--samples",  type=int, default=25000, help="Number of samples to attempt to create")


def main():

    args = parser.parse_args()

    if os.path.isdir(args.outdir):
        None
    else:
        os.mkdir(args.outdir)

    # fetch RGI
    utils.get_rgi_dir(version=args.version)
    eu = utils.get_rgi_region_file(args.region, version=args.version)
    gdf = gpd.read_file(eu)

    # load DEM
    dem = rio.open(args.input, 'r').read(1)
    empty = np.zeros_like(dem)

    print("Attempting to create {} samples".format(args.samples))
    train_path = args.outdir + 'train/images/'
    val_path = args.outdir + 'val/images/'
    _ = flow_train_dataset(dem, empty, (args.shape, args.shape), train_path, val_path, args.max_height, 
                           args.average_height, 'average', args.samples, args.postfix)




if __name__ == '__main__':
    main()
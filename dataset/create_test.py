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
from ..libs.set_generation import create_test_images_from_glacier_center, flow_train_images



parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM tile folder")
parser.add_argument("--outdir", type=str, default="DEM_files/mosaic/mosaic.tif", help="path for the output file")

parser.add_argument("--region",  type=int, default=11, help="RGI region")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")


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

    # sort glaciers
    glacier_frame = contains_glacier(dem, gdf, 0)
    glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
    boolean_series = gdf['RGIId'].isin(glaciers_alps)
    filtered_gdf = gdf[boolean_series]
    filtered_gdf = filtered_gdf.reset_index()

    # convert lat/lon to x/y for images
    coords, RGI = coords_to_xy('merged/mosaic_aster_rgi13b.tif', filtered_gdf)
    coords_frame = pd.DataFrame({'RGIId': RGI, 'rows': coords[:,0], 'cols': coords[:,1]})

    print("Creating test patches and masks:")
    patches, masks, RGIId = create_test_images_from_glacier_center(dem, empty, (args.shape, args.shape), 
                                                                   coords_frame, create_blank=True)

    print("Saving test patches and masks:")
    for i in tqdm(range(len(RGIId))):
        cv2.imwrite(args.outdir + 'images/' + RGIId[i] + '.tif', patches[i].astype(np.uint16))
        cv2.imwrite(args.outdir + 'masks/' + RGIId[i] +'_mask.tif', masks[i].astype(np.float32))


if __name__ == '__main__':
    main()
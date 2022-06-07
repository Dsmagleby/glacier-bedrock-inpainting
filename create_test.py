import os
import glob
import cv2
import gc
import argparse

import geopandas as gpd
import pandas as pd
import rasterio as rio
import numpy as np
import rioxarray
import xarray as xr

from oggm import utils
from tqdm import tqdm
from shapely.geometry import mapping
from shapely.wkt import loads

from Pconv.libs.utils import coords_to_xy, contains_glacier_
from Pconv.libs.set_generation import create_test_images_from_glacier_center, flow_train_images



parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM mosaic")
parser.add_argument("--outdir", type=str, default="dataset/", help="path for the output file")

parser.add_argument("--region",  type=int, default=11, help="RGI region")
parser.add_argument("--shape",  type=int, default=256, help="Size of test patches")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")


def bbox(coords_list):
    box = []
    for i in (0,1):
        res = sorted(coords_list, key=lambda x:x[i])
        box.append((res[0][i], res[-1][i]))
    ret = [box[0][0], box[0][1], box[1][0], box[1][1]]
    return ret

def bounding_box(img, label):
    print('Entered bbox function')
    a = np.where(img == label)
    if a[0].size > 0:
        box = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    else:
        box = 0, 0, 0, 0
    return box

def main():

    args = parser.parse_args()

    if os.path.isdir(args.outdir):
        if os.path.isdir(args.outdir + 'images/'):
            None
        else:
            os.mkdir(args.outdir + 'images/')
        if os.path.isdir(args.outdir + 'masks/'):
            None
        else:
            os.mkdir(args.outdir + 'masks/')
        if os.path.isdir(args.outdir + 'masks_full/'):
            None
        else:
            os.mkdir(args.outdir + 'masks_full/')
    else:
        os.mkdir(args.outdir)
        os.mkdir(args.outdir + 'images/')
        os.mkdir(args.outdir + 'masks/')
        os.mkdir(args.outdir + 'masks_full/')

    # disclaimer - reason for slow method is to ensure spacial qualities are
    # preserved in the masks. A better method would:
    #   -   create raster from glacer center points, large enough to fit glacier
    #   -   draw glacier mask and clip mask to fit
    print("The method used for drawing masks is terribly slow,") 
    print("if you are creating more than a few thousand masks,")
    print("consider rewriting this code.")

    # fetch RGI
    utils.get_rgi_dir(version=args.version)
    eu = utils.get_rgi_region_file(args.region, version=args.version)
    gdf = gpd.read_file(eu)

    # load DEM
    dem = rio.open(args.input, 'r').read(1)
    empty = rio.open(args.input.replace('.tif', '_mask.tif'), 'r').read(1)

    # sort glaciers
    glacier_frame = contains_glacier_(args.input, gdf, 0)
    glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
    boolean_series = gdf['RGIId'].isin(glaciers_alps)
    filtered_gdf = gdf[boolean_series]
    filtered_gdf = filtered_gdf.reset_index()

    print("Creating new glacier center points:")
    for i in tqdm(range(len(filtered_gdf))):
        geometry = filtered_gdf['geometry'][i]
        bounding_box = bbox(mapping(loads(str(geometry))).get('coordinates')[0])
        longitude, latitude = (np.average(bounding_box[:2]), np.average(bounding_box[2:]))
        filtered_gdf.loc[i, 'CenLon'] = longitude
        filtered_gdf.loc[i, 'CenLat'] = latitude



    # convert lat/lon to x/y for images
    message = 'Translating lat/lon to x/y: Done'
    print(message[:28], end='\r')
    coords, RGI = coords_to_xy(args.input, filtered_gdf)
    coords_frame = pd.DataFrame({'RGIId': RGI, 'rows': coords[:,0], 'cols': coords[:,1]})
    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    print(message); print(' ')

    print("Creating test patches and masks: ... this may take a while ...")
    patches, masks, RGIId = create_test_images_from_glacier_center(dem, empty, (args.shape, args.shape), 
                                                                   coords_frame, create_blank=True)


    mask = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif'))
    mask = xr.zeros_like(mask)
    mask_copy = mask
    with tqdm(total=len(coords_frame), leave=True) as progress:
        for glacier in range(len(coords_frame)):
            progress.set_postfix_str(RGIId[glacier])
            
            geom = mapping(loads(str(filtered_gdf['geometry'][glacier])))
            mask.rio.write_nodata(1, inplace=True)
            mask = mask.rio.clip([geom], "EPSG:4326", drop=False, invert=True, all_touched=False)
            
            a = np.where(mask.to_numpy()[0] == 1)
            if a[0].size > 0:
                minx = np.min(a[0])
                maxx = np.max(a[0])
                miny = np.min(a[1])
                maxy = np.max(a[1])
            else:
                minx, maxx, miny, maxy = 0, 0, 0, 0
            
            if maxx-minx > args.shape or maxy-miny > args.shape:
                tqdm.write("{} with bounding box ({},{}) has been excluded.".format(RGIId[glacier], maxx-minx, maxy-miny))
            else:

                r = rows[glacier] - int(256/2) if rows[glacier] >= int(256/2) else rows[glacier]
                c = cols[glacier] - int(256/2) if cols[glacier] >= int(256/2) else cols[glacier]
                mask_patch = mask[0, r:r+256, c:c+256]
                mask_patch = np.array(mask_patch).squeeze()

                cv2.imwrite(args.outdir + 'masks/' + RGIId[glacier] + '_mask.tif', mask_patch.astype(np.float32))
                cv2.imwrite(args.outdir + 'images/' + RGIId[glacier] + '.tif', patches[glacier].astype(np.uint16))
                cv2.imwrite(args.outdir + 'masks_full/' + RGIId[glacier] +'_mask.tif', masks[glacier].astype(np.float32))

            # remove mask
            mask.rio.write_nodata(0, inplace=True)
            mask = mask_copy
            progress.update()
            gc.collect()

if __name__ == '__main__':
    main()
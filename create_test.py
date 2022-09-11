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
from Pconv.libs.set_generation import create_test_images_from_glacier_center




parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM mosaic")
parser.add_argument("--outdir", type=str, default="dataset/test/", help="path for the output file")


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
    print('Translating lat/lon to x/y ...')
    coords, RGI = coords_to_xy(args.input, filtered_gdf)
    coords_frame = pd.DataFrame({'RGIId': RGI, 'rows': coords[:,0], 'cols': coords[:,1]})
    rows = np.array(coords_frame['rows'].tolist())
    cols = np.array(coords_frame['cols'].tolist())
    RGIId = coords_frame['RGIId'].tolist()
    print('Done.')


    mask = rioxarray.open_rasterio(args.input.replace('.tif', '_mask.tif'))
    mask = xr.zeros_like(mask)
    mask_copy = mask

    resolution = float(mask.coords['x'][1] - mask.coords['x'][0])  # calculated on x-axis.
    print(f"Raster resolution: {resolution}")


    with tqdm(total=len(coords_frame), leave=True) as progress:
        for glacier in range(len(coords_frame)):
            progress.set_postfix_str(RGIId[glacier])

            geom = filtered_gdf['geometry'][glacier]
            lon_min, lat_min, lon_max, lat_max = geom.bounds
            dx, dy = (lat_max - lat_min) / resolution, (lon_max - lon_min) / resolution

            mask.rio.write_nodata(1, inplace=True)

            # glacier too small
            if (dx==0 or dy==0):
                tqdm.write(f"{RGIId[glacier]} with bounding box ({int(dx)},{int(dy)}) has been excluded.")

            # glacier too big
            elif (dx > args.shape or dy > args.shape):
                tqdm.write(f"{RGIId[glacier]} with bounding box ({int(dx)},{int(dy)}) has been excluded.")

            # glacier OK
            else:
                r = rows[glacier] - int(256/2) if rows[glacier] >= int(256/2) else rows[glacier]
                c = cols[glacier] - int(256/2) if cols[glacier] >= int(256/2) else cols[glacier]

                # First reduce mask then clip
                mask = mask[0, r:r + 256, c:c + 256] # clipped to speed up, geometry seems to be correct
                mask = mask.rio.clip([geom], "EPSG:4326", drop=False, invert=True, all_touched=False, from_disk=True) 

                mask_patch = mask.to_numpy()
                image_patch = dem[0, r:r + 256, c:c + 256].to_numpy()
                full_mask = empty[0, r:r + 256, c:c + 256].to_numpy()


                # save patches, mask and full masks
                cv2.imwrite(args.outdir + 'masks/' + RGIId[glacier] + '_mask.tif', mask_patch.astype(np.float32))
                cv2.imwrite(args.outdir + 'images/' + RGIId[glacier] + '.tif', image_patch.astype(np.uint16))
                cv2.imwrite(args.outdir + 'masks_full/' + RGIId[glacier] +'_mask.tif', full_mask.astype(np.float32))

            # remove mask
            mask.rio.write_nodata(0, inplace=True)
            mask = mask_copy
            progress.update()
            gc.collect()


if __name__ == '__main__':
    main()
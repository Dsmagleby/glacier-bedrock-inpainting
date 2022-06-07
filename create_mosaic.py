import glob
import argparse
import rasterio as rio
import geopandas as gpd


from tqdm import tqdm
from oggm import utils
from libs.utils import contains_glacier_, rasterio_clip



def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ("yes", "true"):
        return True
    elif s.lower() in ("no", "false"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

parser = argparse.ArgumentParser(description='Create DEM mosaic from DEM tiles')
parser.add_argument("--input",  type=str, default=None, help="path to the DEM tile folder")
parser.add_argument("--output", type=str, default="DEM_files/mosaics/mosaic.tif", help="path for the output file")

parser.add_argument('--create_mask',  default=True, type=str2bool, const=True, nargs='?', help='Create mask mosaic')
parser.add_argument("--region",  type=int, default=None, help="RGI region")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")



def main():

    args = parser.parse_args()

    INDIR = args.input
    if INDIR[:-1] != '/':
            INDIR = ''.join((INDIR, '/'))

    OUTPUT = args.output
    
    REGION = args.region
    if args.create_mask == True:
        if REGION == None:
            print("Creating mosaic mask requires RGI region: ex. --region 11")
            exit()

    # get file path of all tiles
    dem_paths = []
    for filename in glob.iglob(INDIR + '*dem.tif'):
        dem_paths.append(filename)


    # add rasterio objects to be merged to a list
    src_files_to_mosaic = []
    for tile in dem_paths:
        src = rio.open(tile)
        src_files_to_mosaic.append(src)


    mosaic, out_trans = rio.merge.merge(src_files_to_mosaic)
    out_meta = src.meta.copy()

    # update the metadata
    out_meta.update({"driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    })


    with rio.open(OUTPUT, "w", **out_meta) as dest:
        dest.write(mosaic)
    
    
    if args.create_mask:
        
        utils.get_rgi_dir(version=args.version)
        eu = utils.get_rgi_region_file(args.region, version=args.version)
        gdf = gpd.read_file(eu)

        print("Creating mosaic mask:")
        for tile in tqdm(dem_paths, leave=True):
            glacier_frame = contains_glacier_(tile, gdf, .5)
            glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
            boolean_series = gdf['RGIId'].isin(glaciers_alps)
            filtered_gdf = gdf[boolean_series]
            filtered_gdf = filtered_gdf.reset_index()
            _ = rasterio_clip(tile, filtered_gdf, args.epsg)

        # add rasterio objects to be merged to a list
        src_files_to_mask = []
        for tile in dem_paths:
            src = rio.open(tile.replace('.tif', '_mask.tif'))
            src_files_to_mask.append(src)


        mask, out_trans = rio.merge.merge(src_files_to_mask)
        out_meta = src.meta.copy()

        # update the metadata
        out_meta.update({"driver": "GTiff",
                        "height": mask.shape[1],
                        "width": mask.shape[2],
                        "transform": out_trans,
                        })


        with rio.open(OUTPUT.replace('.tif', '_mask.tif'), "w", **out_meta) as dest:
            dest.write(mask)



if __name__ == '__main__':
    main()
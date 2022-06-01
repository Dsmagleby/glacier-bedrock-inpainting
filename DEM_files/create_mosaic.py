import glob
import argparse
import rasterio as rio
import geopandas as gpd

from oggm import utils
from ..libs.utils import contains_glacier, rasterio_clip



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
parser.add_argument("--output", type=str, default="DEM_files/mosaic/mosaic.tif", help="path for the output file")

parser.add_argument("--mosaic_input",  type=str, default=None, help="path to the DEM mosaic")
parser.add_argument('--create_mask',  default=False, type=str2bool, const=True, nargs='?', help='Create mask mosaic')
parser.add_argument("--region",  type=int, default=None, help="RGI region")
parser.add_argument("--version",  type=str, default='62', help="RGI version")
parser.add_argument("--epsg",  type=str, default="EPSG:4326", help="DEM projection")



def main():

    args = parser.parse_args()

    INDIR = args.input
    if INDIR[:-1] != '/':
            INDIR = ''.join((INDIR, '/'))

    OUTDIR = args.output
    if OUTDIR[:-1] != '/':
            OUTDIR = ''.join((OUTDIR, '/'))
    
    REGION = args.region
    if args.create_mask == True:
        if REGION == None:
            print("Creating mosaic mask requires RGI region: ex. --region 11")
            exit()


    if args.mosaic_input == None:

        # get file path of all tiles
        dem_paths = []
        for filename in glob.iglob(INDIR + '*dem.tif'):
            dem_paths.append(filename)


        # add rasterio objects to be merged to a list
        src_files_to_mosaic = []
        for tile in dem_paths:
            src = rio.open(tile)
            src_files_to_mosaic.append(src)


        mosaic, out_trans = rio.merge(src_files_to_mosaic)
        out_meta = src.meta.copy()

        # update the metadata
        out_meta.update({"driver": "GTiff",
                        "height": mosaic.shape[1],
                        "width": mosaic.shape[2],
                        "transform": out_trans,
                        })

    
        with rio.open(OUTDIR, "w", **out_meta) as dest:
            dest.write(mosaic)
    
    
    if args.create_mask:
        
        DEM = args.mosaic_input if args.mosaic_input != None else OUTDIR

        utils.get_rgi_dir(version=args.version)
        eu = utils.get_rgi_region_file(args.region, version=args.version)
        gdf = gpd.read_file(eu)

        glacier_frame = contains_glacier(DEM, gdf, 0)
        glaciers_alps = sum(glacier_frame['RGIId'].tolist(), [])
        boolean_series = gdf['RGIId'].isin(glaciers_alps)
        filtered_gdf = gdf[boolean_series]
        filtered_gdf = filtered_gdf.reset_index()

        _ = rasterio_clip(OUTDIR, filtered_gdf, args.epsg)



if __name__ == '__main__':
    main()
import rioxarray
import os
import sys
import geopandas as gpd
import logging
from shapely.geometry import box

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Path to unprocessed rasters
unprocessed_dir = './Unprocessed'

included_files = ['LC08_L2SP_040035_20130810_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130810_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130810_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130810_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20131216_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130810_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130810_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130420_20200912_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_SR_B2.TIF', 'LC08_L2SP_040035_20130709_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20131130_20200912_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130319_20200913_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_SR_B4.TIF', 'LC08_L2SP_040035_20130623_20200912_02_T1_SR_B6.TIF', 'LC08_L2SP_040035_20130607_20200912_02_T1_SR_B5.TIF', 'LC08_L2SP_040035_20130725_20200912_02_T1_ST_B10.TIF', 'LC08_L2SP_040035_20130522_20200913_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20130927_20200913_02_T1_SR_B3.TIF', 'LC08_L2SP_040035_20131013_20200912_02_T1_SR_B6.TIF']


# Ensure the directory exists
if not os.path.exists(unprocessed_dir):
    logging.error(f"Directory '{unprocessed_dir}' does not exist.")
    sys.exit(1)

# Load your boundary polygon GeoDataFrame
try:
    boundPolygon = gpd.read_file('./Data/area_shp/Polygon_Pahrump_NV.shp')
    logging.info("Boundary polygon loaded successfully.")

    # Ensure the boundary polygon is in EPSG:4326
    if boundPolygon.crs != "EPSG:4326":
        logging.info("Reprojecting boundary polygon to EPSG:4326...")
        boundPolygon = boundPolygon.to_crs("EPSG:4326")
    logging.info(f"CRS of Boundary Polygon: {boundPolygon.crs}")
    
    # Log the original polygon bounds in EPSG:4326
    polygon_bounds_4326 = boundPolygon.geometry.union_all().bounds
    logging.info(f"Polygon Bounds in EPSG:4326: {polygon_bounds_4326}")
except Exception as e:
    logging.error(f"Error loading boundary polygon: {e}")
    sys.exit(1)

# Check if there are any valid geometries in the boundary polygon
if boundPolygon.empty:
    logging.error("Boundary polygon is empty.")
    sys.exit(1)

for geom in boundPolygon.geometry:
    logging.info(f"Geometry Type: {geom.geom_type}, Is Valid: {geom.is_valid}")

# Loop through the specified raster files
for file in included_files:
    logging.info(f"Processing file: {file}")
    try:
        raster_path = os.path.join(unprocessed_dir, file)
        
        # Check if the file exists
        if not os.path.exists(raster_path):
            logging.warning(f"File '{file}' not found in the directory. Skipping...")
            continue
        
        # Load raster
        raster = rioxarray.open_rasterio(raster_path)
        logging.info("Raster loaded successfully!")

        # Print raster CRS
        if raster.rio.crs is not None:
            logging.info(f"Original Raster CRS: {raster.rio.crs}")
        else:
            logging.warning("Raster CRS is not defined. Skipping...")
            continue

        # Reproject raster to EPSG:4326
        raster_reprojected = raster.rio.reproject("EPSG:4326")
        logging.info("Raster reprojected to EPSG:4326")

        # Log the extents for debugging
        raster_bounds_4326 = box(*raster_reprojected.rio.bounds())
        polygon_bounds_4326 = boundPolygon.geometry.union_all().bounds
        logging.info(f"Raster Bounds in EPSG:4326: {raster_bounds_4326.bounds}")
        logging.info(f"Polygon Bounds in EPSG:4326: {polygon_bounds_4326}")

        # Check overlap between raster and polygon
        if not raster_bounds_4326.intersects(box(*polygon_bounds_4326)):
            logging.warning(f"Raster '{file}' does not intersect with the polygon. Skipping...")
            continue

        # Perform a test clip operation with the first geometry
        first_geometry = boundPolygon.geometry.iloc[0]
        clipped = raster_reprojected.rio.clip([first_geometry], "EPSG:4326", drop=True)
        logging.info(f"Clipping successful for raster: {file}")

    except Exception as e:
        logging.error(f"Error processing raster '{file}': {e}")

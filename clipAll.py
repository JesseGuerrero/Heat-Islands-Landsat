#%%
#Setup script
import os
import sys
import geopandas as gpd
import rasterio
import rioxarray
from geopandas import GeoDataFrame
from tqdm import tqdm
from shapely.geometry import box
from shapely.geometry import Point, Polygon
import smtplib
from email.mime.text import MIMEText
import socket
def notifySelf(body):
	try:
		subject = "Jupyter Notebook Cell Output"
		to_email = "19152478204.19152478204.ZXZzbrC-TB@txt.voice.google.com"
		from_email = "jesseguerrero1991@gmail.com"
		password = "yhts ilcq ymyp nfjv"  # Use an app password for Gmail or similar
		msg = MIMEText(f'{socket.gethostname()}: {body}')
		msg["Subject"] = subject
		msg["From"] = from_email
		msg["To"] = to_email
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
			server.login(from_email, password)
			server.sendmail(from_email, to_email, msg.as_string())
	except Exception as e:
		print("unable to send text...")
		pass
def get_file_paths(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            if "oli_label" in str(full_path):
                continue
            file_paths.append(full_path)
    return file_paths

def save_file_paths_to_log(file_paths, log_file="rawPaths.log"):
    with open(log_file, "w") as log:
        for path in tqdm(file_paths, desc="Saving to log"):
            log.write(path + "\n")  # Write each path followed by a newline
    print(f"File paths saved to {log_file}")

def read_file_paths_from_log(log_file="rawPaths.log"):
    with open(log_file, "r") as log:
        file_paths = [line.strip() for line in log]  # Remove any leading/trailing whitespace
    return file_paths

import shutil
def moveToClipped(filePath: str, fileName, typeFolder: str, date, city):
	target_folder = os.path.join('./Clipped', typeFolder, city, date)
	os.makedirs(target_folder, exist_ok=True)
	target_file_path = os.path.join(target_folder, fileName)
	shutil.copy2(filePath, target_file_path)

def checkPolygonInRasterCompletely(polygon: GeoDataFrame, ras: str):
    polygon = polygon.geometry.iloc[0]
    with rasterio.open(ras) as src:
        bounds = src.bounds
        raster_bounds = Polygon([
            (bounds.left, bounds.top),
            (bounds.right, bounds.top),
            (bounds.right, bounds.bottom),
            (bounds.left, bounds.bottom),
            (bounds.left, bounds.top)
        ])
        nodata_value = src.nodata
    is_within = raster_bounds.contains(polygon)
    if not is_within:
        return False
    with rasterio.open(ras) as src:
        for x, y in polygon.exterior.coords:
            row, col = src.index(x, y)
            pixel_value = src.read(1)[row, col]
            if pixel_value == nodata_value:
                return False
    return True

shapefile_folder = "./Data/area_shp/"
cities_aoi = {}
for file in os.listdir(shapefile_folder):
    if file.endswith(".shp"):
        aoi_geodf = gpd.read_file(shapefile_folder + file)
        aoi_geodf = aoi_geodf.to_crs("EPSG:4326")
        if aoi_geodf.empty:
            sys.exit("Error: Shapefile contains no data.")
        cities_aoi[file.replace('Polygon_', '').replace('.shp', '')] = aoi_geodf
'''
#%%
#Get all tifs, copy everything else
notifySelf("Starting clip list of raster paths...")
allGeoFiles = get_file_paths('./RawClippedRasters')
usableTIFFs = []
for geoFilePath in tqdm(allGeoFiles, desc="Make raster list/move txt files"):
    geoFileParts = geoFilePath.split('/')
    fileName, date, city, dataType = geoFileParts[-1], geoFileParts[-2], geoFileParts[-3], geoFileParts[-4]
    targetFile = os.path.join('./Clipped', dataType, city, date, fileName)
    if os.path.exists(targetFile):
        continue
    polygon = cities_aoi[city]
    if geoFilePath.endswith(".tif") or geoFilePath.endswith(".TIF"):
        if checkPolygonInRasterCompletely(polygon, geoFilePath):
            usableTIFFs.append(geoFilePath)
            # print(f"Polygon {city} is fully inside {fileName}")
        # else:
            # print(f"Polygon {city} is NOT fully inside {geoFilePath}")
    elif geoFilePath.endswith(".txt"):
        moveToClipped(geoFilePath, fileName, dataType, date, city)
#%%
save_file_paths_to_log(usableTIFFs)
#%%
'''
filesList = read_file_paths_from_log()
i, fileFivePercent = 0, int(len(filesList)//20)
notifySelf("Starting clip Rasters...")
#Clip, project and move to new folder.
for geoFilePath in tqdm(filesList, desc="Clipping rasters"):
    if i % fileFivePercent == 0:
        percentage_done = int((i / len(filesList)) * 100)
        notifySelf(f'We are at {percentage_done}% clipped')
    i+=1
    geoFileParts = geoFilePath.split('/')
    fileName, date, city, dataType = geoFileParts[-1], geoFileParts[-2], geoFileParts[-3], geoFileParts[-4]
    target_folder = os.path.join('./Clipped', dataType, city, date)
    os.makedirs(target_folder, exist_ok=True)
    target_file = os.path.join('./Clipped', dataType, city, date, fileName)
    if os.path.exists(target_file):
        continue
    polygon = cities_aoi[city]
    raster = rioxarray.open_rasterio(geoFilePath)
    raster_reprojected = raster.rio.reproject("EPSG:4326")
    colormap = None
    with rasterio.open(geoFilePath) as src:
        if src.colorinterp[0] == rasterio.enums.ColorInterp.palette:
            colormap = src.colormap(1)  # Assuming band 1 has the colormap
    clipped = raster_reprojected.rio.clip(polygon.geometry, polygon.crs, drop=True)
    clipped.rio.to_raster(target_file)
    if colormap:
        with rasterio.open(target_file, "r+") as dest:
            dest.write_colormap(1, colormap)
    raster.close()
    raster_reprojected.close()
#%%

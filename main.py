#%%
import shutil

'''Setup entire script and define raw raster download'''
from utils.util import *
import geopandas as gpd
import os
import sys
import time
import pandas as pd
import rasterio
import rioxarray
import socket
import traceback
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="Process data for a range of years.")

# Add arguments for startYear and endYear
parser.add_argument(
    "--startYear",
    type=int,
    required=True,
    help="The starting year of the data range."
)
parser.add_argument(
    "--endYear",
    type=int,
    required=True,
    help="The ending year of the data range."
)
args = parser.parse_args()
startYear, endYear = args.startYear, args.endYear

def gatherRawRasters(dataset, year, city, aoi_geodf):
    print(f'Gathering {dataset} for {year} in {city}.')
    if dataset == 'srtm_v3' and year != 2014:
        return
    bandNames = {'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'ST_B10', 'ST_EMIS', 'QA_PIXEL'}
    for month in range(1, 13):
        if month == 1:
            notifySelf(f'Starting on year {year} in dataset {dataset} as city {city}...')
        #Search for scenes
        print("Starting month", month)
        clear_folder(unprocessed_dir)
        search_payload = createSceneSearchPayload(dataset, aoi_geodf, year, month)
        scenes = sendRequest(serviceUrl + "scene-search", search_payload, apiKey)
        pd.json_normalize(scenes['results'])
        if len(scenes['results']) == 0:
            print("Month for scenes empty, skipping...")
            continue

        # Collect File IDs
        entityIds = [result['entityId'] for result in scenes['results'] if result['options']['bulk']]

        # Add to basket
        listId = f"{dataset}_{year}_{str(month)}_{socket.gethostname()}"
        scn_list_add_payload = {
            "listId": listId,
            'idField': 'entityId',
            "entityIds": entityIds,
            "datasetName": dataset
        }
        sendRequest(serviceUrl + "scene-list-add", scn_list_add_payload, apiKey)

        # Select URL download
        download_opt_payload = {
            "listId": listId,
            "datasetName": dataset,
        }
        products = sendRequest(serviceUrl + "download-options", download_opt_payload, apiKey)
        pd.json_normalize(products)

        # Collect File URLs based on product
        downloads = []
        if 'landsat_ot_c2_l2' in dataset:
            for product in products:
                if product["secondaryDownloads"]:
                    for secDownload in product["secondaryDownloads"]:
                        if secDownload["bulkAvailable"] and any(band in secDownload['displayId'] for band in bandNames):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
                        if secDownload['displayId'].endswith('_MTL.txt'):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
        elif 'srtm_v3' in dataset:
            for product in products:
                if product["bulkAvailable"] and product["entityId"] and product["id"]:
                    downloads.append({"entityId": product["entityId"], "productId": product["id"]})
        elif 'nlcd_collection_lndcov' in dataset:
            for product in products:
                if product["bulkAvailable"] and product["entityId"] and product["id"]:
                    downloads.append({"entityId": product["entityId"], "productId": product["id"]})
        download_req_payload = {
            "downloads": downloads,
            "label": listId
        }
        download_request_results = sendRequest(serviceUrl + "download-request", download_req_payload, apiKey)

        # Download Files Via URL
        if dataset == 'landsat_ot_c2_l2':
            results = download_request_results['availableDownloads']
            for result in results:
                runDownload(threads, result['url'])
        elif dataset == 'srtm_v3':
            results = download_request_results['preparingDownloads']
            for result in results:
                runDownload(threads, result['url'])
        else:
            results = download_request_results['preparingDownloads']
            for result in results:
                runDownload(threads, result['url'])
        for t in threads:
            t.join()
        for file in os.listdir(unprocessed_dir):
            if file.endswith('.tar'):
                try:
                    extract_specific_files(unprocessed_dir + '/' + file, unprocessed_dir)
                except:
                    print(f'Error: Could not extract file {file}')

        # Clear Basket
        remove_scnlst_payload = {"listId": listId}
        sendRequest(serviceUrl + "scene-list-remove", remove_scnlst_payload, apiKey)

        # Re-project Raster Files
        for tif in os.listdir(unprocessed_dir):
            if tif.endswith('.tif') or tif.endswith('.TIF'):
                temp_path = ".temp"  # Temporary file path
                input_path = unprocessed_dir + '/' + tif
                with rasterio.open(input_path) as src:
                    try:
                        color_map = src.colormap(1)
                    except ValueError:
                        color_map = None
                    with rioxarray.open_rasterio(input_path) as raster:
                        raster = raster.rio.reproject("EPSG:4326")
                        raster.rio.to_raster(temp_path, driver="GTiff")
                if color_map:
                    with rasterio.open(temp_path, "r+") as dst:
                        dst.write_colormap(1, color_map)
                os.replace(temp_path, input_path)
                if os.path.exists(temp_path):
                    os.remove(temp_path)

        # Move Unprocessed Files to Raw Folder
        for file in os.listdir(unprocessed_dir):
            if ".txt" in file or ".tif" in file or ".TIF" in file:
                if '1arc_v3' in file:
                    moveToRaw(file, 'DEM', f'{year}-01-01', city)
                    continue
                if 'Annual_NLCD' in file:
                    moveToRaw(file, 'Land_Cover', f'{year}-01-01', city)
                    continue
                date, band, coordinate = getMetaFromLandsatTIRs(file)
                if band in ['MTL', 'B10', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'EMIS', 'PIXEL']:
                    moveToRaw(file, 'oli', date, city)
        print('Finished moving')

        #You only need 1 month for these as they are annual
        if dataset == 'nlcd_collection_lndcov' or dataset == 'srtm_v3':
            break

    #Save progress
    with open('./Logs/raw_progress.txt', "a") as file:
        file.write(str(city) + ":" + str(year) + ":" + dataset + "\n")
    print('progress written for', city, year, dataset)
#%%
datasets = ['landsat_ot_c2_l2', 'srtm_v3', 'nlcd_collection_lndcov']
years = [year for year in range(startYear, endYear+1)]

# Load city footprints from Esri Living Atlas
shapefile_folder = "./Data/area_shp/"
cities = []
aoi_geodfs = []
for file in os.listdir(shapefile_folder):
    if file.endswith(".shp"):
        cities.append(file.replace('Polygon_', '').replace('.shp', ''))
        aoi_geodf = gpd.read_file(shapefile_folder + file)
        aoi_geodf = aoi_geodf.to_crs("EPSG:4326")
        if aoi_geodf.empty:
            sys.exit("Error: Shapefile contains no data.")
        aoi_geodfs.append(aoi_geodf)
print("Shapefiles loaded successfully.")

for j, dataset in enumerate(datasets):
    for year in years:
        while i < int(len(cities)):
            try:
                clear_folder(unprocessed_dir)
                assert len(os.listdir(unprocessed_dir)) == 0, "Unprocessed directory is not empty."
                city, aoi_geodf = cities[i], aoi_geodfs[i]
                if os.path.exists('./Logs/raw_progress.txt'):
                    with open('./Logs/raw_progress.txt', 'r') as file:
                        progress = [line.split(':') for line in file.read().strip().split('\n')]
                    if any(city == instance[0] and str(year) == instance[1] and dataset == instance[2] for instance in progress):
                        print(f"{city}, {year}, {dataset} was gathered in the past.")
                    else:
                        gatherRawRasters(dataset, year, city, aoi_geodf)
            except Exception as e:
                print("An exception occurred:")
                print(f"Exception: {e}")
                notifySelf("An exception occurred:")
                notifySelf(f"Exception: {e}")
                traceback.print_exc()  # Print the full stack trace
                time.sleep(15)
print("Gathered raw rasters successfully.")
notifySelf("Gathered raw rasters successfully.")
#%%
'''Clip Raster, we are reading metadata from rastername'''
shapefile_folder = "./Data/area_shp/"
cities_aoi = {} # Use cities dictionary instead of list to read from the raster itself
for file in os.listdir(shapefile_folder):
    if file.endswith(".shp"):
        aoi_geodf = gpd.read_file(shapefile_folder + file)
        aoi_geodf = aoi_geodf.to_crs("EPSG:4326")
        if aoi_geodf.empty:
            sys.exit("Error: Shapefile contains no data.")
        cities_aoi[file.replace('Polygon_', '').replace('.shp', '')] = aoi_geodf

#Get all tifs, copy everything else
notifySelf("Starting clip list of raster paths...")
allGeoFiles = get_file_paths(raw_dir)
i, file25Percent = 0, int(len(allGeoFiles)//4)
usableTIFFs = []
for geoFilePath in tqdm(allGeoFiles, desc="Make raster list/move txt files"):
    if i % file25Percent == 0:
        percentage_done = int((i / len(allGeoFiles)) * 100)
        notifySelf(f'We are at {percentage_done}% clipped')
    i+=1
    geoFileParts = geoFilePath.split('/')
    fileName, date, city, dataType = geoFileParts[-1], geoFileParts[-2], geoFileParts[-3], geoFileParts[-4]
    targetFile = os.path.join(clipped_dir, dataType, city, date, fileName)
    if os.path.exists(targetFile):
        continue
    polygon = cities_aoi[city]
    if geoFilePath.endswith(".tif") or geoFilePath.endswith(".TIF"):
        if checkPolygonInRasterCompletely(polygon, geoFilePath):
            usableTIFFs.append(geoFilePath)
    elif geoFilePath.endswith(".txt"):
        moveToClipped(geoFilePath, fileName, dataType, date, city)
#%%
save_paths_to_log(usableTIFFs)
#%%
filesList = read_file_paths_from_log()
i, file25Percent = 0, int(len(filesList) // 4)
notifySelf("Starting clip Rasters...")
#Clip, project and move to new folder.
for geoFilePath in tqdm(filesList, desc="Clipping rasters"):
    if i % file25Percent == 0:
        percentage_done = int((i / len(filesList)) * 100)
        notifySelf(f'We are at {percentage_done}% clipped')
    i += 1
    geoFileParts = geoFilePath.split('/')
    fileName, date, city, dataType = geoFileParts[-1], geoFileParts[-2], geoFileParts[-3], geoFileParts[-4]
    target_folder = os.path.join(clipped_dir, dataType, city, date)
    os.makedirs(target_folder, exist_ok=True)
    target_file = os.path.join(clipped_dir, dataType, city, date, fileName)
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
print("Clipped and moved raw rasters successfully.")
notifySelf("Clipped and moved raw rasters successfully.")
#%%
def list_files_in_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def moveToProcess(filePath: str, fileName, typeFolder: str, date, city):
    target_folder = os.path.join(process_dir, typeFolder, city, date)
    os.makedirs(target_folder, exist_ok=True)
    target_file_path = os.path.join(target_folder, fileName)
    if os.path.exists(target_file_path):
        return
    shutil.copy2(filePath, target_file_path)

#Remove scenes with excessive cloud cover
i, file25Percent = 0, int(len(filesList) // 4)
notifySelf("Starting cloud removal...")
process_dir = temp_dir + '/Process'
allClippedFiles, validCloudFiles = [], []
for clippedFilePath in get_file_paths(clipped_dir):
    if 'QA_PIXEL' in clippedFilePath:
        allClippedFiles.append(clippedFilePath)
for clippedFilePath in tqdm(allClippedFiles, desc="Filtering Clouds & Missing Tifs"):
    if i % file25Percent == 0:
        percentage_done = int((i / len(filesList)) * 100)
        notifySelf(f'We are at {percentage_done}% clipped')
    i += 1
    if calculate_cloud_cover_percentage(clippedFilePath) < 10:
        fileParts = clippedFilePath.split('/')
        fileName, date, city, dataType = fileParts[-1], fileParts[-2], fileParts[-3], fileParts[-4]
        sceneIdentifyingName = '_'.join(fileName.split('_')[:7])
        sceneFiles = []
        for sceneFileAbsolutePath in list_files_in_folder(os.path.dirname(os.path.abspath(clippedFilePath))):
            if sceneIdentifyingName in sceneFileAbsolutePath:
                sceneFiles.append(sceneFileAbsolutePath)
        if len(sceneFiles) == 10:
            for sceneFile in sceneFiles:
                moveToProcess(sceneFile, sceneFile.split('/')[-1], dataType, date, city)
#%%
def extract_constants(mtl_file_path):
    constants = {}
    with open(mtl_file_path, 'r') as mtl_file:
        lines = mtl_file.readlines()
    for line in lines:
        line = line.strip()
        if line.startswith("RADIANCE_MULT_BAND_10"):
            constants['ML'] = float(line.split(" = ")[1])
        elif line.startswith("RADIANCE_ADD_BAND_10"):
            constants['AL'] = float(line.split(" = ")[1])
        elif line.startswith("K1_CONSTANT_BAND_10"):
            constants['K1'] = float(line.split(" = ")[1])
        elif line.startswith("K2_CONSTANT_BAND_10"):
            constants['K2'] = float(line.split(" = ")[1])
        for band in range(2, 7):  # Loop through bands 2 to 6
            mult_key = f"REFLECTANCE_MULT_BAND_{band}"
            add_key = f"REFLECTANCE_ADD_BAND_{band}"
            if line.startswith(mult_key):
                constants[mult_key] = float(line.split(" = ")[1])
            elif line.startswith(add_key):
                constants[add_key] = float(line.split(" = ")[1])
    return constants

def retrieve_ndwi(green_band_path, nir_band_path, output_path):
    with rasterio.open(green_band_path) as green_src, rasterio.open(nir_band_path) as nir_src:
        green = green_src.read(1).astype('float32') / 10_000
        nir = nir_src.read(1).astype('float32') / 10_000
        green = np.clip(green, 0, 1)
        nir = np.clip(nir, 0, 1)

        ndwi = np.where((green + nir) == 0, 0, (green - nir) / (green + nir))

        profile = green_src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndwi, 1)

def retrieve_ndvi(nir_band_path, red_band_path, output_path):
    with rasterio.open(nir_band_path) as nir_src, rasterio.open(red_band_path) as red_src:
        nir = nir_src.read(1).astype('float32') / 10_000
        red = red_src.read(1).astype('float32') / 10_000
        nir = np.clip(nir, 0, 1)
        red = np.clip(red, 0, 1)
        ndvi = np.where((nir + red) == 0, 0, (nir - red) / (nir + red))

        profile = nir_src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(ndvi, 1)

def retrieve_lst(band10_path, emis, cloud, constants, output_path, convert_to_fahrenheit=True):
    wavelength = 10.8e-6  # Band 10 wavelength in meters
    rho = 1.438e-2  # Planck's constant divided by Boltzmann constant in mK

    with rasterio.open(band10_path) as band10_src, \
         rasterio.open(emis) as emissivity_src, \
         rasterio.open(cloud) as cloud_src:

        # Read Band 10 pixel values (Q_cal), emissivity, and QA_PIXEL
        q_cal = band10_src.read(1).astype('float32')
        emissivity = emissivity_src.read(1).astype('float32')
        qa_pixel = cloud_src.read(1).astype('uint16')

        # Get nodata value
        band10_nodata = band10_src.nodata
        emissivity_nodata = emissivity_src.nodata

        # Valid mask to exclude nodata values
        valid_mask = (q_cal != band10_nodata) & (emissivity != emissivity_nodata)

        # Scale emissivity if it is scaled by a factor of 10,000
        emissivity = np.where(valid_mask, emissivity / 10000, emissivity_nodata)

        # Extract cloud confidence using the ArcGIS Pro logic
        # Equivalent to Con(BitwiseAnd("QA_PIXEL", 192) < 64, 1, 0)
        cloud_mask = (qa_pixel & 192) < 64  # True for non-cloud-covered pixels

        # Include only non-cloud-covered pixels
        valid_mask = valid_mask & ~cloud_mask

        # Calculate Radiance (L_lambda) only for valid pixels
        l_lambda = np.where(valid_mask, constants['ML'] * q_cal + constants['AL'], band10_nodata)

        # Calculate Brightness Temperature (T_b)
        tb = np.where(valid_mask, constants['K2'] / np.log((constants['K1'] / l_lambda) + 1), band10_nodata)

        # Calculate Land Surface Temperature (LST)
        lst = np.where(valid_mask, tb / (1 + (wavelength * tb / rho) * np.log(emissivity)), band10_nodata)

        if convert_to_fahrenheit:
            lst = np.where(valid_mask, (lst - 273.15) * (9 / 5) + 32, band10_nodata)

        # Write the LST raster to output
        profile = band10_src.profile
        profile.update(dtype=rasterio.float32, count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(lst, 1)

def create_heat_index(lst_path, output_path):
    with rasterio.open(lst_path) as src:
        lst = src.read(1).astype('float32')
        nodata_value = src.nodata

        # Exclude NoData values from min/max calculation
        valid_mask = lst != nodata_value
        valid_lst = lst[valid_mask]
        lst_min, lst_max = np.min(valid_lst), np.max(valid_lst)

        # Create 10 intervals for valid LST range
        category_bounds = np.linspace(lst_min, lst_max, 11)  # Create 10 evenly spaced intervals

        # Initialize heat index array with nodata value
        lst_categories = np.full_like(lst, nodata_value, dtype='int32')

        # Assign categories only to valid pixels
        lst_categories[valid_mask] = np.digitize(valid_lst, category_bounds, right=True)

        profile = src.profile
        profile.update(dtype=rasterio.int32, count=1)

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(lst_categories, 1)


def retrieve_albedo(band2_path, band3_path, band4_path, band5_path, band6_path, output_path):
    # NTB coefficients for Landsat 8 bands
    coefficients = {
        "band2": 0.356,
        "band3": 0.130,
        "band4": 0.373,
        "band5": 0.085,
        "band6": 0.072,
        "offset": -0.018
    }
    with rasterio.open(band2_path) as band2_src, \
         rasterio.open(band3_path) as band3_src, \
         rasterio.open(band4_path) as band4_src, \
         rasterio.open(band5_path) as band5_src, \
         rasterio.open(band6_path) as band6_src:
        band2 = band2_src.read(1).astype('float32') / 10000
        band3 = band3_src.read(1).astype('float32') / 10000
        band4 = band4_src.read(1).astype('float32') / 10000
        band5 = band5_src.read(1).astype('float32') / 10000
        band6 = band6_src.read(1).astype('float32') / 10000
        band2 = np.clip(band2, 0, 1)
        band3 = np.clip(band3, 0, 1)
        band4 = np.clip(band4, 0, 1)
        band5 = np.clip(band5, 0, 1)
        band6 = np.clip(band6, 0, 1)

        albedo = (
            coefficients["band2"] * band2 +
            coefficients["band3"] * band3 +
            coefficients["band4"] * band4 +
            coefficients["band5"] * band5 +
            coefficients["band6"] * band6 +
            coefficients["offset"]
        )
        profile = band2_src.profile
        profile.update(dtype=rasterio.float32, count=1)
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(albedo, 1)

def getDataPath(fileName, typeFolder: str, date, city):
    target_folder = os.path.join(data_dir, typeFolder, city, date)
    os.makedirs(target_folder, exist_ok=True)
    target_file_path = os.path.join(target_folder, fileName)
    return target_file_path

def getClippedPath(typeFolder: str, date, city):
    target_folder = os.path.join(clipped_dir, typeFolder, city, date)
    if os.path.exists(target_folder):
        target_files = os.listdir(target_folder)
        if len(target_files) > 0:
            targetFile = os.listdir(target_folder)[0]
            return os.path.join(target_folder, targetFile)
    return 'invalid'

allQAPixelFiles = []
for filePath in get_file_paths(process_dir):
    if 'QA_PIXEL' in filePath:
        allQAPixelFiles.append(filePath)
for filePath in allQAPixelFiles:
    fileParts = filePath.split('/')
    fileName, date, city, dataType = fileParts[-1], fileParts[-2], fileParts[-3], fileParts[-4]
    # if "San_Antonio" not in city:
    #     continue
    sceneIdentifyingName = '_'.join(fileName.split('_')[:7])
    sceneFiles = []
    for sceneFileAbsolutePath in list_files_in_folder(os.path.dirname(os.path.abspath(filePath))):
        if sceneIdentifyingName in sceneFileAbsolutePath:
            sceneFiles.append(sceneFileAbsolutePath)
    for sceneFile in sceneFiles:
        band = sceneFile.split('_')[-1].replace('.TIF', '').replace('.txt', '').replace('.tif', '')
        if band == 'MTL':
            MTL = sceneFile
        if band == 'PIXEL':
            cloud = sceneFile
        if band == 'B10':
            band10 = sceneFile
        if band == 'B2':
            band2 = sceneFile
        if band == 'B3':
            band3 = sceneFile
        if band == 'B4':
            band4 = sceneFile
        if band == 'B5':
            band5 = sceneFile
        if band == 'B6':
            band6 = sceneFile
        if band == 'B7':
            band7 = sceneFile
        if band == 'EMIS':
            emis = sceneFile
    constants = extract_constants(MTL)
    demPath = getClippedPath('DEM', '2014-01', city)
    landCover = getClippedPath('Land_Cover', '2014-01', city)
    if os.path.exists(demPath) and os.path.exists(landCover):
        shutil.copy2(demPath, getDataPath('DEM.tif', 'X', date, city))
        shutil.copy2(landCover, getDataPath('Land_Cover.tif', 'X', date, city))
        retrieve_ndvi(band5, band4, getDataPath('NDVI.tif', 'X', date, city))
        retrieve_ndwi(band3, band5, getDataPath('NDWI.tif', 'X', date, city))
        retrieve_albedo(band2, band3, band4, band5, band6, getDataPath('Albedo.tif', 'X', date, city))
        retrieve_lst(band10, emis, cloud, constants, getDataPath('LST.tif', 'y', date, city))
        create_heat_index(getDataPath('LST.tif', 'y', date, city), getDataPath('Heat_Index.tif', 'y', date, city))
#%%

#%%

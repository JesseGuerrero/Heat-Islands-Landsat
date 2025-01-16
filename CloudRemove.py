#%%
from utils.util import *
import geopandas as gpd
import os
import sys
import time
import pandas as pd
import rasterio
import argparse

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Process a range of years.")

# Add required arguments for startYear and endYear
parser.add_argument('--startYear', type=int, required=True, help="The starting year (required)")
parser.add_argument('--endYear', type=int, required=True, help="The ending year (required)")

# Parse the arguments
args = parser.parse_args()

# Access the arguments
start_year = args.startYear
end_year = args.endYear

# Check and handle the years
if start_year > end_year:
    print(f"Error: startYear ({start_year}) must be less than or equal to endYear ({end_year}).")
else:
    print(f"Processing data from {start_year} to {end_year}.")
def gatherData(dataset, year, city, aoi_geodf):
    print(f'Gathering {dataset} for {year} in {city}.')
    bandNames = {'QA_PIXEL'}
    includeMetadata = False
    for month in range(1, 13):
        print("Starting month", month)
        clear_folder(unprocessed_dir)
        search_payload = createSceneSearchPayload(dataset, aoi_geodf, year, month)
        # print(search_payload)
        scenes = sendRequest(serviceUrl + "scene-search", search_payload, apiKey)
        pd.json_normalize(scenes['results'])
        if len(scenes['results']) == 0:
            print("Month for scenes empty, skipping...")
            continue
        #%%
        # Cell 7: Collect Entity IDs
        entityIds = [result['entityId'] for result in scenes['results'] if result['options']['bulk']]
        #%%
        # Cell 8: Prepare Scene List for Download
        listId = f"{dataset}_{year}_{str(month)}_Cloud"
        scn_list_add_payload = {
            "listId": listId,
            'idField': 'entityId',
            "entityIds": entityIds,
            "datasetName": dataset
        }
        sendRequest(serviceUrl + "scene-list-add", scn_list_add_payload, apiKey)
        #%%
        # Cell 9: Prepare Download Options
        download_opt_payload = {
            "listId": listId,
            "datasetName": dataset,
        }
        products = sendRequest(serviceUrl + "download-options", download_opt_payload, apiKey)
        pd.json_normalize(products)
        downloads = []
        if 'landsat_ot_c2_l2' in dataset:
            for product in products:
                if product["secondaryDownloads"]:
                    for secDownload in product["secondaryDownloads"]:
                        if secDownload["bulkAvailable"] and any(band in secDownload['displayId'] for band in bandNames):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
                        if includeMetadata and secDownload['displayId'].endswith('_MTL.txt'):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
        download_req_payload = {
            "downloads": downloads,
            "label": listId
        }
        download_request_results = sendRequest(serviceUrl + "download-request", download_req_payload, apiKey)
        if dataset == 'landsat_ot_c2_l2':
            results = download_request_results['availableDownloads']
            for result in results:
                runDownload(threads, result['url'])
        for t in threads:
            t.join()
        for file in os.listdir('./Unprocessed'):
            if file.endswith('.tar'):
                try:
                    extract_specific_files('./Unprocessed/' + file, './Unprocessed')
                except:
                    print(f'Error: Could not extract file {file}')
        remove_scnlst_payload = {"listId": listId}
        sendRequest(serviceUrl + "scene-list-remove", remove_scnlst_payload, apiKey)
        print('Unprocessed Dir:', os.listdir(unprocessed_dir))
        tifs = [] # QA_PIXEL
        for file in os.listdir(unprocessed_dir):
            if file.endswith(".TIF") or file.endswith(".tif"):
                tifs.append(file)
        #Reproject
        for tif in tifs:
            temp_path = ".temp"  # Temporary file path
            input_path = unprocessed_dir + '/' + tif
            with rasterio.open(input_path) as src:
                try:
                    color_map = src.colormap(1)  # Assuming a single band with colormap
                except ValueError:
                    color_map = None  # No colormap present
                with rioxarray.open_rasterio(input_path) as raster:
                    raster = raster.rio.reproject("EPSG:4326")
                    raster.rio.to_raster(temp_path, driver="GTiff")
            if color_map:
                with rasterio.open(temp_path, "r+") as dst:
                    dst.write_colormap(1, color_map)
            os.replace(temp_path, input_path)
            if os.path.exists(temp_path):
                os.remove(temp_path)
            for file in os.listdir(unprocessed_dir):
                moveToCloud(file, 'oli', f'{year}-{str(0) + str(month) if len(str(month)) == 1 else str(month)}', city)
    with open('Cloud.txt', "a") as file:
        file.write(str(city) + ":" + str(year) + ":" + dataset + "\n")
    print('progress written for', city, year, dataset)
#%%
datasets = ['landsat_ot_c2_l2']
years = [year for year in range(start_year, end_year)]

# Cell 2: Load shape file
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
import traceback
for j, dataset in enumerate(datasets):
    for year in years:
        i = 0
        while i < len(cities):
            if i % 7 == 0:
                notifySelf(f'Starting on year {year} for clouds as city {cities[i]}...')
            try:
                clear_folder(unprocessed_dir)
                assert len(os.listdir(unprocessed_dir)) == 0, "Unprocessed directory is not empty."
                city, aoi_geodf =  cities[i], aoi_geodfs[i]
                if os.path.exists('Cloud.txt'):
                    with open('Cloud.txt', 'r') as file:
                        progress = [line.split(':') for line in file.read().strip().split('\n')]
                    if any(city == instance[0] and str(year) == instance[1] and dataset == instance[2] for instance in progress):
                        print(f"{city}, {year}, {dataset} was gathered in the past.")
                    else:
                        gatherData(dataset, year, city, aoi_geodf)
                i += 1
            except Exception as e:
                notifySelf("An exception occurred:")
                notifySelf(f"Exception: {e}")
                traceback.print_exc()  # Print the full stack trace
                time.sleep(15)
notifySelf("Gathered data successfully.")
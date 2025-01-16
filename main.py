#%%
from utils.util import *
import geopandas as gpd
from shapely.geometry import shape
import folium
import os
import sys
import time
import pandas as pd
from rasterio.merge import merge

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


def gatherData(dataset, year, city, aoi_geodf, isGroundTruth = False):
    truthString = "GroundTruth" if isGroundTruth else "InputData"
    print(f'Gathering {truthString} {dataset} for {year} in {city}.')
    bandNames = {'B2', 'B3', 'B4', 'B5', 'B6', 'ST_B10'}
    if isGroundTruth:
        bandNames = {'ST_B10'}
        return
    includeMetadata = True
    #%%
    # Cell 6: Search for Scenes
    if dataset == 'srtm_v3' and year != 2014:
        return
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
        listId = f"{dataset}_{year}_{str(month)}_{'truth' if isGroundTruth else 'notTruth'}"
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
        # print(products)
        #%%
        # Cell 10: Collect Files to Download
        downloads = []
        if 'landsat_ot_c2_l2' in dataset:
            for product in products:
                if product["secondaryDownloads"]:
                    for secDownload in product["secondaryDownloads"]:
                        if secDownload["bulkAvailable"] and any(band in secDownload['displayId'] for band in bandNames):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
                        if includeMetadata and secDownload['displayId'].endswith('_MTL.txt'):
                            downloads.append({"entityId": secDownload["entityId"], "productId": secDownload["id"]})
        elif 'srtm_v3' in dataset:
            for product in products:
                if product["bulkAvailable"] and product["entityId"] and product["id"]:
                    downloads.append({"entityId": product["entityId"], "productId": product["id"]})
        elif 'nlcd_collection_lndcov' in dataset:
            for product in products:
                if product["bulkAvailable"] and product["entityId"] and product["id"]:
                    downloads.append({"entityId": product["entityId"], "productId": product["id"]})
        #%%
        # Cell 11: Submit Download Request
        download_req_payload = {
            "downloads": downloads,
            "label": listId
        }
        download_request_results = sendRequest(serviceUrl + "download-request", download_req_payload, apiKey)
        #%%
        # print(download_request_results)
        #%%
        # Cell 12: Download Files
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
        # print("before join")
        for t in threads:
            t.join()
        # print("after join")
        for file in os.listdir('./Unprocessed'):
            if file.endswith('.tar'):
                try: 
                    extract_specific_files('./Unprocessed/' + file, './Unprocessed')
                except:
                    print(f'Error: Could not extract file {file}')
        #%%
        # Cell 13: Clean Up
        remove_scnlst_payload = {"listId": listId}
        sendRequest(serviceUrl + "scene-list-remove", remove_scnlst_payload, apiKey)
        #%%
        # Cell 15: Verify Downloads
        import rasterio
        from shapely.geometry import box
        
        print('Unprocessed Dir:', os.listdir(unprocessed_dir))
        tifs = [[],[],[],[],[],[],[],[]] # B2 B3 B4 B5 B6 B10 Land_Cover v3 
        for file in os.listdir(unprocessed_dir):
            if file.endswith(".TIF") or file.endswith(".tif"):
                if '1arc_v3' in file:
                    tifs[7].append(file)
                elif 'Annual_NLCD' in file:
                    tifs[6].append(file)
                else:
                    date, band, coordinate = getMetaFromLandsatTIRs(file)
                    if band == 'B10':
                        tifs[5].append(file)
                    elif band == 'B2':
                        tifs[0].append(file)
                    elif band == 'B3':
                        tifs[1].append(file)
                    elif band == 'B4':
                        tifs[2].append(file)
                    elif band == 'B5':
                        tifs[3].append(file)
                    elif band == 'B6':
                        tifs[4].append(file)
        for tifs_to_merge in tifs:
            if tifs_to_merge:
                for tif in tifs_to_merge:
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
        # for tifs_to_merge in tifs:
        #     if tifs_to_merge:
        #         output_path = unprocessed_dir + '/' + tifs_to_merge[0]  # Overwrite the first file in the list
        #
        #         # Extract metadata and colormap from the first source
        #         with rasterio.open(unprocessed_dir + '/' + tifs_to_merge[0]) as src:
        #             meta = src.meta
        #             try:
        #                 color_map = src.colormap(1)  # Assuming single-band with colormap
        #             except ValueError:
        #                 color_map = None  # Handle files without a colormap
        #
        #         # Keep the datasets open in the sources list
        #         sources = [rasterio.open(unprocessed_dir + '/' + tif) for tif in tifs_to_merge]
        #
        #         # Perform the merge
        #         merged_array, merged_transform = merge(sources)
        #         meta.update({
        #             "driver": "GTiff",
        #             "height": merged_array.shape[1],
        #             "width": merged_array.shape[2],
        #             "transform": merged_transform,
        #             "count": merged_array.shape[0]  # Number of bands
        #         })
        #
        #         # Close all opened datasets
        #         for source in sources:
        #             source.close()
        #
        #         # Write the merged raster with the preserved colormap
        #         with rasterio.open(output_path, "w", **meta) as dest:
        #             dest.write(merged_array)
        #             if color_map:  # Apply the colormap if it exists
        #                 dest.write_colormap(1, color_map)
        #
        #         print(f"Overwritten: {output_path}")
        usableTIFFs, aoi_geodf_proj = [], aoi_geodf.to_crs("EPSG:4326")
        for tifsToMerge in tifs:
            if len(tifsToMerge) == 0:
                continue
            for tif in tifsToMerge:
                tif_path = unprocessed_dir + '/' + tif
                with rasterio.open(tif_path) as src:
                    tif_bounds = box(*src.bounds)
                    # Check containment
                    for idx, geom in enumerate(aoi_geodf_proj.geometry):
                        if tif_bounds.contains(geom):
                            usableTIFFs.append(tif)
                            print(f"Polygon {city} is fully inside {tif}")
                        else:
                            print(f"Polygon {city} is NOT fully inside {tif}")
        #%%
        # goodCoordinates = clipUnprocessedRasters(usableTIFFs, aoi_geodf_proj)
        #%%
        for file in os.listdir(unprocessed_dir):
            if ".txt" in file or ".tif" in file or ".TIF" in file:
                if '1arc_v3' in file:
                    moveToRaw(file, 'DEM', f'{year}-01-01', city)
                    continue
                if 'Annual_NLCD' in file:
                    moveToRaw(file, 'Land_Cover', f'{year}-01-01', city)
                    continue
                date, band, coordinate = getMetaFromLandsatTIRs(file)
                # if coordinate not in goodCoordinates:
                #     continue
                print(date, city, band)
                if band == 'MTL':
                    if isGroundTruth:
                        moveToRaw(file, 'oli_label', date, city)
                    else:
                        moveToRaw(file, 'oli', date, city)
                if band == 'B10':
                    if isGroundTruth:
                        moveToRaw(file, 'oli_label', date, city)
                    else:
                        moveToRaw(file, 'oli', date, city)
                if band == 'B2':
                    moveToRaw(file, 'oli', date, city)
                if band == 'B3':
                    moveToRaw(file, 'oli', date, city)
                if band == 'B4':
                    moveToRaw(file, 'oli', date, city)
                if band == 'B5':
                    moveToRaw(file, 'oli', date, city)
                if band == 'B6':
                    moveToRaw(file, 'oli', date, city)

                # if band == 'B10':
                #     if isGroundTruth:
                #         moveToRaw(file, 'labelLST', date, city)
                #         continue
                #     else:
                #         moveToRaw(file, 'LST', date, city)
                # if band == 'B2':
                #     moveToRaw(file, 'Albedo', date, city)
                # if band == 'B3':
                #     moveToRaw(file, 'NDWI', date, city)
                # if band == 'B4':
                #     moveToRaw(file, 'Albedo', date, city)
                #     moveToRaw(file, 'NDVI', date, city)
                # if band == 'B5':
                #     moveToRaw(file, 'Albedo', date, city)
                #     moveToRaw(file, 'NDVI', date, city)
                #     moveToRaw(file, 'NDWI', date, city)
                # if band == 'B6':
                #     moveToRaw(file, 'Albedo', date, city)
                # if band == 'MTL':
                #     moveToRaw(file, 'Albedo', date, city)
                #     moveToRaw(file, 'LST', date, city)
                #     moveToRaw(file, 'NDVI', date, city)
                #     moveToRaw(file, 'NDWI', date, city)
        #%%
        print('Finished moving')
        if dataset == 'nlcd_collection_lndcov' or dataset == 'srtm_v3':
            break
    truthString = "InputData"
    if isGroundTruth:
        truthString = "GroundTruth"
    with open('progress.txt', "a") as file:
        file.write(str(truthString + ":" + str(city) + ":" + str(year) + ":" + dataset + "\n"))
    print('progress written for', truthString, city, year, dataset)
#%%
datasets = ['landsat_ot_c2_l2', 'srtm_v3', 'nlcd_collection_lndcov', 'landsat_ot_c2_l2'] 
years = [year for year in range(start_year, end_year)]

# Cell 2: Load shape file
shapefile_folder = "./Data/area_shp/"
shapeGroundTruth_folder = "./Data/label_shp/"
cities = []
citiesT = []
aoi_geodfs = []
aoi_geodfsTruth = []
for file in os.listdir(shapefile_folder):
    if file.endswith(".shp"):
        cities.append(file.replace('Polygon_', '').replace('.shp', ''))
        aoi_geodf = gpd.read_file(shapefile_folder + file)
        aoi_geodf = aoi_geodf.to_crs("EPSG:4326")
        if aoi_geodf.empty:
            sys.exit("Error: Shapefile contains no data.")
        aoi_geodfs.append(aoi_geodf)
for file in os.listdir(shapeGroundTruth_folder):
    if file.endswith(".shp"):
        citiesT.append(file.replace('Polygon_', '').replace('.shp', ''))
        aoi_geodf = gpd.read_file(shapefile_folder + file)
        aoi_geodf = aoi_geodf.to_crs("EPSG:4326")
        if aoi_geodf.empty:
            sys.exit("Error: Shapefile contains no data.")
        aoi_geodfsTruth.append(aoi_geodf)
print("Shapefiles loaded successfully.")
import traceback
for j, dataset in enumerate(datasets):
    for year in years:

        i = 0
        while i < len(cities):
            if i % 7 == 0:
                notifySelf(f'Starting on year {year} in dataset {dataset} as city {cities[i]}...')
            try:
                clear_folder(unprocessed_dir)
                assert len(os.listdir(unprocessed_dir)) == 0, "Unprocessed directory is not empty."
                isGroundTruth = j == len(datasets) - 1
                truthString = "GroundTruth" if isGroundTruth else "InputData"
                city, aoi_geodf = citiesT[i] if isGroundTruth else cities[i], aoi_geodfsTruth[i] if isGroundTruth else aoi_geodfs[i]
                if os.path.exists('progress.txt'):
                    with open('progress.txt', 'r') as file:
                        progress = [line.split(':') for line in file.read().strip().split('\n')]
                    if any(truthString == instance[0] and city == instance[1] and str(year) == instance[2] and dataset == instance[3] for instance in progress):
                        print(f"{truthString}, {city}, {year}, {dataset} was gathered in the past.")
                    else:
                        gatherData(dataset, year, city, aoi_geodf, isGroundTruth)
                i += 1
            except Exception as e:
                print("An exception occurred:")
                print(f"Exception: {e}")
                notifySelf("An exception occurred:")
                notifySelf(f"Exception: {e}")
                traceback.print_exc()  # Print the full stack trace
                time.sleep(15)
print("Gathered data successfully.")
notifySelf("Gathered data successfully.")

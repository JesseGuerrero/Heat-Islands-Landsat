import json
import requests
import sys
import time
import re
import threading
import datetime
import socket
import rasterio
import os
import warnings

import tarfile
import shutil
import numpy as np
from datetime import datetime
from tqdm import tqdm
from pyproj import Transformer
from shapely.geometry import Polygon
from geopandas import GeoDataFrame
from utils.voice import notifySelf
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray")

maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
threads = []
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
temp_dir = './Temp'
unprocessed_dir = temp_dir + '/Temp'
raw_dir = temp_dir + '/RawRasters'
clipped_dir = temp_dir + '/Clipped'
process_dir = temp_dir + '/Process'
data_dir = './Data'

for i, log in enumerate(['credentials.txt', 'voice.txt', 'raw_progress.txt', 'clipped_processed.txt', 'formula_progress.txt']):
    if not os.path.exists('./Logs/' + log):
        if i == 0:
            print('In credentials place...\n---\nUsername\nToken\n\n---')
            sys.exit()
        if i == 1:
            print('Optional Google Voice Notification:\nPlace the following in voice.txt\n---\nsubject\nto_email\nfrom_email\napiToken\n---')
        with open('./Logs/' + log, "w") as file:
            pass

def getLongitudeLatitudeOfTif(filePath) -> list:
	# Extract raster bounds using rasterio
	with rasterio.open(filePath) as src:
		bounds = src.bounds
		crs = src.crs

	# Convert bounds to latitude and longitude if needed
	transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
	min_lon, min_lat = transformer.transform(bounds.left, bounds.bottom)
	max_lon, max_lat = transformer.transform(bounds.right, bounds.top)

	# Calculate center of the raster
	latitude = (min_lat + max_lat) / 2
	longitude = (min_lon + max_lon) / 2
	return [longitude, latitude]

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

def getMetaFromLandsatTIRs(fileName) -> tuple:
	date = datetime.strptime(fileName.split('_')[3], "%Y%m%d").strftime("%Y-%m-%d")
	band = fileName.split('_')[-1].replace('.TIF', '').replace('.txt', '').replace('.tif', '')
	coordinates = fileName.split('_')[2]
	return date, band, coordinates

def moveToRaw(file: str, typeFolder: str, date, city):
	filePath = os.path.join(unprocessed_dir, file)
	dateFolder = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m")
	target_folder = os.path.join(raw_dir, typeFolder, city, dateFolder)
	os.makedirs(target_folder, exist_ok=True)
	target_file_path = os.path.join(target_folder, file)
	shutil.copy2(filePath, target_file_path)

def moveToClipped(filePath: str, fileName, typeFolder: str, date, city):
    target_folder = os.path.join(clipped_dir, typeFolder, city, date)
    os.makedirs(target_folder, exist_ok=True)
    target_file_path = os.path.join(target_folder, fileName)
    if os.path.exists(target_file_path):
        return
    shutil.copy2(filePath, target_file_path)

def clear_folder(folder_path):
    os.makedirs(folder_path, exist_ok=True)
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            os.rmdir(file_path)

def get_file_paths(folder_path):
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            full_path = os.path.abspath(os.path.join(root, file))
            file_paths.append(full_path)
    return file_paths

def save_paths_to_log(file_paths, log_file="./Logs/clipped_processed.txt"):
    with open(log_file, "a") as log:
        for path in tqdm(file_paths, desc="Saving to log"):
            log.write(path + "\n")  # Write each path followed by a newline
    print(f"File paths saved to {log_file}")

def read_file_paths_from_log(log_file="./Logs/clipped_processed.txt"):
    with open(log_file, "r") as log:
        file_paths = [line.strip() for line in log]  # Remove any leading/trailing whitespace
    return file_paths

def sendRequest(url, data, apiKey=None, exitIfNoResponse=True): #Official sendRequest script from USGS website
	json_data = json.dumps(data)

	if apiKey == None:
		response = requests.post(url, json_data)
	else:
		headers = {'X-Auth-Token': apiKey}
		response = requests.post(url, json_data, headers=headers)

	try:
		httpStatusCode = response.status_code
		if response == None:
			print("No output from service")
			if exitIfNoResponse:
				sys.exit()
			else:
				return False
		output = json.loads(response.text)
		if output['errorCode'] != None:
			print(output['errorCode'], "- ", output['errorMessage'])
			if exitIfNoResponse:
				sys.exit()
			else:
				return False
		if httpStatusCode == 404:
			print("404 Not Found")
			if exitIfNoResponse:
				sys.exit()
			else:
				return False
		elif httpStatusCode == 401:
			print("401 Unauthorized")
			if exitIfNoResponse:
				sys.exit()
			else:
				return False
		elif httpStatusCode == 400:
			print("Error Code", httpStatusCode)
			if exitIfNoResponse:
				sys.exit()
			else:
				return False
	except Exception as e:
		response.close()
		print(e)
		if exitIfNoResponse:
			sys.exit()
		else:
			return False
	response.close()
	return output['data']

def extract_specific_files(tar_path, extract_to, include_keywords=None):
    with tarfile.open(tar_path, "r") as tar:
        for member in tar.getmembers():
            if include_keywords is None or any(keyword in member.name for keyword in include_keywords):
                tar.extract(member, extract_to)
                print(f"Extracted: {member.name}")

def runDownload(threads, url):
    thread = threading.Thread(target=downloadFile, args=(url,))
    threads.append(thread)
    thread.start()

def downloadFile(url):
    """Slightly edited official M2M Function. We can change this if desired."""
    sema.acquire()
    try:
        while True:
            response = requests.get(url, stream=True)
            if response.status_code == 429:
                print("HTTP 429: Rate limit reached during download. Waiting 16 minutes before retrying...")
                time.sleep(16 * 60)  # Wait for 16 minutes
                continue
            elif response.status_code == 200:
                disposition = response.headers.get('content-disposition', '')
                filename = re.findall("filename=(.+)", disposition)
                filename = filename[0].strip("\"") if filename else "unknown_file"
                print(f"Downloading: {filename}...")
                with open(os.path.join(unprocessed_dir, filename), 'wb') as file:
                    file.write(response.content)
                break
            else:
                print(f"Failed to download from {url}. HTTP Status: {response.status_code}. Retrying...")
                time.sleep(10)  # Retry after 10 seconds
    except Exception as e:
        print(f"Failed to download from {url} due to error: {e}")
    finally:
        sema.release()

def prompt_ERS_login():
    print("Logging in...\n")
    notifySelf("Logging in...")

    # Read credentials from the file
    with open('./Logs/credentials.txt', 'r') as file:
        username = file.readline().strip()
        token = file.readline().strip()

    # Use requests.post() to make the login request
    response = requests.post(f"{serviceUrl}login-token", json={
        'username': username,
        'token': token
    })

    if response.status_code == 200:  # Check for successful response
        apiKey = response.json().get('data')
        print('\nLogin Successful, API Key Received!')
        notifySelf("Login Successful, API Key Received!")
        headers = {'X-Auth-Token': apiKey}
        return apiKey
    else:
        print("\nLogin was unsuccessful, please try again or create an account at: https://ers.cr.usgs.gov/register.")

def createSceneSearchPayload(datasetName, aoi_geodf, year, month, cloudMax=15):
    month = str(month)
    if len(month) == 1:
        month = "0" + month
    spatialFilter = {
        'filterType': 'mbr',
        'lowerLeft': {
            'latitude': aoi_geodf.geometry.bounds.miny[0],
            'longitude': aoi_geodf.geometry.bounds.minx[0]
        },
        'upperRight': {
            'latitude': aoi_geodf.geometry.bounds.maxy[0],
            'longitude': aoi_geodf.geometry.bounds.maxx[0]
        }
    }
    cloudCoverFilter = {'min': 0, 'max': cloudMax}
    if datasetName == 'landsat_ot_c2_l2':
        temporal = {'start': f'{year}-{month}-01', 'end': f'{year}-{month}-31'}
    elif datasetName == 'nlcd_collection_lndcov':
        temporal = {'start': f'{year}-01-01', 'end': f'{year}-12-31'}
    else:
        temporal = {}
    return {
        'datasetName': datasetName,
        'sceneFilter': {
            'spatialFilter': spatialFilter,
            'acquisitionFilter': temporal,
            'cloudCoverFilter': cloudCoverFilter
        }
    }

def calculate_cloud_cover_percentage(qa_pixel_path):
    with rasterio.open(qa_pixel_path) as src:
        qa_pixel = src.read(1)  # Read the first band (QA_PIXEL)
        nodata_value = src.nodata  # Get NoData value from raster metadata

    if nodata_value is not None:
        nodata_mask = (qa_pixel == nodata_value)
    else:
        nodata_mask = np.zeros_like(qa_pixel, dtype=bool)  # No NoData pixels

    cloud_confidence_mask = (qa_pixel & 0b00011000) >> 3
    cloud_shadow_mask = (qa_pixel & 0b00100000) >> 5
    cloud_pixels = (cloud_confidence_mask >= 1) | (cloud_shadow_mask == 1)

    valid_pixels = ~nodata_mask
    cloud_pixels = cloud_pixels & valid_pixels

    total_valid_pixels = np.sum(valid_pixels)
    cloud_pixel_count = np.sum(cloud_pixels)

    if total_valid_pixels > 0:
        cloud_cover_percentage = (cloud_pixel_count / total_valid_pixels) * 100
    else:
        cloud_cover_percentage = 0.0

    return int(cloud_cover_percentage)

apiKey = prompt_ERS_login()
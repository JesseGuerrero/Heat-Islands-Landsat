import os
os.environ['PROJ_LIB'] = r'C:\OSGeo4W\share\proj'
import json
import requests
from getpass import getpass
import sys
import time
import re
import threading
import datetime
import pandas as pd
from geojson import Polygon, Feature, FeatureCollection, dump
import geopandas as gpd
import rioxarray

maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
threads = []
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
unprocessed_dir = './Unprocessed'
raw_dir = './RawClippedRasters'
data_dir = './Data'
dirs = [unprocessed_dir, data_dir, raw_dir,
		data_dir + "/LST", data_dir + "/NDVI", data_dir + "/NDWI", data_dir + "/Land_Cover", data_dir + "/Albedo", data_dir + "/DEM",
		raw_dir + "/LST", raw_dir + "/NDVI", raw_dir + "/NDWI", raw_dir + "/Land_Cover", raw_dir + "/Albedo", raw_dir + "/DEM"]
for d in dirs:
	if not os.path.exists(d):
		try:
			os.makedirs(d)
			print(f"Directory '{d}' created successfully.")
		except OSError as e:
			print(f"Error creating directory '{d}': {e}")
	else:
		print(f"Directory '{d}' already exists.")

import rasterio
from pyproj import Transformer
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

import os
import rioxarray
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="xarray")
def clipUnprocessedRasters(tifs, boundPolygon):
    goodCoordinates = []
    for tif in tifs:
        tif_path = unprocessed_dir + '/' + tif

        # Open the raster using rioxarray
        raster = rioxarray.open_rasterio(tif_path)

        # Extract colormap from the original raster
        colormap = None
        with rasterio.open(tif_path) as src:
            if src.colorinterp[0] == rasterio.enums.ColorInterp.palette:
                colormap = src.colormap(1)  # Assuming band 1 has the colormap

        # Clip the raster using the bounding polygon
        clipped = raster.rio.clip(boundPolygon.geometry, boundPolygon.crs, drop=True)

        # Reproject to EPSG:4326
        reprojected = clipped.rio.reproject("EPSG:4326")

        # Save the output raster, preserving metadata
        clipped_file_path = unprocessed_dir + f"/Clipped_{tif}"
        reprojected.rio.to_raster(clipped_file_path)

        # Reapply colormap to the saved raster if it exists
        if colormap:
            with rasterio.open(clipped_file_path, "r+") as dest:
                dest.write_colormap(1, colormap)

        # Add the coordinate info from filename to the list
        goodCoordinates.append(tif.split('_')[2])
        print(f"Clipped, reprojected, and color-preserved TIF saved as {clipped_file_path}")

    return goodCoordinates


from datetime import datetime
def getMetaFromLandsatTIRs(fileName) -> tuple:
	date = datetime.strptime(fileName.split('_')[(4 if 'Clipped_' in fileName else 3)], "%Y%m%d").strftime("%Y-%m-%d")
	band = fileName.split('_')[-1].replace('.TIF', '').replace('.txt', '')
	coordinates = fileName.split('_')[(3 if 'Clipped_' in fileName else 2)]
	return date, band, coordinates

import shutil
def moveToRaw(file: str, typeFolder: str, date, city):
	filePath = os.path.join(unprocessed_dir, file)
	dateFolder = datetime.strptime(date, "%Y-%m-%d").strftime("%m-%Y")
	target_folder = os.path.join(raw_dir,  typeFolder, dateFolder, city)
	os.makedirs(target_folder, exist_ok=True)
	target_file_path = os.path.join(target_folder, file)
	shutil.copy2(filePath, target_file_path)

def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        elif os.path.isdir(file_path):
            os.rmdir(file_path)

def sendRequest(url, data, apiKey=None, exitIfNoResponse=True):
	"""
	Send a request to an M2M endpoint and returns the parsed JSON response.

	Parameters:
	endpoint_url (str): The URL of the M2M endpoint
	payload (dict): The payload to be sent with the request

	Returns:
	dict: Parsed JSON response
	"""

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

import tarfile
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
	sema.acquire()
	try:
		response = requests.get(url, stream=True)
		disposition = response.headers['content-disposition']
		filename = re.findall("filename=(.+)", disposition)[0].strip("\"")
		print(f"    Downloading: {filename}...")

		open(os.path.join(unprocessed_dir, filename), 'wb').write(response.content)
		sema.release()
	except Exception as e:
		print(f"\nFailed to download from {url}. Will try to re-download.")
		sema.release()
		runDownload(threads, url)

def prompt_ERS_login():
    print("Logging in...\n")

    # Read credentials from the file
    with open('credentials.txt', 'r') as file:
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
        headers = {'X-Auth-Token': apiKey}
        return apiKey
    else:
        print("\nLogin was unsuccessful, please try again or create an account at: https://ers.cr.usgs.gov/register.")

def createSceneSearchPayload(datasetName, aoi_geodf, year, cloudMax):
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
    if datasetName == 'landsat_ot_c2_l2' or datasetName == 'ccdc_v1_3':
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

apiKey = prompt_ERS_login()
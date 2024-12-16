import json
import requests
from getpass import getpass
import sys
import time
import re
import threading
import datetime
import os
import pandas as pd
from geojson import Polygon, Feature, FeatureCollection, dump
import geopandas as gpd

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

from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import os
def clipUnprocessedRasters(tifs, boundPolygon):
	for tif in tifs:
		tif_path = os.path.join(unprocessed_dir, tif)
		with rasterio.open(tif_path) as clip_src:

			# Mask the original file
			out_image, out_transform = mask(clip_src, [boundPolygon.geom], crop=True)

			# Update metadata after masking
			masked_meta = clip_src.meta.copy()
			masked_meta.update({
				"driver": "GTiff",
				"height": out_image.shape[1],
				"width": out_image.shape[2],
				"transform": out_transform
			})

			# Calculate bounds from the masked raster
			left, bottom = out_transform * (0, out_image.shape[1])
			right, top = out_transform * (out_image.shape[2], 0)

			# Reproject the masked raster to EPSG:4326
			dst_crs = 'EPSG:4326'
			transform, width, height = calculate_default_transform(
				clip_src.crs, dst_crs, out_image.shape[2], out_image.shape[1], left, bottom, right, top
			)

			reprojected_meta = masked_meta.copy()
			reprojected_meta.update({
				"crs": dst_crs,
				"transform": transform,
				"width": width,
				"height": height
			})

			with rasterio.MemoryFile() as memfile:
				with memfile.open(**reprojected_meta) as reprojected:
					for i in range(1, clip_src.count + 1):
						reproject(
							source=out_image[i - 1],  # Bands are 0-indexed in the array
							destination=rasterio.band(reprojected, i),
							src_transform=out_transform,
							src_crs=clip_src.crs,
							dst_transform=transform,
							dst_crs=dst_crs,
							resampling=Resampling.nearest
						)

					# Save the final reprojected file
					clipped_file_path = os.path.join(unprocessed_dir, f"Clipped_{tif}")
					with rasterio.open(clipped_file_path, "w", **reprojected_meta) as dest:
						dest.write(reprojected.read())
						print(f"Masked and reprojected TIF saved as {clipped_file_path}")

from datetime import datetime
def getMetaFromLandsatTIRs(fileName) -> tuple:
	date = datetime.strptime(fileName['entityId'].split('_')[4], "%Y%m%d").strftime("%Y-%m-%d")
	band = fileName['entityId'].split('_')[-2]
	return date, band

import shutil
def moveToRaw(file: str, typeFolder: str, date, city, band):
	filePath = os.path.join(unprocessed_dir, file)
	dateFolder = datetime.strptime(date, "%Y-%m-%d").strftime("%m-%Y")
	target_folder = os.path.join(raw_dir,  typeFolder, dateFolder, city)
	os.makedirs(target_folder, exist_ok=True)
	target_file_path = os.path.join(target_folder, file)
	shutil.copy2(filePath, target_file_path)

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

def runDownload(threads, url):
    thread = threading.Thread(target=downloadFile, args=(url))
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


apiKey = prompt_ERS_login()
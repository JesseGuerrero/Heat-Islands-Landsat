import os
# os.environ['PROJ_LIB'] = r'C:\OSGeo4W\share\proj'
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
		data_dir + "/LST", data_dir + "/NDVI", data_dir + "/NDWI", data_dir + "/Land_Cover", data_dir + "/Albedo", data_dir + "/DEM", data_dir + "/labelLST",
		raw_dir + "/LST", raw_dir + "/NDVI", raw_dir + "/NDWI", raw_dir + "/Land_Cover", raw_dir + "/Albedo", raw_dir + "/DEM", raw_dir + "/labelLST"]
for d in dirs:
	if not os.path.exists(d):
		try:
			os.makedirs(d)
			print(f"Directory '{d}' created successfully.")
		except OSError as e:
			print(f"Error creating directory '{d}': {e}")
	else:
		print(f"Directory '{d}' already exists.")
if not os.path.exists('progress.txt'):
	with open('progress.txt', "w") as file:
		pass
if not os.path.exists('credentials.txt'):
	print('In credentials place...\n---\nUsername\nToken\n---')
	with open('credentials.txt', "w") as file:
		pass
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

import smtplib
from email.mime.text import MIMEText

def notifySelf(body):
	try:
		subject = "Jupyter Notebook Cell Output"
		to_email = "19152478204.19152478204.ZXZzbrC-TB@txt.voice.google.com"
		from_email = "jesseguerrero1991@gmail.com"
		password = "yhts ilcq ymyp nfjv"  # Use an app password for Gmail or similar
		msg = MIMEText(body)
		msg["Subject"] = subject
		msg["From"] = from_email
		msg["To"] = to_email
		with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
			server.login(from_email, password)
			server.sendmail(from_email, to_email, msg.as_string())
	except Exception as e:
		print("unable to send text...")
		pass

def clipUnprocessedRasters(tifs, boundPolygon):
	# Reproject the bounding polygon to EPSG:4326
	if boundPolygon.crs != "EPSG:4326":
		# print("Reprojecting bounding polygon to EPSG:4326...")
		boundPolygon = boundPolygon.to_crs("EPSG:4326")

	goodCoordinates = []
	for tif in tifs:
		if 'Clipped_' in tif:
			continue
		tif_path = os.path.join(unprocessed_dir, tif)

		# Open the raster using rioxarray
		raster = rioxarray.open_rasterio(tif_path)
		# print(f"Opened raster: {tif}, Original CRS: {raster.rio.crs}")

		# Reproject the raster to EPSG:4326
		raster_reprojected = raster.rio.reproject("EPSG:4326")
		# print(f"Raster reprojected to EPSG:4326 for clipping.")

		# Extract colormap from the original raster
		colormap = None
		with rasterio.open(tif_path) as src:
			if src.colorinterp[0] == rasterio.enums.ColorInterp.palette:
				colormap = src.colormap(1)  # Assuming band 1 has the colormap

		# Clip the raster using the bounding polygon
		clipped = raster_reprojected.rio.clip(boundPolygon.geometry, boundPolygon.crs, drop=True)
		# print(f"Clipping completed for raster: {tif}")

		# Save the output raster, preserving metadata
		clipped_file_path = os.path.join(unprocessed_dir, f"Clipped_{tif}")
		clipped.rio.to_raster(clipped_file_path)
		# print(f"Clipped raster saved at: {clipped_file_path}")

		# Reapply colormap to the saved raster if it exists
		if colormap:
			with rasterio.open(clipped_file_path, "r+") as dest:
				dest.write_colormap(1, colormap)
			# print(f"Colormap reapplied to raster: {clipped_file_path}")

		# Add the coordinate info from filename to the list
		goodCoordinates.append(tif.split('_')[2])
		raster.close()
		raster_reprojected.close()
		print(f"Clipped, reprojected, and color-preserved TIF saved as {clipped_file_path}")

	return goodCoordinates


from datetime import datetime
def getMetaFromLandsatTIRs(fileName) -> tuple:
	date = datetime.strptime(fileName.split('_')[(4 if 'Clipped_' in fileName else 3)], "%Y%m%d").strftime("%Y-%m-%d")
	band = fileName.split('_')[-1].replace('.TIF', '').replace('.txt', '').replace('.tif', '')
	coordinates = fileName.split('_')[(3 if 'Clipped_' in fileName else 2)]
	return date, band, coordinates

import shutil
def moveToRaw(file: str, typeFolder: str, date, city):
	filePath = os.path.join(unprocessed_dir, file)
	dateFolder = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m")
	target_folder = os.path.join(raw_dir, typeFolder, city, dateFolder)
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
    """Download a file and handle HTTP 429 (rate limitation)."""
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

apiKey = prompt_ERS_login()
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
import warnings
from geojson import Polygon, Feature, FeatureCollection, dump
import geopandas as gpd
#
# warnings.filterwarnings("ignore")

maxthreads = 5 # Threads count for downloads
sema = threading.Semaphore(value=maxthreads)
label = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Customized label using date time
threads = []
serviceUrl = "https://m2m.cr.usgs.gov/api/api/json/stable/"
data_dir = 'Data'
utils_dir = 'utils'
dirs = [ data_dir, utils_dir]
for d in dirs:
        if not os.path.exists(d):
            try:
                os.makedirs(d)
                print(f"Directory '{d}' created successfully.")
            except OSError as e:
                print(f"Error creating directory '{d}': {e}")
        else:
            print(f"Directory '{d}' already exists.")

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

		open(os.path.join(data_dir, filename), 'wb').write(response.content)
		sema.release()
	except Exception as e:
		print(f"\nFailed to download from {url}. Will try to re-download.")
		sema.release()
		runDownload(threads, url)

def prompt_ERS_login():
    print("Logging in...\n")

    p = ['Enter EROS Registration System (ERS) Username: ', 'Enter ERS Account Token: ']

    # Use requests.post() to make the login request
    response = requests.post(f"{serviceUrl}login-token", json={'username': getpass(prompt=p[0]), 'token': getpass(prompt=p[1])})

    if response.status_code == 200:  # Check for successful response
        apiKey = response.json()['data']
        print('\nLogin Successful, API Key Received!')
        headers = {'X-Auth-Token': apiKey}
        return apiKey
    else:
        print("\nLogin was unsuccessful, please try again or create an account at: https://ers.cr.usgs.gov/register.")

apiKey = prompt_ERS_login()
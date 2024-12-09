#%%
from utils.util import *
from shapely.geometry import Point
fileType = 'band'
bandNames = {''''SR_B4', 'SR_B5', 'ST_B10',''' '_MTL.txt'}
point = Point(-149.8555, 61.2433)
#%%
features = []
features = [Feature(geometry=point, properties={"city": "Anchorage", "state": "Alaska"})]

feature_collection = FeatureCollection(features)

with open('./utils/Anchorage_Alaska_aoi.geojson', 'w') as f:
    dump(feature_collection, f)
    
aoi_geodf =  gpd.read_file('./utils/Anchorage_Alaska_aoi.geojson') #aoi geopandas dataframe
aoi_geodf.crs
#%%
import folium
m = folium.Map(location=[aoi_geodf.geometry.y[0], aoi_geodf.geometry.x[0]], zoom_start=12, tiles="openstreetmap", 
              width="90%", height="90%", attributionControl=0)
#%%
folium.Marker([aoi_geodf.geometry.y[0], aoi_geodf.geometry.x[0]], popup="Anchorage").add_to(m)
m
#%%
datasetName = 'landsat_ot_c2_l2'
#%%
# Corrected spatial filter using a small circular area
spatialFilter = {
    'filterType': 'circle',
    'centerPoint': {
        'latitude': aoi_geodf.geometry.y[0],
        'longitude': aoi_geodf.geometry.x[0]
    },
    'radius': 0.01  # Small radius in kilometers
}

temporalFilter = {'start' : '2020-03-01', 'end' : '2020-03-15'}
cloudCoverFilter = {'min' : 0, 'max' : 20}
search_payload = {
    'datasetName' : datasetName,
    'sceneFilter' : {
        'spatialFilter' : spatialFilter,
        'acquisitionFilter' : temporalFilter,
        'cloudCoverFilter' : cloudCoverFilter
    }
}
search_payload
#%%
scenes = sendRequest(serviceUrl + "scene-search", search_payload, apiKey)
pd.json_normalize(scenes['results'])
#%%
idField = 'entityId'
entityIds = []
for result in scenes['results']:
    if result['options']['bulk'] == True:
        entityIds.append(result[idField])
entityIds
#%%
listId = f"temp_{datasetName}_list" # customized list id
scn_list_add_payload = {
    "listId": listId,
    'idField' : idField,
    "entityIds": entityIds,
    "datasetName": datasetName
}
scn_list_add_payload
#%%
count = sendRequest(serviceUrl + "scene-list-add", scn_list_add_payload, apiKey) 
count
#%%
sendRequest(serviceUrl + "scene-list-get", {'listId' : scn_list_add_payload['listId']}, apiKey) 
#%%
download_opt_payload = {
    "listId": listId,
    "datasetName": datasetName
}

if fileType == 'band_group':
    download_opt_payload['includeSecondaryFileGroups'] = True

download_opt_payload
#%%
products = sendRequest(serviceUrl + "download-options", download_opt_payload, apiKey)
pd.json_normalize(products)
#%%
filegroups = sendRequest(serviceUrl + "dataset-file-groups", {'datasetName' : datasetName}, apiKey)  
pd.json_normalize(filegroups['secondary'])
#%%
fileGroupIds = {"ls_c2l2_sr_band"}
#%%

# Select products
print("Selecting products...")
downloads = []
if fileType == 'bundle':
    # Select bundle files
    print("    Selecting bundle files...")
    for product in products:        
        if product["bulkAvailable"] and product['downloadSystem'] != 'folder':               
            downloads.append({"entityId":product["entityId"], "productId":product["id"]})


elif fileType == 'band':
    # Select band files
    print("    Selecting band files...")
    for product in products:  
        if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
            for secondaryDownload in product["secondaryDownloads"]:
                for bandName in bandNames:
                    if secondaryDownload["bulkAvailable"] and bandName in secondaryDownload['displayId']:
                        downloads.append({"entityId":secondaryDownload["entityId"], "productId":secondaryDownload["id"]})


elif fileType == 'band_group':        
    # Get secondary dataset ID and file group IDs with the scenes
    print("    Checking for scene band groups and get secondary dataset ID and file group IDs with the scenes...")
    sceneFileGroups = []
    entityIds = []
    datasetId = None
    for product in products:  
        if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
            for secondaryDownload in product["secondaryDownloads"]:
                if secondaryDownload["bulkAvailable"] and secondaryDownload["fileGroups"] is not None:
                    if datasetId == None:
                        datasetId = secondaryDownload['datasetId']
                    for fg in secondaryDownload["fileGroups"]:                            
                        if fg not in sceneFileGroups:
                            sceneFileGroups.append(fg)
                        if secondaryDownload['entityId'] not in entityIds:
                            entityIds.append(secondaryDownload['entityId'])

    # Send dataset request to get the secondary dataset name by the dataset ID
    data_req_payload = {
        "datasetId": datasetId,
    }
    results = sendRequest(serviceUrl + "dataset", data_req_payload, apiKey)
    secondaryDatasetName = results['datasetAlias']

    # Add secondary scenes to a list
    secondaryListId = f"temp_{datasetName}_scecondary_list" # customized list id
    sec_scn_add_payload = {
        "listId": secondaryListId,
        "entityIds": entityIds,
        "datasetName": secondaryDatasetName
    }

    print("    Adding secondary scenes to list...")
    count = sendRequest(serviceUrl + "scene-list-add", sec_scn_add_payload, apiKey)    
    print("    Added", count, "secondary scenes\n")

    # Compare the provided file groups Ids with the scenes' file groups IDs
    if fileGroupIds:
        fileGroups = []
        for fg in fileGroupIds:
            fg = fg.strip() 
            if fg in sceneFileGroups:
                fileGroups.append(fg)
    else:
        fileGroups = sceneFileGroups
else:
    # Select all available files
    for product in products:        
        if product["bulkAvailable"]:
            if product['downloadSystem'] != 'folder':            
                downloads.append({"entityId":product["entityId"], "productId":product["id"]})
            if product["secondaryDownloads"] is not None and len(product["secondaryDownloads"]) > 0:
                for secondaryDownload in product["secondaryDownloads"]:
                    if secondaryDownload["bulkAvailable"]:
                        downloads.append({"entityId":secondaryDownload["entityId"], "productId":secondaryDownload["id"]})            

#%%
if fileType != 'band_group':
    download_req2_payload = {
        "downloads": downloads,
        "label": label
    }
else:
    if len(fileGroups) > 0:
        download_req2_payload = {
            "dataGroups": [
                {
                    "fileGroups": fileGroups,
                    "datasetName": secondaryDatasetName,
                    "listId": secondaryListId
                }
            ],
            "label": label
        }
    else:
        print('No file groups found')
        sys.exit()

print(f"Sending download request ...")
download_request_results = sendRequest(serviceUrl + "download-request", download_req2_payload, apiKey)
print(f"Done sending download request") 

if len(download_request_results['newRecords']) == 0 and len(download_request_results['duplicateProducts']) == 0:
    print('No records returned, please update your scenes or scene-search filter')
    sys.exit()

#%%
# Attempt the download URLs
for result in download_request_results['availableDownloads']:
    print(f"Get download url: {result['url']}\n" )
    runDownload(threads, result['url'])
    
preparingDownloadCount = len(download_request_results['preparingDownloads'])
preparingDownloadIds = []
if preparingDownloadCount > 0:
    for result in download_request_results['preparingDownloads']:  
        preparingDownloadIds.append(result['downloadId'])

    download_ret_payload = {"label" : label}                
    # Retrieve download URLs
    print("Retrieving download urls...\n")
    download_retrieve_results = sendRequest(serviceUrl + "download-retrieve", download_ret_payload, apiKey, False)
    if download_retrieve_results != False:
        print(f"    Retrieved: \n" )
        for result in download_retrieve_results['available']:
            if result['downloadId'] in preparingDownloadIds:
                preparingDownloadIds.remove(result['downloadId'])
                runDownload(threads, result['url'])
                print(f"       {result['url']}\n" )
            
        for result in download_retrieve_results['requested']:   
            if result['downloadId'] in preparingDownloadIds:
                preparingDownloadIds.remove(result['downloadId'])
                runDownload(threads, result['url'])
                print(f"       {result['url']}\n" )
    
    # Didn't get all download URLs, retrieve again after 30 seconds
    while len(preparingDownloadIds) > 0: 
        print(f"{len(preparingDownloadIds)} downloads are not available yet. Waiting for 30s to retrieve again\n")
        time.sleep(30)
        download_retrieve_results = sendRequest(serviceUrl + "download-retrieve", download_ret_payload, apiKey, False)
        if download_retrieve_results != False:
            for result in download_retrieve_results['available']:                            
                if result['downloadId'] in preparingDownloadIds:
                    preparingDownloadIds.remove(result['downloadId'])
                    print(f"    Get download url: {result['url']}\n" )
                    runDownload(threads, result['url'])
                    
print("\nDownloading files... Please do not close the program\n")
for thread in threads:
    thread.join()        
#%%
remove_scnlst_payload = {
    "listId": listId
}
sendRequest(serviceUrl + "scene-list-remove", remove_scnlst_payload, apiKey)

if fileType == 'band_group':    
    # Remove the secondary scene list
    remove_scnlst2_payload = {
        "listId": secondaryListId
    }
    sendRequest(serviceUrl + "scene-list-remove", remove_scnlst2_payload, apiKey)
#%%
endpoint = "logout"  
if sendRequest(serviceUrl + endpoint, None, apiKey) == None:        
    print("\nLogged Out\n")
else:
    print("\nLogout Failed\n")
#%%
os.listdir(data_dir)
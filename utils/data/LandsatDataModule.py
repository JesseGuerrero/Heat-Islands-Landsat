from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import rasterio
import torch
import pytorch_lightning as pl
import numpy as np
import rasterio
import cv2
import torch.nn as nn
import os
from datetime import datetime
from dateutil.relativedelta import relativedelta

class LandsatDataset(Dataset):
    def __init__(self, file_list, transform=None, nodata_fill_value=-9999.0):
        self.file_list = file_list
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value
        self.input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif']
        self.ranges = {
            'Albedo.tif': (-0.018, 0.998),           # Typical albedo range
            'DEM.tif': (-31.0, 2161.0),        # Approximate elevation range
            'Land_Cover.tif': (11.0, 95.0),      # Assuming land cover classes
            'NDVI.tif': (-1.0, 1.0),            # NDVI range
            'NDWI.tif': (-1.0, 1.0),            # NDWI range
            'LST.tif': (-80.9723, 211.73),             # Typical LST range in Fahrenheit
            'HeatIndex.tif': (1, 25)
        }

    def normalize(self, sample):
        x = sample['input']  # batch 5 512 512
        y = sample['target']  # batch 2 512 512
        
        # Normalize each input channel
        for i, channel_name in enumerate(self.input_keys):
            min_val, max_val = self.ranges[channel_name]           
            # Only normalize values within the range
            x[:, i:i+1, :, :][x!=-9999] = (x[:, i:i+1, :, :][x!=-9999] - min_val) / (max_val - min_val)
        
        # Normalize target LST
        min_val, max_val = self.ranges['LST.tif']
        y[:, 0:1, :, :][y!=-9999] = (y[:, 0:1, :, :][y!=-9999] - min_val) / (max_val - min_val)
        
        # Normalize target Heat Index
        min_val, max_val = self.ranges['HeatIndex.tif']        
        y[:, 1:2, :, :][y!=-9999] = (y[:, 1:2, :, :][y!=-9999] - min_val) / (max_val - min_val)
        
        return {'input': x, 'target': y, 'mask': sample['mask']}

    @staticmethod
    def denormalize(sample):
        if isinstance(sample, dict):
            x = sample['input'].clone()  # Clone to avoid modifying original
            y = sample['target'].clone()  # Clone to avoid modifying original
        else:
            x = -1
            y = sample.clone()
        ranges = {
            'Albedo.tif': (-0.018, 0.999),
            'DEM.tif': (-31.0, 2161.0),
            'Land_Cover.tif': (11.0, 95.0),
            'NDVI.tif': (-1.0, 1.0),
            'NDWI.tif': (-1.0, 1.0),
            'LST.tif': (28.0, 175.0),
            'HeatIndex.tif': (1, 25)
        }
        input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif']
        
        if x != -1:
            # Denormalize each input channel
            for i, channel_name in enumerate(input_keys):
                min_val, max_val = ranges[channel_name]
                mask = x[:, i:i+1, :, :] != -9999
                x[:, i:i+1, :, :][mask] = x[:, i:i+1, :, :][mask] * (max_val - min_val) + min_val
                
            # Denormalize target LST
            min_val, max_val = ranges['LST.tif']
            mask = y[:, 0:1, :, :] != -9999
            y[:, 0:1, :, :][mask] = y[:, 0:1, :, :][mask] * (max_val - min_val) + min_val
            
            # Denormalize target Heat Index
            min_val, max_val = ranges['HeatIndex.tif']
            mask = y[:, 1:2, :, :] != -9999
            y[:, 1:2, :, :][mask] = y[:, 1:2, :, :][mask] * (max_val - min_val) + min_val
        
        if x == -1:
            return y
        return {'input': x, 'target': y, 'mask': sample['mask']}

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_files = self.file_list[idx]

        channels = []
        channel_masks = []
        for key in self.input_keys:
            with rasterio.open(sample_files[key]) as src:
                channel = src.read(1).astype(np.float32) 
                valid_mask = ~np.isnan(channel) & (channel != -9999)
                channel = np.where(valid_mask, channel, -9999)
                channels.append(channel)
                channel_masks.append(valid_mask)
        # print(f"Sample scene: {sample_files}")
        x = np.stack(channels, axis=0)
        input_mask = np.stack(channel_masks, axis=0)

        with rasterio.open(sample_files['LST.tif']) as src:
            y = src.read(1).astype(np.float32)
            target_mask = ~np.isnan(y) & (y != -9999)
            y = np.where(target_mask, y, -9999)

        combined_mask = np.all(input_mask, axis=0) & target_mask
        
        for i in range(x.shape[0]):
            x[i] = np.where(combined_mask, x[i], -9999)
        y = np.where(combined_mask, y, -9999)

        y = np.expand_dims(y, axis=0)
        combined_mask = np.expand_dims(combined_mask, axis=0)

        sample = {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'mask': torch.from_numpy(combined_mask)
        }

        sample = self.normalize(sample)
        if self.transform:
            sample = self.transform(sample)

        if sample['input'].shape[1:] != sample['target'].shape[1:]:
            raise ValueError("Mismatch between input and target spatial dimensions.")
        return sample

class LandsatDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            monthsAhead: int = 0,
            batch_size: int = 1,
            num_workers: int = 2,
            train_ratio: float = 0.8,
            transform=None,
            byCity: bool = False,
            debug: bool = False,
            nodata_fill_value: float = -9999.0,
            normalize: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.monthsAhead = monthsAhead
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value
        self.byCity = byCity
        self.debug = debug
        self.normalize = normalize
        self.train_files = []
        self.val_files = []
        self.test_files = []
        

    def setup(self, stage=None):        
        if self.byCity:
            self.prepare_by_city()
        else:
            self.prepare_by_scene()


    def prepare_by_city(self):
        def sortCitiesToFileList(cities_for_task, all_cities):
            file_list = []
            for city in cities_for_task:
                for scene in all_cities[city].values():
                    file_list.append(scene)
            return file_list
        cities = {}
        albedo_files = []
        x_dir = os.path.join(self.data_dir, 'preprocess', 'X', 'less5CloudCover')
        for file_path in tqdm(self.get_file_paths(x_dir), desc='Gathering scenes(Sort by City)...'):
            date = file_path.split('/')[-2]
            if self.debug and '2014' not in date: 
                continue
            if 'Albedo' in file_path:
                albedo_files.append(file_path)
        
        for albedo_path in tqdm(albedo_files, desc='Preparing scene by city...'):
            fileParts = albedo_path.split('/')
            date, city = fileParts[-2], fileParts[-3]
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]
            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path
            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')
            date_object = datetime.strptime(date, "%Y-%m")
            date_object = date_object + relativedelta(months=self.monthsAhead)
            monthsAhead = date_object.strftime("%Y-%m")
            lst_path = lst_path.replace(date, monthsAhead)
            if not os.path.exists(lst_path):
                continue
            raster_dict['LST.tif'] = lst_path 
            raster_dict['HeatIndex.tif'] = lst_path.replace('LST.tif', 'HeatIndex.tif')
            if city not in cities:
                cities[city] = {}
            cities[city][date] = raster_dict                    
    
        train_size = int(len(list(cities.keys())) * self.train_ratio)
        val_size = int((len(list(cities.keys())) - train_size) / 2)        
    
        train_cities = list(cities.keys())[:train_size]
        val_cities = list(cities.keys())[train_size:train_size + val_size]
        test_cities = list(cities.keys())[train_size + val_size:]
        
        self.train_files = sortCitiesToFileList(train_cities, cities)
        self.val_files = sortCitiesToFileList(val_cities, cities)
        self.test_files = sortCitiesToFileList(test_cities, cities)
        print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

    def prepare_by_scene(self):
        file_list = []
        albedo_files = []

        x_dir = os.path.join(self.data_dir, 'preprocess', 'X', 'less5CloudCover')
        for file_path in tqdm(self.get_file_paths(x_dir), desc='Gathering scenes (Sort by Random Scene)...'):
            date = file_path.split('/')[-2]
            if self.debug and '2014' not in date: 
                continue
            if 'Albedo' in file_path:
                albedo_files.append(file_path)

        for albedo_path in tqdm(albedo_files, desc='Preparing scene by scene...'):
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]

            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path

            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')
            date = lst_path.split('/')[-2]
            date_object = datetime.strptime(date, "%Y-%m")
            date_object = date_object + relativedelta(months=self.monthsAhead)
            monthsAhead = date_object.strftime("%Y-%m")       
            lst_path = lst_path.replace(date, monthsAhead)
            if not os.path.exists(lst_path):
                continue
            raster_dict['LST.tif'] = lst_path
            raster_dict['HeatIndex.tif'] = lst_path.replace('LST.tif', 'HeatIndex.tif')
            file_list.append(raster_dict)
        train_size = int(len(file_list) * self.train_ratio)
        val_size = int((len(file_list) - train_size) / 2)

        self.train_files = file_list[:train_size]
        self.val_files = file_list[train_size:train_size + val_size]
        self.test_files = file_list[train_size + val_size:]
        # print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

    def get_file_paths(self, folder_path: str) -> list[str]:
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(full_path)
        return file_paths    

    def train_dataloader(self):
        return DataLoader(
            LandsatDataset(self.train_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            LandsatDataset(self.val_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            LandsatDataset(self.test_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
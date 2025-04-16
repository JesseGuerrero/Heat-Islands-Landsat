from torch.utils.data import ConcatDataset, Dataset, DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import os
import random
from torchvision import transforms
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from typing import Callable, List, Dict, Tuple, Optional, Any
from datetime import datetime
from dateutil.relativedelta import relativedelta

class RandomFlip:
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, sample):
        if random.random() < self.p:
            # Flip all components in the same way
            sample['input'] = torch.flip(sample['input'], dims=[-1])
            sample['target'] = torch.flip(sample['target'], dims=[-1])
            sample['mask'] = torch.flip(sample['mask'], dims=[-1])
        return sample

class RandomRotation:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles
        
    def __call__(self, sample):
        k = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 degrees
        if k > 0:
            # Rotate all components in the same way
            sample['input'] = torch.rot90(sample['input'], k, dims=[-2, -1])
            sample['target'] = torch.rot90(sample['target'], k, dims=[-2, -1])
            sample['mask'] = torch.rot90(sample['mask'], k, dims=[-2, -1])
        return sample

class TiledGeotiffDataset(Dataset):
    def __init__(self, file_list: List[Dict[str, str]], tile_size: int = 128, 
                 tile_overlap: float = 0.0, augment: bool = False, nodata_fill_value=-9999.0):
        """
        Dataset for tiled processing of geotiffs.
        
        Args:
            file_dict: Dictionary mapping band names to file paths
            tile_size: Size of the tiles
            tile_overlap: Overlap between tiles (0.0-1.0)
            transform: Transforms to apply
            nodata_fill_value: Value to use for no data
        """
        self.file_list = file_list # List[Dict()]
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.augment = augment
        self.nodata_fill_value = nodata_fill_value
        self.ranges = {
            'Albedo.tif': (-0.018, 0.998),           # Typical albedo range
            'DEM.tif': (-93.0, 3061.0),        # Approximate elevation range
            'Land_Cover.tif': (11.0, 95.0),      # Assuming land cover classes
            'NDVI.tif': (-1.0, 1.0),            # NDVI range
            'NDWI.tif': (-1.0, 1.0),            # NDWI range
            'NDBI.tif': (-1.0, 1.0),    
            'LSTInput.tif': (-80.9723, 211.73), 
            'LST.tif': (-80.9723, 211.73),             # Typical LST range in Fahrenheit
            'HeatIndex.tif': (1, 25)
        }
        self.input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif', 'NDBI.tif', 'LSTInput.tif']
        self.output_keys = ['LST.tif', 'HeatIndex.tif']


        tile_coordinates_list = [] # List[List[Tuple[int, int, int, int]]]
        src_transforms = [] # List[transforms]
        for scene in self.file_list:                        
            with rasterio.open(list(scene.values())[0]) as src:
                self.width, self.height = src.width, src.height
                src_transforms.append(src.transform)
                self.crs = src.crs
                
                # Get the cell sizes in x and y directions
                # Affine transform matrix elements: [a, b, c, d, e, f]
                # a: width of a pixel
                # e: height of a pixel (negative)
                self.cell_width = abs(src.transform.a)
                self.cell_height = abs(src.transform.e)
            
                # Generate tile coordinates using original cell sizes
                tile_coordinates_list.append(self._get_tiles(img_size=(self.width, self.height), 
                                                tile_size=tile_size, 
                                                tile_overlap=tile_overlap))
        self.all_tiles = []
        self.all_files = []
        self.all_transforms = []
        for sceneI, sceneTiles in enumerate(tile_coordinates_list):
            for tileBox in sceneTiles:
                self.all_tiles.append(tileBox)
                self.all_files.append(self.file_list[sceneI])
                self.all_transforms.append(src_transforms[sceneI])
        
    def normalize(self, sample):
        x = sample['input']  #6 512 512
        y = sample['target']  #2 512 512
        
        # Normalize each input channel
        for i, channel_name in enumerate(self.input_keys):
            min_val, max_val = self.ranges[channel_name]           
            # Only normalize values within the range
            mask = x[i:i+1, :, :] != -9999
            x[i:i+1, :, :][mask] = (x[i:i+1, :, :][mask] - min_val) / (max_val - min_val)
        
        # Normalize target LST
        min_val, max_val = self.ranges['LST.tif']
        mask = y[0:1, :, :] != -9999
        y[0:1, :, :][mask] = (y[0:1, :, :][mask] - min_val) / (max_val - min_val)
        
        # Normalize target Heat Index
        min_val, max_val = self.ranges['HeatIndex.tif']  
        mask = y[0:1, :, :] != -9999      
        y[1:2, :, :][mask] = (y[1:2, :, :][mask] - min_val) / (max_val - min_val)
        
        return {'input': x, 'target': y, 'mask': sample['mask']}

    @staticmethod
    def denormalize(sample):
        if isinstance(sample, dict):
            x = sample['input'].clone()  # Clone to avoid modifying original
            y = sample['target'].clone()  # Clone to avoid modifying original
        else:
            y = sample.clone()
        ranges = {
            'Albedo.tif': (-0.018, 0.998),           # Typical albedo range
            'DEM.tif': (-93.0, 3061.0),        # Approximate elevation range
            'Land_Cover.tif': (11.0, 95.0),      # Assuming land cover classes
            'NDVI.tif': (-1.0, 1.0),            # NDVI range
            'NDWI.tif': (-1.0, 1.0),            # NDWI range
            'NDBI.tif': (-1.0, 1.0),   
            'LSTInput.tif': (-80.9723, 211.73), 
            'LST.tif': (-80.9723, 211.73),             # Typical LST range in Fahrenheit
            'HeatIndex.tif': (1, 25)
        }
        input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif', 'NDBI.tif', 'LSTInput.tif']
        
        if isinstance(sample, dict):
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
        mask = y[:, 0:1, :, :] != -9999
        y[:, 1:2, :, :][mask] = y[:, 1:2, :, :][mask] * (max_val - min_val) + min_val

        # Apply integer conversion and limits to Heat Index
        y[:, 1:2, :, :][mask] = torch.round(y[:, 1:2, :, :][mask])  # Convert to integers
        y[:, 1:2, :, :][mask] = torch.clamp(y[:, 1:2, :, :][mask], min=1, max=25)  # Apply limits
    
        if isinstance(sample, dict):
            return {'input': x, 'target': y, 'mask': sample['mask'], 'box': sample['box'], 'transform': sample['transform'], 'file_dict': sample['file_dict']}
        return y

    def _get_tiles(self, img_size: Tuple[int, int], tile_size: int, 
                  tile_overlap: float) -> List[Tuple[int, int, int, int]]:
        """Generates tile coordinates with specified overlap, respecting original cell size."""
        tiles = []
        img_w, img_h = img_size
        
        # Calculate strides in pixel units based on original cell sizes
        # If we want consistent geographic coverage, we adjust the stride in pixels
        # to match the real-world distance
        stride_w = int((1 - tile_overlap) * tile_size)
        stride_h = int((1 - tile_overlap) * tile_size)
        
        for y in range(0, img_h - tile_size + 1, stride_h):
            for x in range(0, img_w - tile_size + 1, stride_w):
                x2 = x + tile_size
                y2 = y + tile_size
                
                # Don't include partial tiles at the edges
                if x2 <= img_w and y2 <= img_h:
                    tiles.append((x, y, x2, y2))
        
        return tiles
    
    def __len__(self):
        return len(self.all_tiles)
    
    def __getitem__(self, idx):
        box = self.all_tiles[idx]
        scene = self.all_files[idx]
        transform = self.all_transforms[idx]
                
        # Read input bands
        channels = []
        channel_masks = []
        for key in self.input_keys:
            xmin, ymin, xmax, ymax = box
            window = Window(col_off=xmin, row_off=ymin, width=xmax-xmin, height=ymax-ymin)
            tile_transform = rasterio.windows.transform(window, transform)
            with rasterio.open(scene[key]) as src:
                channel = src.read(1, window=window).astype(np.float32)
                valid_mask = ~np.isnan(channel)
                valid_mask = valid_mask & (channel != self.nodata_fill_value)
                channel = np.where(valid_mask, channel, self.nodata_fill_value)
                channels.append(channel)
                channel_masks.append(valid_mask)
        x = np.stack(channels, axis=0)
        input_mask = np.stack(channel_masks, axis=0)
        
        # Read target LST
        with rasterio.open(scene['LST.tif']) as src:
            lst = src.read(1, window=window).astype(np.float32)
            lst_mask = ~np.isnan(lst)
            lst_mask = lst_mask & (lst != self.nodata_fill_value)
            lst = np.where(lst_mask, lst, self.nodata_fill_value)
        
        # Read target Heat Index and add as a second channel
        with rasterio.open(scene['HeatIndex.tif']) as src:
            heat_index = src.read(1, window=window).astype(np.float32)
            heat_index_mask = ~np.isnan(heat_index)
            heat_index_mask = heat_index_mask & (heat_index != self.nodata_fill_value)
            heat_index = np.where(heat_index_mask, heat_index, self.nodata_fill_value)

        # Combine all masks
        target_mask = lst_mask & heat_index_mask
        combined_mask = np.all(input_mask, axis=0) & target_mask
        
        # Apply combined mask to input and target data
        for i in range(x.shape[0]):
            x[i] = np.where(combined_mask, x[i], self.nodata_fill_value)
        
        lst = np.where(combined_mask, lst, self.nodata_fill_value)
        heat_index = np.where(combined_mask, heat_index, self.nodata_fill_value)
        
        # Stack LST and Heat Index into a 2-channel target tensor
        y = np.stack([lst, heat_index], axis=0)
        
        # Expand mask dimension
        combined_mask = np.expand_dims(combined_mask, axis=0)

        sample = {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'mask': torch.from_numpy(combined_mask)
        }

        sample = self.normalize(sample)
        if self.augment:
            transform = transforms.Compose([
                RandomFlip(p=0.5),
                RandomRotation()
            ])
            sample = transform(sample)
        
        sample = {
            'input': sample['input'],
            'target': sample['target'],
            'mask': sample['mask'],
            'box': [xmin, ymin, xmax, ymax],  # Include tile coordinates for reference
            'transform': tile_transform,  # Include the geographic transform
            'file_dict': scene
        }
        return sample

class TiledLandsatDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            monthsAhead: int = 0,
            batch_size: int = 1,
            num_workers: int = 2,
            train_ratio: float = 0.8,
            augment: bool = False,
            byCity: bool = False,
            shuffleTrain: bool = True,
            debug: bool = False,
            nodata_fill_value: float = -9999.0,
            normalize: bool = True,
            tile_size: int = 128,
            tile_overlap: float = 0.0,
            seedForScene: int = 1,
            onlyTrain: bool = False,
            includeYears: list = []
    ):
        super().__init__()
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.data_dir = data_dir
        self.monthsAhead = monthsAhead
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.augment = augment
        self.nodata_fill_value = nodata_fill_value
        self.byCity = byCity
        self.suffleTrain = shuffleTrain
        self.debug = debug
        self.normalize = normalize
        self.seedForScene = seedForScene
        self.onlyTrain = onlyTrain
        if self.debug:
            includeYears = ["2014"]
        self.includeYears = includeYears
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
        x_dir = os.path.join(self.data_dir, f'preprocess_{self.monthsAhead}monthsahead', 'X', 'less5CloudCover')
        for file_path in tqdm(self.get_file_paths(x_dir), desc='Gathering scenes(Sort by City)...'):
            date = file_path.split('/')[-2]
            for year in self.includeYears:
                if year in date:
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
            date = albedo_path.split('/')[-2]
            date_object = datetime.strptime(date, "%Y-%m")
            date_object = date_object + relativedelta(months=self.monthsAhead)
            dateAhead = date_object.strftime("%Y-%m")
            lst_path = albedo_path.replace('/X/', '/y/').replace(date, dateAhead).replace('Albedo.tif', 'LST.tif')
            if not os.path.exists(lst_path):
                continue
            raster_dict['LST.tif'] = lst_path 
            raster_dict['HeatIndex.tif'] = lst_path.replace('LST.tif', 'HeatIndex.tif')
            if city not in cities:
                cities[city] = {}
            cities[city][date] = raster_dict                    
    
        if self.onlyTrain:
            self.train_cities = sortCitiesToFileList(list(cities.keys())[:len(list(cities.keys()))-2])
            self.val_cities = [sortCitiesToFileList(list(cities.keys())[-1])]
            self.test_cities = [sortCitiesToFileList(list(cities.keys())[-2])]
            print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")
            return
            
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

        x_dir = os.path.join(self.data_dir, f'preprocess_{self.monthsAhead}monthsahead', 'X', 'less5CloudCover')
        for file_path in tqdm(self.get_file_paths(x_dir), desc='Gathering scenes (Sort by Random Scene)...'):
            date = file_path.split('/')[-2]
            for year in self.includeYears:
                if year in date:
                    if 'Albedo' in file_path:
                        albedo_files.append(file_path)

        for albedo_path in tqdm(albedo_files, desc='Preparing scene by scene...'):
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]

            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path

            date = albedo_path.split('/')[-2]
            date_object = datetime.strptime(date, "%Y-%m")
            date_object = date_object + relativedelta(months=self.monthsAhead)
            dateAhead = date_object.strftime("%Y-%m")
            lst_path = albedo_path.replace('/X/', '/y/').replace(date, dateAhead).replace('Albedo.tif', 'LST.tif')
            if not os.path.exists(lst_path):
                continue
            raster_dict['LST.tif'] = lst_path
            raster_dict['HeatIndex.tif'] = lst_path.replace('LST.tif', 'HeatIndex.tif')
            file_list.append(raster_dict)
        if self.onlyTrain:
            self.train_files = file_list[:len(file_list)-2]
            self.val_files = [file_list[-1]]
            self.test_files = [file_list[-2]]
            print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")
            return

        train_size = int(len(file_list) * self.train_ratio)
        val_size = int((len(file_list) - train_size) / 2)

        self.train_files = file_list[:train_size]
        self.val_files = file_list[train_size:train_size + val_size]
        self.test_files = file_list[train_size + val_size:]
        print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

    def get_file_paths(self, folder_path: str) -> list[str]:
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(full_path)
        return file_paths   
    
    def train_dataloader(self):
        train_dataset = TiledGeotiffDataset(
                self.train_files, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                augment=self.augment, 
                nodata_fill_value=self.nodata_fill_value
            ) 
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=self.suffleTrain,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        val_dataset = TiledGeotiffDataset(
                self.val_files, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                nodata_fill_value=self.nodata_fill_value
            )
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        test_dataset = TiledGeotiffDataset(
                self.test_files, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                nodata_fill_value=self.nodata_fill_value
            )
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
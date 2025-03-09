from torch.utils.data import ConcatDataset, Dataset, DataLoader
import pytorch_lightning as pl
import torch
import numpy as np
import os
from tqdm import tqdm
import rasterio
from rasterio.windows import Window
from typing import Callable, List, Dict, Tuple, Optional, Any

# Import your existing LandsatDataModule
from utils.data.LandsatDataModule import LandsatDataModule

class TiledGeotiffDataset(Dataset):
    def __init__(self, file_dict: Dict[str, str], tile_size: int = 128, 
                 tile_overlap: float = 0.0, transform=None, nodata_fill_value=-9999.0):
        """
        Dataset for tiled processing of geotiffs.
        
        Args:
            file_dict: Dictionary mapping band names to file paths
            tile_size: Size of the tiles
            tile_overlap: Overlap between tiles (0.0-1.0)
            transform: Transforms to apply
            nodata_fill_value: Value to use for no data
        """
        self.file_dict = file_dict
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value
        
        # Get image dimensions and cell size from first band
        with rasterio.open(list(file_dict.values())[0]) as src:
            self.width, self.height = src.width, src.height
            self.src_transform = src.transform
            self.crs = src.crs
            
            # Get the cell sizes in x and y directions
            # Affine transform matrix elements: [a, b, c, d, e, f]
            # a: width of a pixel
            # e: height of a pixel (negative)
            self.cell_width = abs(self.src_transform.a)
            self.cell_height = abs(self.src_transform.e)
        
        # Generate tile coordinates using original cell sizes
        self.tiles = self._get_tiles(img_size=(self.width, self.height), 
                                    tile_size=tile_size, 
                                    tile_overlap=tile_overlap)
        
        # Input keys for consistent ordering
        self.input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif']

            
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
        return len(self.tiles)
    
    def __getitem__(self, idx):
        box = self.tiles[idx]
        # print(box)
        xmin, ymin, xmax, ymax = box
        window = Window(col_off=xmin, row_off=ymin, width=xmax-xmin, height=ymax-ymin)
        
        # Get the actual geographic coordinates for this tile
        # This preserves the actual cell size information
        tile_transform = rasterio.windows.transform(window, self.src_transform)
        
        # Read input bands
        channels = []
        channel_masks = []
        
        for key in self.input_keys:
            with rasterio.open(self.file_dict[key]) as src:
                channel = src.read(1, window=window).astype(np.float32)
                valid_mask = ~np.isnan(channel)
                valid_mask = valid_mask & (channel != self.nodata_fill_value)
                channel = np.where(valid_mask, channel, 0.0)
                channels.append(channel)
                channel_masks.append(valid_mask)
                
        x = np.stack(channels, axis=0)
        input_mask = np.stack(channel_masks, axis=0)
        
        # Read target LST
        with rasterio.open(self.file_dict['LST.tif']) as src:
            lst = src.read(1, window=window).astype(np.float32)
            lst_mask = ~np.isnan(lst)
            lst_mask = lst_mask & (lst != self.nodata_fill_value)
            lst = np.where(lst_mask, lst, 0.0)
        
        # Read target Heat Index and add as a second channel
        with rasterio.open(self.file_dict['HeatIndex.tif']) as src:
            heat_index = src.read(1, window=window).astype(np.float32)
            heat_index_mask = ~np.isnan(heat_index)
            heat_index_mask = heat_index_mask & (heat_index != self.nodata_fill_value)
            heat_index = np.where(heat_index_mask, heat_index, 0.0)
        
        # Combine all masks
        target_mask = lst_mask & heat_index_mask
        combined_mask = np.all(input_mask, axis=0) & target_mask
        
        # Apply combined mask to input and target data
        for i in range(x.shape[0]):
            x[i] = np.where(combined_mask, x[i], 0.0)
        
        lst = np.where(combined_mask, lst, 0.0)
        heat_index = np.where(combined_mask, heat_index, 0.0)
        
        # Stack LST and Heat Index into a 2-channel target tensor
        y = np.stack([lst, heat_index], axis=0)
        
        # Expand mask dimension
        combined_mask = np.expand_dims(combined_mask, axis=0)
        
        sample = {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'mask': torch.from_numpy(combined_mask),
            'box': [xmin, ymin, xmax, ymax],  # Include tile coordinates for reference
            'transform': tile_transform,  # Include the geographic transform
            'file_dict': self.file_dict
        }
        
        return sample

class TiledLandsatDataModule(LandsatDataModule):
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
            tile_size: int = 128,
            tile_overlap: float = 0.0
    ):
        """
        Data module for tiled Landsat data.
        
        Args:
            data_dir: Directory containing the data
            batch_size: Batch size
            num_workers: Number of workers for data loading
            train_ratio: Ratio of data for training
            transform: Transforms to apply
            byCity: Whether to split data by city
            debug: Whether to use debug mode
            nodata_fill_value: Value to use for no data
            normalize: Whether to normalize the data
            tile_size: Size of the tiles
            tile_overlap: Overlap between tiles (0.0-1.0)
        """
        super().__init__(
            data_dir=data_dir,
            monthsAhead=monthsAhead,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            transform=transform,
            byCity=byCity,
            debug=debug,
            nodata_fill_value=nodata_fill_value,
            normalize=normalize
        )
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
    
    def train_dataloader(self):
        train_datasets = [
            TiledGeotiffDataset(
                file_dict, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                transform=self.transform, 
                nodata_fill_value=self.nodata_fill_value
            ) 
            for file_dict in self.train_files
        ]
        
        train_dataset = ConcatDataset(train_datasets)
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        val_datasets = [
            TiledGeotiffDataset(
                file_dict, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                transform=self.transform, 
                nodata_fill_value=self.nodata_fill_value
            )
            for file_dict in self.val_files
        ]
        
        val_dataset = ConcatDataset(val_datasets)
        
        return DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        test_datasets = [
            TiledGeotiffDataset(
                file_dict, 
                tile_size=self.tile_size, 
                tile_overlap=self.tile_overlap,
                transform=self.transform, 
                nodata_fill_value=self.nodata_fill_value
            )
            for file_dict in self.test_files
        ]
        
        test_dataset = ConcatDataset(test_datasets)
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
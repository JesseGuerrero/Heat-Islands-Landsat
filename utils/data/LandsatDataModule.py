from tqdm import tqdm

class LandsatDataset(Dataset):
    def __init__(self, file_list, transform=None, nodata_fill_value=-9999.0):
        self.file_list = file_list
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_files = self.file_list[idx]

        channels = []
        channel_masks = []
        input_keys = ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif']
        for key in input_keys:
            with rasterio.open(sample_files[key]) as src:
                channel = src.read(1).astype(np.float32)
                valid_mask = ~np.isnan(channel)
                valid_mask = valid_mask & (channel != self.nodata_fill_value)
                channel = np.where(valid_mask, channel, 0.0)
                channels.append(channel)
                channel_masks.append(valid_mask)
        
        x = np.stack(channels, axis=0)
        input_mask = np.stack(channel_masks, axis=0)

        with rasterio.open(sample_files['LST.tif']) as src:
            y = src.read(1).astype(np.float32)
            target_mask = ~np.isnan(y)
            target_mask = target_mask & (y != self.nodata_fill_value)
            y = np.where(target_mask, y, 0.0)

        combined_mask = np.all(input_mask, axis=0) & target_mask
        
        for i in range(x.shape[0]):
            x[i] = np.where(combined_mask, x[i], 0.0)
        y = np.where(combined_mask, y, 0.0)

        y = np.expand_dims(y, axis=0)
        combined_mask = np.expand_dims(combined_mask, axis=0)

        sample = {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'mask': torch.from_numpy(combined_mask)
        }

        if self.transform:
            sample = self.transform(sample)

        if sample['input'].shape[1:] != sample['target'].shape[1:]:
            raise ValueError("Mismatch between input and target spatial dimensions.")
        return sample

class LandsatDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 1,
            num_workers: int = 2,
            train_ratio: float = 0.8,
            transform=None,
            byCity: bool = False,
            debug: bool = False,
            nodata_fill_value: float = -9999.0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value
        self.byCity = byCity
        self.debug = debug
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def setup(self, stage=None):
        pass


    def preprocessImages(self):
        albedo_files = []
        x_dir = os.path.join(self.data_dir, 'X', 'less5CloudCover')
        def clip_and_save_raster(src_path, dst_path):
            if os.path.exists(dst_path):
                return
            with rasterio.open(src_path) as src:
                data = src.read(1)
                profile = src.profile.copy()
                
                height, width = data.shape
                new_height = (height // 32) * 32
                new_width = (width // 32) * 32
                
                start_y = (height - new_height) // 2
                start_x = (width - new_width) // 2
                
                clipped_data = data[start_y:start_y + new_height, 
                                start_x:start_x + new_width]
                
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': rasterio.windows.transform(
                        rasterio.windows.Window(start_x, start_y, new_width, new_height),
                        src.transform
                    )
                })

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(clipped_data, 1)

        for file_path in tqdm(self.get_file_paths(x_dir), desc='Gathering scenes (Preproccessing)...'):
            date = file_path.split('/')[-2]
            if self.debug and '2014' not in date: 
                continue
            if 'Albedo' in file_path:
                albedo_files.append(file_path)
        
        for albedo_path in tqdm(albedo_filesdesc='Preprocessing images...'):
            date = file_path.split('/')[-2]
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                        if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]
            
            for raster_file in scene_files:
                src_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                dst_path = src_path.replace('Data/', 'Data/preprocess/')                
                clip_and_save_raster(src_path, dst_path)
            
            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')            
            dst_path = lst_path.replace('Data/', 'Data/preprocess/')
            clip_and_save_raster(lst_path, dst_path)

    def prepare_by_city(self):
        self.preprocessImages()
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
            date, city = fileParts[-1], fileParts[-2], fileParts[-3], fileParts[-4], fileParts[-5]
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]
    
            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path
    
            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')
            raster_dict['LST.tif'] = lst_path 
            if city not in cities:
                cities[city] = {}
            cities[city][date] = raster_dict                    
    
        # print(list(cities.keys()))
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
            raster_dict['LST.tif'] = lst_path
            file_list.append(raster_dict)

        train_size = int(len(file_list) * self.train_ratio)
        val_size = int((len(file_list) - train_size) / 2)

        self.train_files = file_list[:train_size]
        self.val_files = file_list[train_size:train_size + val_size]
        self.test_files = file_list[train_size + val_size:]
        # print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

    def get_file_paths(self, folder_path: str) -> List[str]:
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(full_path)
        return file_paths

    def prepare_data(self):
        if self.byCity:
            self.prepare_by_city()
        else:
            self.prepare_by_scene()

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

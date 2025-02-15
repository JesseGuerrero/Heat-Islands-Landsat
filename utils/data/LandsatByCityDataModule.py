
class LandsatDataset(Dataset):
    def __init__(self, file_list, transform=None, nodata_fill_value=-9999.0):
        self.file_list = file_list
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        sample_files = self.file_list[idx]

        # Process input channels
        channels = []
        for key in ['Albedo.tif', 'DEM.tif', 'Land_Cover.tif', 'NDVI.tif', 'NDWI.tif']:
            with rasterio.open(sample_files[key]) as src:
                channel = src.read(1).astype(np.float32)
            channel = np.where(np.isnan(channel), 0.0, channel)
            channels.append(channel)

        # Dynamic resizing
        ref_shape = channels[0].shape
        fixed_channels = []
        for ch in channels:
            if ch.shape != ref_shape:
                ch_resized = cv2.resize(ch, (ref_shape[1], ref_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                fixed_channels.append(ch_resized)
            else:
                fixed_channels.append(ch)
        channels = fixed_channels

        # Find the center crop dimensions divisible by 32
        h, w = ref_shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32

        # Calculate the starting positions for center crop
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2

        # Crop all channels
        cropped_channels = []
        for ch in channels:
            cropped_ch = ch[start_h:start_h + new_h, start_w:start_w + new_w]
            cropped_channels.append(cropped_ch)

        x = np.stack(cropped_channels, axis=0)

        # Process target (LST)
        with rasterio.open(sample_files['LST.tif']) as src:
            y = src.read(1).astype(np.float32)

        valid_mask = ~np.isnan(y)
        valid_mask = valid_mask & (y != self.nodata_fill_value)
        y = np.where(valid_mask, y, 0.0)

        if y.shape != ref_shape:
            y = cv2.resize(y, (ref_shape[1], ref_shape[0]),
                           interpolation=cv2.INTER_LINEAR)
            valid_mask = cv2.resize(valid_mask.astype(np.uint8),
                                    (ref_shape[1], ref_shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            valid_mask = valid_mask.astype(bool)

        y = y[start_h:start_h + new_h, start_w:start_w + new_w]
        valid_mask = valid_mask[start_h:start_h + new_h, start_w:start_w + new_w]

        y = np.expand_dims(y, axis=0)
        valid_mask = np.expand_dims(valid_mask, axis=0)

        sample = {
            'input': torch.from_numpy(x),
            'target': torch.from_numpy(y),
            'mask': torch.from_numpy(valid_mask)
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

    def prepare_by_city(self):
        def sortCitiesToFileList(cities_for_task, all_cities):
            file_list = []
            for city in cities_for_task:
                for scene in all_cities[city].values():
                    file_list.append(scene)
            return file_list
        cities = {}
        albedo_files = []
        x_dir = os.path.join(self.data_dir, 'X', 'less5CloudCover')
        for file_path in self.get_file_paths(x_dir):
            date = file_path.split('/')[-2]
            if self.debug and '2014' not in date: 
                continue
            if 'Albedo' in file_path:
                albedo_files.append(file_path)
        
        for albedo_path in albedo_files:
            fileParts = albedo_path.split('/')
            fileName, date, city, cloudCategory, dataType = fileParts[-1], fileParts[-2], fileParts[-3], fileParts[-4], fileParts[-5]
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]
    
            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path
    
            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')
            if os.path.exists(lst_path):  # Only add if LST file exists
                raster_dict['LST.tif'] = lst_path 
                if city not in cities:
                    cities[city] = {}
                cities[city][date] = raster_dict                    
    
        # Split datasets
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
        def get_file_paths(self, folder_path: str) -> List[str]:
            file_paths = []
            for root, _, files in os.walk(folder_path):
                for file in files:
                    full_path = os.path.abspath(os.path.join(root, file))
                    file_paths.append(full_path)
            return file_paths
        file_list = []
        albedo_files = []

        x_dir = os.path.join(self.data_dir, 'X', 'less5CloudCover')
        for file_path in get_file_paths(x_dir):
            date = file_path.split('/')[-2]
            if self.debug and '2014' not in date: 
                continue
            if 'Albedo' in file_path:
                albedo_files.append(file_path)

        for albedo_path in albedo_files:
            scene_files = [f for f in os.listdir(os.path.dirname(albedo_path))
                           if os.path.isfile(os.path.join(os.path.dirname(albedo_path), f))]

            raster_dict = {}
            for raster_file in scene_files:
                raster_path = os.path.join(os.path.dirname(albedo_path), raster_file)
                raster_dict[raster_file] = raster_path

            lst_path = albedo_path.replace('/X/', '/y/').replace('Albedo.tif', 'LST.tif')
            if os.path.exists(lst_path):  # Only add if LST file exists
                raster_dict['LST.tif'] = lst_path
                file_list.append(raster_dict)

        # Split datasets
        train_size = int(len(file_list) * self.train_ratio)
        val_size = int((len(file_list) - train_size) / 2)

        self.train_files = file_list[:train_size]
        self.val_files = file_list[train_size:train_size + val_size]
        self.test_files = file_list[train_size + val_size:]
        # print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

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

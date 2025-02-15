class RasterDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str,
            batch_size: int = 1,
            num_workers: int = 2,
            train_ratio: float = 0.8,
            transform=None,
            nodata_fill_value: float = -9999.0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.transform = transform
        self.nodata_fill_value = nodata_fill_value
        self.train_files = []
        self.val_files = []
        self.test_files = []

    def setup(self, stage=None):
        if not self.train_files:  # Only prepare data if not already prepared
            self.prepare_data()

    def prepare_data(self):
        cities = {}
        albedo_files = []
        x_dir = os.path.join(self.data_dir, 'X', 'less5CloudCover')
        for file_path in self.get_file_paths(x_dir):
            date = file_path.split('/')[-2]
            # Use this to only do 1 year, otherwise you do all years, you can comment out
            if '2014' not in date: 
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
        
        self.train_files = self.sortCitiesToFileList(train_cities, cities)
        self.val_files = self.sortCitiesToFileList(val_cities, cities)
        self.test_files = self.sortCitiesToFileList(test_cities, cities)
        print(f"Dataset splits - Train: {len(self.train_files)}, Val: {len(self.val_files)}, Test: {len(self.test_files)}")

    def sortCitiesToFileList(self, cities_for_task, all_cities):
        file_list = []
        for city in cities_for_task:
            for scene in all_cities[city].values():
                file_list.append(scene)
        return file_list

    def get_file_paths(self, folder_path: str) -> List[str]:
        file_paths = []
        for root, _, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.abspath(os.path.join(root, file))
                file_paths.append(full_path)
        return file_paths

    def train_dataloader(self):
        return DataLoader(
            RasterDataset(self.train_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            RasterDataset(self.val_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            RasterDataset(self.test_files, self.transform, self.nodata_fill_value),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

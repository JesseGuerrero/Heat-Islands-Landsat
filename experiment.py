# %%
from torchgeo.trainers import PixelwiseRegressionTask
import torch
import pytorch_lightning as pl
import numpy as np
import rasterio
import cv2
import logging
from typing import List
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
import torch.nn as nn
import os
from datetime import datetime
from utils.model import LSTNowcaster
from utils.data.TiledLandsatDataModule import TiledLandsatDataModule
from utils.voice import notifySelf
torch.cuda.empty_cache()
torch.set_float32_matmul_precision('high')

os.environ["WANDB_NOTEBOOK_NAME"] = "TrainUNet-Basic.ipynb"
os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb/.cache/wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb/.config/wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb/.cache/wandb-data"
os.environ["WANDB_ARTIFACT_DIR"] = "./wandb/artifacts"
import sys

i = -1
# Get the first argument passed after the script name
if len(sys.argv) > 1:
    i = int(sys.argv[1])  # Convert string to integer
    batchSize = int(sys.argv[2])
    deviceCount = int(sys.argv[3])
config = {
    "experiment_name": "7 channels 3 months debug segformer",
    "debug": False,
    "by_city": False,
    "months_ahead": 1,
    "tile_size": 128,
    "tile_overlap": 0.0,
    "learning_rate": 1e-4,
    "model": "segformer",
    "backbone": "b5",
    "dataset": "pure_landsat",
    "augment": True,
    "epochs": 160,
    "batch_size": batchSize,
    "pretrained_weights": True,
    "deterministic": True,
    "random_seed_by_scene": 1,
    "in_channels": 7,
    "only_train": False,
    "skip_years": []
}

if i == 1:        
    config["model"] = "segformer"
    config["backbone"] = "b5"
    config["months_ahead"] = 1
if i == 2:        
    config["model"] = "segformer"
    config["backbone"] = "b5"
    config["months_ahead"] = 3
if i == 3:        
    config["model"] = "segformer"
    config["backbone"] = "b3"
    config["months_ahead"] = 1
if i == 4:        
    config["model"] = "segformer"
    config["backbone"] = "b3"
    config["months_ahead"] = 3
if i == 5:        
    config["model"] = "deeplabv3+"
    config["backbone"] = "resnet50"
    config["months_ahead"] = 1
if i == 6:        
    config["model"] = "deeplabv3+"
    config["backbone"] = "resnet50"
    config["months_ahead"] = 3
#b3
if i == 7:        
    config["model"] = "deeplabv3+"
    config["backbone"] = "resnet18"
    config["months_ahead"] = 1
if i == 8:        
    config["model"] = "deeplabv3+"
    config["backbone"] = "resnet18"
    config["months_ahead"] = 3
if i == 9:        
    config["model"] = "unet"
    config["backbone"] = "resnet50"
    config["months_ahead"] = 1
if i == 10:        
    config["model"] = "unet"
    config["backbone"] = "resnet50"
    config["months_ahead"] = 3
if i == 11:        
    config["model"] = "unet"
    config["backbone"] = "resnet18"
    config["months_ahead"] = 1
if i == 12:        
    config["model"] = "unet"
    config["backbone"] = "resnet18"
    config["months_ahead"] = 3
if i <= -1:
    pass
else:
    config["experiment_name"] = f'Exp. #{i}: {config["model"]},Month {config["months_ahead"]}, {config["backbone"]}'

notifySelf(f'Starting {config["experiment_name"]}!')
wandb_logger = WandbLogger(
    project="heat-island",
    name=config['experiment_name'],
    log_model="best",
    save_code=True,
    save_dir="./wandb",
)
wandb_logger.log_hyperparams(config)    

# Create model
model = LSTNowcaster(
    model=config["model"], 
    backbone=config["backbone"], 
    in_channels=config["in_channels"], 
    learning_rate=config["learning_rate"], 
    pretrained_weights=config["pretrained_weights"]
)

class PercentageProgressCallback(Callback):
    def __init__(self, total_epochs, experiment_name):
        super().__init__()
        self.total_epochs = total_epochs
        self.experiment_name = experiment_name

    def on_train_epoch_end(self, trainer, pl_module):
        # Only run on main process
        if trainer.is_global_zero:
            current_epoch = trainer.current_epoch
            if current_epoch % 20 == 0:
                current_percentage = min(100, int(current_epoch / self.total_epochs * 100))
                wandb.alert(title="Training Update", 
                        text=f'{self.experiment_name} is at {current_percentage:.2f}%', 
                        level=wandb.AlertLevel.INFO)

percentage_callback = PercentageProgressCallback(total_epochs=config["epochs"], experiment_name=config["experiment_name"])    
wandb_run_id = wandb_logger.experiment.id    
current_date = datetime.now()                
date_string = current_date.strftime("%B%d")
checkpoint_callback = ModelCheckpoint(
    dirpath=f"./wandb/heat-island/checkpoints/{wandb_run_id}_{date_string}",
    filename= f"{wandb_run_id}_{date_string}_" + "{epoch:03d}_{val_rmse_F:.4f}",
    monitor="val_rmse_p",
    mode="min",
    save_top_k=1,
    every_n_epochs=1,
    save_last=False  # Also save the last model for comparison
)
allYears = ["2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023"]
for year in config["skip_years"]:
    allYears.remove(year)
# for subYears in [allYears[:5], allYears[5:]]:
trainer = pl.Trainer(
    max_epochs=config['epochs'],
    gradient_clip_val=0.5,
    log_every_n_steps=10,
    enable_progress_bar=True,
    enable_model_summary=False,
    # deterministic=config["deterministic"],
    num_sanity_val_steps=2,
    logger=wandb_logger,
    callbacks=[checkpoint_callback, percentage_callback],
    # devices=deviceCount,                         # Use all 4 GPUs
    accelerator="gpu",                 # Use GPU acceleration
    # strategy="ddp",                    # Use DistributedDataParallel
    precision="16-mixed"               # Add mixed precision for memory efficiency
)                             

data_module = TiledLandsatDataModule(
    data_dir="./Data",
    monthsAhead=config["months_ahead"],
    batch_size=config["batch_size"],
    num_workers=8,
    byCity=config["by_city"],
    debug=config["debug"],
    tile_size=config["tile_size"],
    tile_overlap=config["tile_overlap"],
    augment=config["augment"],
    seedForScene=config["random_seed_by_scene"],
    onlyTrain = config["only_train"],
    includeYears=allYears
)
data_module.setup()

# Train model
trainer.fit(model=model, datamodule=data_module)
del trainer
del data_module
# Force garbage collection and clear CUDA cache
import gc
gc.collect()
torch.cuda.empty_cache()
# After deleting objects
for i in range(torch.cuda.device_count()):
    with torch.cuda.device(i):
        torch.cuda.empty_cache()

# Register the best model as a W&B artifact
best_model_path = checkpoint_callback.best_model_path
if best_model_path and os.path.exists(best_model_path):
    artifact = wandb.Artifact(
        name=f"{best_model_path.split('/')[-1].replace('=','.')}", 
        type="model",
        description=f"Best model at {best_model_path.split('/')[-1]}" 
    )
    artifact.add_file(best_model_path)
    wandb_logger.experiment.log_artifact(artifact)

# End Experiment
wandb.finish()
notifySelf(f"Finished {config['experiment_name']}...")
del model
del wandb_logger
del checkpoint_callback

# Force garbage collection and clear CUDA cache
import gc
for obj in gc.get_objects():   
    try:
        if torch.is_tensor(obj) and obj.device.type == 'cuda':
            del obj
    except:
        pass
gc.collect()

# After deleting objects
for j in range(torch.cuda.device_count()):
    with torch.cuda.device(j):
        x = torch.zeros(1024, 1024, 1024, device=f'cuda:{j}')
        del x
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

# 4. Wait for GPU processes to complete
torch.cuda.synchronize()

# Print memory stats for debugging
if torch.cuda.is_available():
    print(f"Loop {i} completed. CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
notifySelf("Batch experiment ended.")

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
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
import os
from utils.model import LSTNowcaster
from utils.data.TiledLandsatDataModule import TiledLandsatDataModule
# from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.profilers import PyTorchProfiler
torch.cuda.empty_cache()

os.environ["WANDB_NOTEBOOK_NAME"] = "TrainUNet-Basic.ipynb"
os.environ["WANDB_DIR"] = "./wandb"
os.environ["WANDB_CACHE_DIR"] = "./wandb/.cache/wandb"
os.environ["WANDB_CONFIG_DIR"] = "./wandb/.config/wandb"
os.environ["WANDB_DATA_DIR"] = "./wandb/.cache/wandb-data"
os.environ["WANDB_ARTIFACT_DIR"] = "./wandb/artifacts"

config = {
    "experiment_name": "Test normalization fixes",
    "debug": True,
    "by_city": False,
    "months_ahead": 0,
    "tile_size": 512,
    "learning_rate": 1e-4,
    "model": "unet",
    "backbone": "resnet50",
    "dataset": "pure_landsat",
    "epochs": 1,
    "batch_size": 32,
    "pretrained_weights": True,
    "deterministic": True,
    "in_channels": 5
}

for i in range(3):
    if i == 0:
        config["experiment_name"] = "Test 1 Batch 32"
        config["batch_size"] = 32
    if i == 1:
        config["experiment_name"] = "Test 1 Batch 4"
        config["batch_size"] = 4
    if i == 2:
        config["experiment_name"] = "Test 1 Batch 1"
        config["batch_size"] = 1
        
    wandb_logger = WandbLogger(
        project="heat-island",
        name=config['experiment_name'],
        log_model="all",
        save_code=True,
        save_dir="./wandb",
    )
    wandb_logger.log_hyperparams(config)
    
    if config["dataset"] == "pure_landsat":
        data_module = TiledLandsatDataModule(
            data_dir="./Data",
            monthsAhead=config["months_ahead"],
            batch_size=config["batch_size"],
            num_workers=5,
            byCity=config["by_city"],
            debug=config["debug"],
            tile_size=config["tile_size"],
            tile_overlap=0.2
        )
        data_module.setup()

    checkpoint_callback = ModelCheckpoint(dirpath="./wandb/heat-island/checkpoints", monitor="val_rmse_F", mode="min")

    # Initialize trainer with explicit steps
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        gradient_clip_val=0.5,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=config["deterministic"],
        num_sanity_val_steps=2,
        reload_dataloaders_every_n_epochs=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )

    # Create model
    model = LSTNowcaster(
        model=config["model"], 
        backbone=config["backbone"], 
        in_channels=config["in_channels"], 
        learning_rate=config["learning_rate"], 
        pretrained_weights=config["pretrained_weights"]
    )

    # Train model
    trainer.fit(model=model, datamodule=data_module)
    
    # Clean up resources
    del model
    del trainer
    del data_module
    del wandb_logger
    del checkpoint_callback
    
    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # Print memory stats for debugging
    if torch.cuda.is_available():
        print(f"Loop {i} completed. CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")




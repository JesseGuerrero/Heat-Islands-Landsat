
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

class LSTNowcaster(pl.LightningModule):
    def __init__(self, model="unet", backbone="resnet50", in_channels=5, learning_rate=1e-4, pretrained_weights=True):
        super().__init__()
        self.save_hyperparameters()
        self.model = PixelwiseRegressionTask(
            model=model,
            backbone=backbone,
            weights=pretrained_weights,
            in_channels=in_channels,
            num_outputs=1,
            loss="mse",
            lr=learning_rate
        )

        #Replace for two channels:
        old_head = self.model.model.segmentation_head[0]
        new_head = nn.Conv2d(
            old_head.in_channels,
            2,  # Set to 2 output channels
            kernel_size=old_head.kernel_size,
            stride=old_head.stride,
            padding=old_head.padding
        )                
        self.model.model.segmentation_head[0] = new_head

        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.train_rmse = []
        self.test_rmse = []
        self.validate_rmse = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.train_rmse)
        return {"loss": loss}
    
    def on_train_epoch_start(self):
        self.train_rmse = []

    def on_train_epoch_end(self):
        avg_rmse = torch.stack(self.train_rmse).mean()
        self.log("train_rmse_F", avg_rmse, 
             on_step=False,
             on_epoch=True,
             prog_bar=True,
             sync_dist=True)
    
    def save_rmse(self, batch, outputs, rmse_list):        
        targets = batch['target']
        mask = batch['mask']
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_f = torch.mean((outputs[expanded_mask] - targets[expanded_mask])**2)
        rmse_f = torch.sqrt(mse_f)
                
        rmse_list.append(rmse_f)

    def validation_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.validate_rmse)
        return mse_loss
    
    def on_validation_epoch_start(self):
        self.validate_rmse = []

    def on_validation_epoch_end(self):
        avg_rmse = torch.stack(self.validate_rmse).mean()
        self.log("val_rmse_F", avg_rmse, prog_bar=True)

    def test_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.test_rmse)
        return mse_loss
    
    def on_test_epoch_start(self):
        self.test_rmse = []

    def on_test_epoch_end(self):
        avg_rmse = torch.stack(self.test_rmse).mean()
        self.log("test_rmse_F", avg_rmse, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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
from utils.data.TiledLandsatDataModule import TiledGeotiffDataset
from transformers import SegformerForSemanticSegmentation
from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation, OneFormerModel

class LSTNowcaster(pl.LightningModule):
    def __init__(self, model="unet", backbone="resnet50", in_channels=7, learning_rate=1e-4, pretrained_weights=True):
        super().__init__()
        self.save_hyperparameters()

        # Load the pre-trained segformer model
        if model == "segformer":
            if backbone == "b5":
                self.model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-b5-finetuned-ade-640-640")
            else:
                self.model = SegformerForSemanticSegmentation.from_pretrained(f"nvidia/segformer-{backbone}-finetuned-ade-512-512")

            # Modify input projection to match your input channels
            orig_proj = self.model.segformer.encoder.patch_embeddings[0].proj
            new_proj = nn.Conv2d(
                in_channels, 
                orig_proj.out_channels,
                kernel_size=orig_proj.kernel_size,
                stride=orig_proj.stride,
                padding=orig_proj.padding
            )
            self.model.segformer.encoder.patch_embeddings[0].proj = new_proj

            # Modify the final classifier to output 2 channels instead of semantic classes
            old_classifier = self.model.decode_head.classifier
            self.model.decode_head.classifier = nn.Conv2d(
                old_classifier.in_channels,
                2,  # 2 output channels for LST and Heat Index
                kernel_size=old_classifier.kernel_size,
                stride=old_classifier.stride,
                padding=old_classifier.padding
            )
            
            # Create a wrapper that ensures output is at the original input resolution
            class SegformerWrapper(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model
                    
                def forward(self, x):
                    # Get the output from the model
                    output = self.model(x).logits
                    
                    # Check if the spatial dimensions match the input
                    if output.shape[2:] != x.shape[2:]:
                        # Resize to match the input resolution
                        output = nn.functional.interpolate(
                            output, 
                            size=x.shape[2:],  # Match input spatial dimensions
                            mode='bilinear', 
                            align_corners=False
                        )
                    
                    return output
            
            # Wrap the model
            self.model = SegformerWrapper(self.model)
        if model == "unet" or model == "deeplabv3+":
            self.model = PixelwiseRegressionTask(
                model=model,
                backbone=backbone,
                weights=pretrained_weights,
                in_channels=in_channels,
                num_outputs=1,
                loss="mse",
                lr=learning_rate
            )

            # Replace for two channels:
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
        self.train_rmse_lst, self.train_rmse_heat_index = [], []
        self.test_rmse_lst, self.test_rmse_heat_index = [], []
        self.validate_rmse_lst, self.validate_rmse_heat_index = [], []

    def forward(self, x):
        with torch.cuda.amp.autocast(enabled=False):
            return self.model(x.float())

    def training_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        outputs = self(inputs) # Outputs is NaN, check inputs
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.train_rmse_lst, self.train_rmse_heat_index)
        return {"loss": loss}

    def on_train_epoch_start(self):
        self.train_rmse_lst, self.train_rmse_heat_index = [], []

    def on_train_epoch_end(self):
        avg_rmse = torch.stack(self.train_rmse_lst).mean()
        self.log("train_rmse_F", avg_rmse, 
             on_step=False,
             on_epoch=True,
             prog_bar=True,
             sync_dist=True)
        avg_rmse = torch.stack(self.train_rmse_heat_index).mean()
        self.log("train_rmse_P", avg_rmse, 
             on_step=False,
             on_epoch=True,
             prog_bar=True,
             sync_dist=True)
    
    def save_rmse(self, batch, outputs, rmse_list_lst, rmse_list_heatindex):  
        # print('output', torch.mean(outputs[:, 0:1, :, :]))
        # print('output', torch.mean(outputs[:, 1:2, :, :]))
        # mask = batch['target'][:, 0:1, :, :] != -9999
        # print('target', torch.mean(batch['target'][:, 0:1, :, :][mask]))
        # print('target', torch.mean(batch['target'][:, 1:2, :, :][mask]))
        # print("Output shape -> ", outputs.shape)
        outputs = TiledGeotiffDataset.denormalize(outputs)      
        targets = TiledGeotiffDataset.denormalize(batch['target'])
        # print('output', torch.mean(outputs[:, 0:1, :, :]))
        # print('output', torch.mean(outputs[:, 1:2, :, :]))
        # mask = batch['target'][:, 0:1, :, :] != -9999
        # print('target', torch.mean(targets[:, 0:1, :, :][mask]))
        # print('target', torch.mean(targets[:, 1:2, :, :][mask]))
        mask = batch['mask']
        lst = targets[:, 0:1, :, :]        
        heatIndex = targets[:, 1:2, :, :]
        
        # Expand mask to match output channels
        mask = mask.expand_as(lst)
        
        mse_f = torch.mean((outputs[:, 0:1, :, :][mask] - lst[mask])**2)
        rmse_f = torch.sqrt(mse_f)
        rmse_list_lst.append(rmse_f)
        mse_f = torch.mean((outputs[:, 1:2, :, :][mask] - heatIndex[mask])**2)
        rmse_f = torch.sqrt(mse_f)
        rmse_list_heatindex.append(rmse_f)

    def validation_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.validate_rmse_lst, self.validate_rmse_heat_index)
        return mse_loss
    
    def on_validation_epoch_start(self):
        self.validate_rmse_lst, self.validate_rmse_heat_index = [], []

    def on_validation_epoch_end(self):
        avg_rmse = torch.stack(self.validate_rmse_lst).mean()
        self.log("val_rmse_F", avg_rmse, prog_bar=True, sync_dist=True)
        avg_rmse = torch.stack(self.validate_rmse_heat_index).mean()
        self.log("val_rmse_p", avg_rmse, prog_bar=True, sync_dist=True)

    def test_step(self, batch):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.test_rmse_lst, self.test_rmse_heat_index)
        return mse_loss
    
    def on_test_epoch_start(self):
        self.test_rmse_lst, self.test_rmse_heat_index = [], []

    def on_test_epoch_end(self):
        avg_rmse = torch.stack(self.test_rmse_lst).mean()
        self.log("test_rmse_F", avg_rmse, prog_bar=True, sync_dist=True)
        avg_rmse = torch.stack(self.test_rmse_heat_index).mean()
        self.log("test_rmse_P", avg_rmse, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Add cosine annealing scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,  # Total number of epochs
            eta_min=1e-6  # Minimum learning rate
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update scheduler after each epoch
                "frequency": 1,
                "monitor": "val_rmse_p",  # Optional: monitor validation metric
                "strict": True,
                "name": "cosine_lr"
            }
        }
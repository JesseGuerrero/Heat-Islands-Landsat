import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Sam2Model, Sam2Config
from transformers.models.sam2.modeling_sam2 import Sam2MaskDecoder, Sam2Neck
import torch.nn.functional as F


class SAM2Regression(pl.LightningModule):
    def __init__(self, in_channels=6, out_channels=2, learning_rate=1e-4, fine_tune=True, image_size=512):
        super().__init__()
        self.save_hyperparameters()
        
        # Load SAM 2 configuration and model
        self.config = Sam2Config.from_pretrained("facebook/sam2")
        
        # Modify config for regression task
        self.config.image_size = image_size
        
        # Load pre-trained SAM 2 model (encoder only)
        self.sam = Sam2Model.from_pretrained("facebook/sam2", config=self.config)
        
        # Freeze encoder if not fine-tuning
        if not fine_tune:
            for param in self.sam.vision_encoder.parameters():
                param.requires_grad = False
        
        # Modify input projection for custom number of channels
        # Get original projection layer
        orig_proj = self.sam.vision_encoder.patch_embed.proj
        
        # Create new projection with custom in_channels but same properties otherwise
        self.sam.vision_encoder.patch_embed.proj = nn.Conv2d(
            in_channels,
            orig_proj.out_channels,
            kernel_size=orig_proj.kernel_size,
            stride=orig_proj.stride,
            padding=orig_proj.padding
        )

        # Custom regression neck (similar to SAM2 neck but adapted for regression)
        self.regression_neck = Sam2Neck(
            in_channels=self.config.vision_encoder_config.hidden_size,
            hidden_size=256,
            out_channels=256,
            num_layers=2
        )
        
        # Custom regression decoder
        self.regression_decoder = RegressionDecoder(
            transformer_dim=256,
            out_channels=out_channels
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        
        # Metrics tracking
        self.train_rmse_lst, self.train_rmse_heat_index = [], []
        self.val_rmse_lst, self.val_rmse_heat_index = [], []
        self.test_rmse_lst, self.test_rmse_heat_index = [], []

    def forward(self, x):
        # Extract image features using SAM2's vision encoder
        vision_outputs = self.sam.vision_encoder(x)
        
        # Process features through regression neck
        neck_outputs = self.regression_neck(vision_outputs.last_hidden_state)
        
        # Get regression output from decoder
        regression_output = self.regression_decoder(neck_outputs)
        
        # Ensure output is at the same resolution as input
        if regression_output.shape[2:] != x.shape[2:]:
            regression_output = F.interpolate(
                regression_output,
                size=x.shape[2:],
                mode='bilinear',
                align_corners=False
            )
        
        return regression_output

    def training_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.train_rmse_lst, self.train_rmse_heat_index)
        
        return {"loss": loss}

    def on_train_epoch_start(self):
        self.train_rmse_lst, self.train_rmse_heat_index = [], []

    def on_train_epoch_end(self):
        if len(self.train_rmse_lst) > 0:
            avg_rmse = torch.stack(self.train_rmse_lst).mean()
            self.log("train_rmse_F", avg_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        if len(self.train_rmse_heat_index) > 0:
            avg_rmse = torch.stack(self.train_rmse_heat_index).mean()
            self.log("train_rmse_P", avg_rmse, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
    
    def save_rmse(self, batch, outputs, rmse_list_lst, rmse_list_heatindex):
        # Import here to avoid circular imports
        from utils.data.TiledLandsatDataModule import TiledGeotiffDataset
        
        outputs = TiledGeotiffDataset.denormalize(outputs)      
        targets = TiledGeotiffDataset.denormalize(batch['target'])
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

    def validation_step(self, batch, batch_idx):
        inputs = batch['input']
        targets = batch['target']
        mask = batch['mask']
        
        outputs = self(inputs)
        
        # Expand mask to match output channels
        expanded_mask = mask.expand_as(targets)
        
        mse_loss = self.criterion(outputs[expanded_mask], targets[expanded_mask])
        self.save_rmse(batch, outputs, self.val_rmse_lst, self.val_rmse_heat_index)
        
        return mse_loss
    
    def on_validation_epoch_start(self):
        self.val_rmse_lst, self.val_rmse_heat_index = [], []

    def on_validation_epoch_end(self):
        if len(self.val_rmse_lst) > 0:
            avg_rmse = torch.stack(self.val_rmse_lst).mean()
            self.log("val_rmse_F", avg_rmse, prog_bar=True, sync_dist=True)
        if len(self.val_rmse_heat_index) > 0:
            avg_rmse = torch.stack(self.val_rmse_heat_index).mean()
            self.log("val_rmse_P", avg_rmse, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
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
        if len(self.test_rmse_lst) > 0:
            avg_rmse = torch.stack(self.test_rmse_lst).mean()
            self.log("test_rmse_F", avg_rmse, prog_bar=True, sync_dist=True)
        if len(self.test_rmse_heat_index) > 0:
            avg_rmse = torch.stack(self.test_rmse_heat_index).mean()
            self.log("test_rmse_P", avg_rmse, prog_bar=True, sync_dist=True)
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


class RegressionDecoder(nn.Module):
    """
    Custom decoder for regression tasks, adapted from SAM2's mask decoder
    but simplified and modified to output continuous values instead of masks
    """
    def __init__(self, transformer_dim=256, out_channels=2):
        super().__init__()
        
        # Progressive upsampling with residual blocks
        self.upscale1 = ResUpscaleBlock(transformer_dim, transformer_dim // 2)
        self.upscale2 = ResUpscaleBlock(transformer_dim // 2, transformer_dim // 4)
        self.upscale3 = ResUpscaleBlock(transformer_dim // 4, transformer_dim // 8)
        
        # Final prediction layer
        self.predictor = nn.Conv2d(transformer_dim // 8, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        # Progressive upsampling
        x = self.upscale1(features)
        x = self.upscale2(x)
        x = self.upscale3(x)
        
        # Final prediction
        output = self.predictor(x)
        
        return output


class ResUpscaleBlock(nn.Module):
    """
    Residual block with upsampling for the decoder
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv to match dimensions
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # Upsample input
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        
        # Skip connection
        identity = self.skip(x_up)
        
        # Main path
        out = self.conv1(x_up)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection
        out += identity
        out = self.relu(out)
        
        return out
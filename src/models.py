"""
ML models for archaeological feature detection using hybrid U-Net + Vision Transformer
architecture for satellite imagery analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List, Dict, Optional
import numpy as np

class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels: int, key_channels: int = 128):
        super().__init__()
        self.key_channels = key_channels
        self.query_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, key_channels, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # Create query, key, value projections
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        # Calculate attention
        attention = F.softmax(torch.bmm(query, key), dim=-1)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class ViTBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.vit = timm.create_model('vit_base_patch16_224', pretrained=True,
                                   num_heads=num_heads, qkv_bias=True)
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )

class UNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class HybridUNetViT(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        
        # UNet encoder
        self.enc1 = UNetBlock(in_channels, 64)
        self.enc2 = UNetBlock(64, 128)
        self.enc3 = UNetBlock(128, 256)
        
        # ViT integration
        self.vit = ViTBlock(256)
        
        # UNet decoder
        self.dec3 = UNetBlock(512, 256)
        self.dec2 = UNetBlock(256, 128)
        self.dec1 = UNetBlock(128, 64)
        
        # Final classification
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Pool and upsample
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        
        # ViT processing
        vit_out = self.vit(e3)
        
        # Reshape ViT output and concatenate with encoder features
        vit_out = vit_out.reshape(e3.shape[0], -1, e3.shape[2], e3.shape[3])
        
        # Decoder
        d3 = self.dec3(torch.cat([e3, vit_out], dim=1))
        d2 = self.dec2(self.up(d3))
        d1 = self.dec1(self.up(d2))
        
        return self.final(d1)

def get_model(config: dict) -> Tuple[nn.Module, dict]:
    """Create model instance based on configuration"""
    model = HybridUNetViT(
        in_channels=config['in_channels'],
        num_classes=config['num_classes']
    )
    
    training_params = {
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'batch_size': 8
    }
    
    return model, training_params

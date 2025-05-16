"""
Model training utilities and dataset handling
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from tqdm import tqdm

from .models import get_model
from .satellite_processing import SatelliteDataProcessor

@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    device: str
    scheduler: Dict
    validation_split: float = 0.2
    early_stopping_patience: int = 10

class SatelliteDataset(Dataset):
    def __init__(self, data_root: Path, split: str = 'train', transform=None):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        
        # Load split data
        split_file = self.data_root / f'{split}_data.json'
        with open(split_file, 'r') as f:
            self.data_files = json.load(f)
        
        # Initialize satellite processor
        self.processor = SatelliteDataProcessor()
    
    def __len__(self) -> int:
        return len(self.data_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_info = self.data_files[idx]
        
        # Load input data
        input_path = self.data_root / 'processed' / data_info['input_file']
        input_data = np.load(input_path)
        
        # Load target mask
        target_path = self.data_root / 'processed' / data_info['target_file']
        target = np.load(target_path)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_data).float()
        target_tensor = torch.from_numpy(target).long()
        
        if self.transform:
            input_tensor = self.transform(input_tensor)
        
        return input_tensor, target_tensor

class Trainer:
    def __init__(self, config_path: str, data_root: Path):
        self.logger = logging.getLogger('xingu_pipeline.training')
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.training_config = TrainingConfig(**config['model']['training'])
        
        # Set device
        self.device = torch.device(self.training_config.device if torch.cuda.is_available() else "cpu")
        
        # Initialize model
        self.model, _ = get_model(config['model'])
        self.model = self.model.to(self.device)
        
        # Initialize datasets
        self.train_dataset = SatelliteDataset(data_root, split='train')
        self.val_dataset = SatelliteDataset(data_root, split='val')
        
        # Initialize data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=True,
            num_workers=4
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.training_config.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_config.scheduler['T_max'],
            eta_min=self.training_config.scheduler['eta_min']
        )
        
        # Initialize criterion
        self.criterion = nn.CrossEntropyLoss()
        
        # Initialize metrics tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_iou': [],
            'val_iou': []
        }
    
    def train(self) -> Dict[str, List[float]]:
        """Execute training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.training_config.epochs):
            # Training phase
            self.model.train()
            train_loss, train_iou = self._train_epoch()
            
            # Validation phase
            self.model.eval()
            val_loss, val_iou = self._validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch+1}/{self.training_config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}"
            )
            
            # Update metrics history
            self.metrics_history['train_loss'].append(train_loss)
            self.metrics_history['val_loss'].append(val_loss)
            self.metrics_history['train_iou'].append(train_iou)
            self.metrics_history['val_iou'].append(val_iou)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.training_config.early_stopping_patience:
                    self.logger.info("Early stopping triggered!")
                    break
        
        return self.metrics_history
    
    def _train_epoch(self) -> Tuple[float, float]:
        """Execute one training epoch"""
        total_loss = 0
        intersection_sum = 0
        union_sum = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (data, target) in enumerate(progress_bar):
            # Move to device
            data, target = data.to(self.device), target.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            intersection = ((pred == 1) & (target == 1)).float().sum()
            union = ((pred == 1) | (target == 1)).float().sum()
            intersection_sum += intersection
            union_sum += union
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        iou = intersection_sum / (union_sum + 1e-8)
        
        return avg_loss, iou.item()
    
    def _validate(self) -> Tuple[float, float]:
        """Execute validation phase"""
        total_loss = 0
        intersection_sum = 0
        union_sum = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                # Move to device
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Update metrics
                total_loss += loss.item()
                pred = torch.argmax(output, dim=1)
                intersection = ((pred == 1) & (target == 1)).float().sum()
                union = ((pred == 1) | (target == 1)).float().sum()
                intersection_sum += intersection
                union_sum += union
        
        # Calculate validation metrics
        avg_loss = total_loss / len(self.val_loader)
        iou = intersection_sum / (union_sum + 1e-8)
        
        return avg_loss, iou.item()
    
    def _save_checkpoint(self, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'metrics_history': self.metrics_history
        }
        
        checkpoint_path = Path(self.model_dir) / f'model_checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

import os
import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm

from src.config import Config
from src.utils.reproducibility import set_deterministic_mode
from src.utils.augmentation import MixUpCutMixAugmentation
from src.utils.schedulers import get_scheduler
from src.utils.early_stopping import EarlyStopping

class TrainingManager:
    """Comprehensive training management system."""
    
    def __init__(self, config: Config, output_dir: str):
        """Initialize training manager.
        
        Args:
            config: Configuration object
            output_dir: Directory for saving outputs
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Setup device
        self.device = self._setup_device()
        
        # Initialize training components
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        self.augmentation = MixUpCutMixAugmentation(config.training)
        
        # Training state
        self.current_epoch = 0
        self.best_val_score = 0.0
        self.training_history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'learning_rate': []
        }
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=config.training.early_stopping_patience,
            min_delta=config.training.early_stopping_delta,
            mode='max'
        )
    
    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.config.experiment.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.experiment.device)
        
        self.logger.info(f"Using device: {device}")
        
        if device.type == "cuda":
            self.logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
            self.logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def train(self, model: nn.Module, train_loader: DataLoader, 
              val_loader: DataLoader) -> str:
        """Execute complete training procedure.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Path to best model checkpoint
        """
        self.logger.info("Starting training procedure")
        
        # Move model to device
        model = model.to(self.device)
        
        # Setup optimizer and scheduler
        optimizer = self._setup_optimizer(model)
        scheduler = self._setup_scheduler(optimizer, len(train_loader))
        criterion = self._setup_criterion()
        
        # Save initial model configuration
        self._save_model_config(model)
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch(
                model, train_loader, optimizer, criterion, scheduler
            )
            
            # Validation phase
            if epoch % self.config.training.validation_frequency == 0:
                val_metrics = self._validate_epoch(model, val_loader, criterion)
            else:
                val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
            
            # Update training history
            self._update_training_history(train_metrics, val_metrics, optimizer)
            
            # Log progress
            self._log_epoch_progress(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_score
            if is_best:
                self.best_val_score = val_metrics['accuracy']
                self._save_checkpoint(model, optimizer, scheduler, epoch, is_best=True)
            
            # Regular checkpoint saving
            if epoch % self.config.training.save_frequency == 0:
                self._save_checkpoint(model, optimizer, scheduler, epoch, is_best=False)
            
            # Early stopping check
            self.early_stopping(val_metrics['accuracy'])
            if self.early_stopping.should_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f} seconds")
        
        # Save final results
        self._save_training_results(total_time)
        
        # Return path to best model
        return str(self.output_dir / "best_model.pth")
    
    def _setup_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Setup optimizer based on configuration."""
        if self.config.training.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.training.learning_rate,
                betas=self.config.training.betas,
                eps=self.config.training.eps
            )
        elif self.config.training.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.training.learning_rate,
                momentum=0.9,
                weight_decay=self.config.training.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.training.optimizer}")
        
        self.logger.info(f"Using optimizer: {self.config.training.optimizer}")
        return optimizer
    
    def _setup_scheduler(self, optimizer: optim.Optimizer, steps_per_epoch: int) -> Any:
        """Setup learning rate scheduler."""
        return get_scheduler(
            self.config.training,
            optimizer,
            steps_per_epoch
        )
    
    def _setup_criterion(self) -> nn.Module:
        """Setup loss criterion."""
        if self.config.training.criterion.lower() == 'cross_entropy':
            criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config.training.label_smoothing
            )
        else:
            criterion = nn.CrossEntropyLoss()
        
        return criterion
    
    def _train_epoch(self, model: nn.Module, train_loader: DataLoader,
                    optimizer: optim.Optimizer, criterion: nn.Module,
                    scheduler: Any) -> Dict[str, float]:
        """Execute one training epoch."""
        model.train()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Apply augmentation
            if self.config.training.use_mixup or self.config.training.use_cutmix:
                inputs, targets_a, targets_b, lam = self.augmentation(inputs, targets)
                
                # Mixed precision training
                if self.scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    
                    self.scaler.scale(loss).backward()
                    
                    # Gradient clipping
                    if self.config.training.gradient_clipping > 0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clipping)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(inputs)
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
                    loss.backward()
                    
                    # Gradient clipping
                    if self.config.training.gradient_clipping > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clipping)
                    
                    optimizer.step()
                
                # Approximate accuracy for mixed samples
                _, predicted = outputs.max(1)
                correct_predictions += (lam * predicted.eq(targets_a).sum().item() + 
                                      (1 - lam) * predicted.eq(targets_b).sum().item())
            
            else:
                # Standard training
                if self.scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                    
                    self.scaler.scale(loss).backward()
                    
                    if self.config.training.gradient_clipping > 0:
                        self.scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clipping)
                    
                    self.scaler.step(optimizer)
                    self.scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    
                    if self.config.training.gradient_clipping > 0:
                        nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clipping)
                    
                    optimizer.step()
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
            
            # Update scheduler
            if scheduler is not None and hasattr(scheduler, 'step') and not hasattr(scheduler, 'step_batch'):
                # Only step if it's not a per-batch scheduler
                pass
            
            # Update metrics
            total_loss += loss.item()
            total_samples += targets.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.0 * correct_predictions / total_samples:.2f}%'
            })
        
        # Update scheduler (epoch-wise schedulers)
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                try:
                    scheduler.step()
                except:
                    # Some schedulers might need different parameters
                    pass
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct_predictions / total_samples
        }
    
    def _validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                       criterion: nn.Module) -> Dict[str, float]:
        """Execute one validation epoch."""
        model.eval()
        
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation", leave=False):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                if self.scaler is not None:
                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
                total_samples += targets.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': correct_predictions / total_samples
        }
    
    def _update_training_history(self, train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float], optimizer: optim.Optimizer):
        """Update training history with current epoch metrics."""
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        self.training_history['val_loss'].append(val_metrics['loss'])
        self.training_history['val_acc'].append(val_metrics['accuracy'])
        self.training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
    
    def _log_epoch_progress(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float]):
        """Log epoch progress."""
        self.logger.info(
            f"Epoch {epoch+1:3d}/{self.config.training.num_epochs:3d} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                        scheduler: Any, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'best_val_score': self.best_val_score,
            'training_history': self.training_history,
            'config': self.config.__dict__
        }
        
        if is_best:
            checkpoint_path = self.output_dir / "best_model.pth"
            self.logger.info(f"Saving best model checkpoint (val_acc: {self.best_val_score:.4f})")
        else:
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        torch.save(checkpoint, checkpoint_path)
    
    def _save_model_config(self, model: nn.Module):
        """Save model configuration and architecture info."""
        config_dict = {
            'model_config': self.config.model.__dict__,
            'total_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        config_path = self.output_dir / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def _save_training_results(self, total_time: float):
        """Save final training results and history."""
        results = {
            'training_completed': True,
            'total_training_time': total_time,
            'best_validation_accuracy': self.best_val_score,
            'final_epoch': self.current_epoch,
            'training_history': self.training_history,
            'config': {
                'experiment': self.config.experiment.__dict__,
                'training': self.config.training.__dict__,
                'model': self.config.model.__dict__
            }
        }
        
        results_path = self.output_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Training results saved to {results_path}")
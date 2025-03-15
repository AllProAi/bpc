"""
Training module for 6DoF pose estimation models.

This module contains training functionality for both standard and one-shot
approaches to 6DoF pose estimation, with support for different loss functions,
optimizers, and training strategies.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging
from tqdm import tqdm
import wandb

from ..models.pose_estimation import create_model


class PoseEstimationTrainer:
    """
    Trainer class for 6DoF pose estimation models.
    
    This class handles the training, validation, and early stopping
    of pose estimation models, with support for logging, checkpointing,
    and visualization.
    
    Attributes:
        model: The pose estimation model to train
        device: The device to use for training (CPU or GPU)
        optimizer: The optimizer for model parameter updates
        scheduler: Optional learning rate scheduler
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        loss_fn: Loss function for pose estimation
        num_epochs: Maximum number of epochs to train
        checkpoint_dir: Directory to save model checkpoints
        use_wandb: Whether to use Weights & Biases for logging
        early_stopping: Whether to use early stopping
        patience: Number of epochs to wait for improvement before stopping
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        num_epochs: int = 100,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "models/checkpoints",
        use_wandb: bool = True,
        early_stopping: bool = True,
        patience: int = 10
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Pose estimation model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for model parameter updates
            loss_fn: Loss function for pose estimation
            device: Device to use for training
            num_epochs: Maximum number of epochs to train
            scheduler: Optional learning rate scheduler
            checkpoint_dir: Directory to save model checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.early_stopping = early_stopping
        self.patience = patience
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(checkpoint_dir, "training.log")),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize training metrics
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {}
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        position_loss = 0.0
        rotation_loss = 0.0
        
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            rgb_images = batch['rgb_images'].to(self.device)
            depth_maps = batch.get('depth_maps')
            if depth_maps is not None:
                depth_maps = depth_maps.to(self.device)
            
            # Get ground truth poses
            gt_positions = batch['object_poses'][:, :, :3, 3].to(self.device)  # Extract translation components
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(rgb_images, depth_maps)
            
            # Calculate loss
            loss, loss_dict = self.loss_fn(outputs, {
                'position': gt_positions,
                'rotation': batch['object_poses'][:, :, :3, :3].to(self.device),  # Extract rotation components
                'object_ids': batch.get('object_ids')
            })
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            position_loss += loss_dict.get('position_loss', 0.0)
            rotation_loss += loss_dict.get('rotation_loss', 0.0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_loss': f"{loss_dict.get('position_loss', 0.0):.4f}",
                'rot_loss': f"{loss_dict.get('rotation_loss', 0.0):.4f}"
            })
        
        # Calculate average epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_position_loss = position_loss / num_batches
        avg_rotation_loss = rotation_loss / num_batches
        
        # Update scheduler if exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'position_loss': avg_position_loss,
            'rotation_loss': avg_rotation_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """
        Validate the model on the validation dataset.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_loss = 0.0
        position_loss = 0.0
        rotation_loss = 0.0
        
        num_batches = len(self.val_loader)
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                # Move data to device
                rgb_images = batch['rgb_images'].to(self.device)
                depth_maps = batch.get('depth_maps')
                if depth_maps is not None:
                    depth_maps = depth_maps.to(self.device)
                
                # Get ground truth poses
                gt_positions = batch['object_poses'][:, :, :3, 3].to(self.device)  # Extract translation components
                
                # Forward pass
                outputs = self.model(rgb_images, depth_maps)
                
                # Calculate loss
                loss, loss_dict = self.loss_fn(outputs, {
                    'position': gt_positions,
                    'rotation': batch['object_poses'][:, :, :3, :3].to(self.device),  # Extract rotation components
                    'object_ids': batch.get('object_ids')
                })
                
                # Update metrics
                val_loss += loss.item()
                position_loss += loss_dict.get('position_loss', 0.0)
                rotation_loss += loss_dict.get('rotation_loss', 0.0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'val_loss': f"{loss.item():.4f}"
                })
        
        # Calculate average validation metrics
        avg_loss = val_loss / num_batches
        avg_position_loss = position_loss / num_batches
        avg_rotation_loss = rotation_loss / num_batches
        
        return {
            'loss': avg_loss,
            'position_loss': avg_position_loss,
            'rotation_loss': avg_rotation_loss
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint at {checkpoint_path}")
        
        # Save best model if specified
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model at {best_path}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        Train the model for the specified number of epochs.
        
        Returns:
            Training history with metrics
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        
        start_time = time.time()
        
        for epoch in range(1, self.num_epochs + 1):
            epoch_start_time = time.time()
            
            # Train for one epoch
            self.logger.info(f"Epoch {epoch}/{self.num_epochs}")
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_metrics['loss'])
            
            # Log metrics
            epoch_time = time.time() - epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
            self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
            
            if self.use_wandb:
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_position_loss': train_metrics['position_loss'],
                    'train_rotation_loss': train_metrics['rotation_loss'],
                    'val_loss': val_metrics['loss'],
                    'val_position_loss': val_metrics['position_loss'],
                    'val_rotation_loss': val_metrics['rotation_loss'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                wandb.log(log_dict)
            
            # Check if this is the best model
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping
            if self.early_stopping and self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        
        return self.training_history


class OneShot6DPoseTrainer(PoseEstimationTrainer):
    """
    Specialized trainer for one-shot 6D pose estimation.
    
    This trainer implements techniques specifically for one-shot learning,
    where the model needs to generalize to unseen objects without specific
    training data.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: Callable,
        device: torch.device,
        num_epochs: int = 100,
        scheduler: Optional[Any] = None,
        checkpoint_dir: str = "models/one_shot_checkpoints",
        use_wandb: bool = True,
        early_stopping: bool = True,
        patience: int = 10,
        feature_adaptation: bool = True,
        domain_adaptation: bool = True
    ):
        """
        Initialize the one-shot trainer.
        
        Args:
            model: Pose estimation model
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            optimizer: Optimizer for model parameter updates
            loss_fn: Loss function for pose estimation
            device: Device to use for training
            num_epochs: Maximum number of epochs to train
            scheduler: Optional learning rate scheduler
            checkpoint_dir: Directory to save model checkpoints
            use_wandb: Whether to use Weights & Biases for logging
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            feature_adaptation: Whether to use feature adaptation techniques
            domain_adaptation: Whether to use domain adaptation techniques
        """
        super().__init__(
            model, train_loader, val_loader, optimizer, loss_fn, device,
            num_epochs, scheduler, checkpoint_dir, use_wandb, early_stopping, patience
        )
        
        self.feature_adaptation = feature_adaptation
        self.domain_adaptation = domain_adaptation
        
        # Additional metrics for one-shot learning
        self.training_history['metrics']['feature_distance'] = []
        self.training_history['metrics']['domain_loss'] = []
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch with one-shot learning strategies.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_loss = 0.0
        position_loss = 0.0
        rotation_loss = 0.0
        feature_distance = 0.0
        domain_loss = 0.0
        
        num_batches = len(self.train_loader)
        progress_bar = tqdm(self.train_loader, desc="One-Shot Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            rgb_images = batch['rgb_images'].to(self.device)
            depth_maps = batch.get('depth_maps')
            if depth_maps is not None:
                depth_maps = depth_maps.to(self.device)
            
            # Get ground truth poses
            gt_positions = batch['object_poses'][:, :, :3, 3].to(self.device)  # Extract translation components
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(rgb_images, depth_maps)
            
            # Calculate standard loss
            loss, loss_dict = self.loss_fn(outputs, {
                'position': gt_positions,
                'rotation': batch['object_poses'][:, :, :3, :3].to(self.device),  # Extract rotation components
                'object_ids': batch.get('object_ids')
            })
            
            # Add one-shot specific losses if applicable
            if self.feature_adaptation:
                # Feature adaptation loss (example: triplet or contrastive loss)
                # This encourages features to be similar for the same object type
                # and different for different object types
                feat_dist = self._calculate_feature_distance(outputs.get('features', None), batch.get('object_ids'))
                loss += 0.1 * feat_dist  # Weight can be adjusted
                feature_distance += feat_dist.item()
            
            if self.domain_adaptation:
                # Domain adaptation loss (example: adversarial or MMD loss)
                # This encourages features to be invariant to domain shifts
                dom_loss = self._calculate_domain_loss(outputs.get('features', None), batch.get('domain_labels', None))
                loss += 0.1 * dom_loss  # Weight can be adjusted
                domain_loss += dom_loss.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            position_loss += loss_dict.get('position_loss', 0.0)
            rotation_loss += loss_dict.get('rotation_loss', 0.0)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'pos_loss': f"{loss_dict.get('position_loss', 0.0):.4f}",
                'rot_loss': f"{loss_dict.get('rotation_loss', 0.0):.4f}"
            })
        
        # Calculate average epoch metrics
        avg_loss = epoch_loss / num_batches
        avg_position_loss = position_loss / num_batches
        avg_rotation_loss = rotation_loss / num_batches
        avg_feature_distance = feature_distance / num_batches if self.feature_adaptation else 0.0
        avg_domain_loss = domain_loss / num_batches if self.domain_adaptation else 0.0
        
        # Update scheduler if exists
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'loss': avg_loss,
            'position_loss': avg_position_loss,
            'rotation_loss': avg_rotation_loss,
            'feature_distance': avg_feature_distance,
            'domain_loss': avg_domain_loss
        }
    
    def _calculate_feature_distance(self, features, object_ids):
        """
        Calculate feature distance loss for similar and different objects.
        
        This is a placeholder implementation. The actual implementation would depend
        on the specific one-shot learning approach being used.
        
        Args:
            features: Feature representations from the model
            object_ids: Object identifiers for each sample
            
        Returns:
            Feature distance loss
        """
        if features is None or object_ids is None:
            return torch.tensor(0.0, device=self.device)
        
        # Placeholder implementation
        # In a real implementation, this would calculate a triplet or contrastive loss
        # to encourage similar features for the same object and different features for
        # different objects
        return torch.tensor(0.0, device=self.device)
    
    def _calculate_domain_loss(self, features, domain_labels):
        """
        Calculate domain adaptation loss to generalize across domains.
        
        This is a placeholder implementation. The actual implementation would depend
        on the specific domain adaptation approach being used.
        
        Args:
            features: Feature representations from the model
            domain_labels: Domain labels for each sample
            
        Returns:
            Domain adaptation loss
        """
        if features is None or domain_labels is None:
            return torch.tensor(0.0, device=self.device)
        
        # Placeholder implementation
        # In a real implementation, this would calculate a domain adaptation loss
        # such as maximum mean discrepancy (MMD) or adversarial loss
        return torch.tensor(0.0, device=self.device)


# Factory function to create trainers
def create_trainer(
    trainer_type: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    config: Dict[str, Any]
) -> Union[PoseEstimationTrainer, OneShot6DPoseTrainer]:
    """
    Create a trainer based on the specified type.
    
    Args:
        trainer_type: Type of trainer ('standard' or 'one_shot')
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer for model parameters
        loss_fn: Loss function
        device: Device to train on
        config: Additional configuration parameters
        
    Returns:
        Initialized trainer
    """
    if trainer_type == 'standard':
        return PoseEstimationTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=config.get('num_epochs', 100),
            scheduler=config.get('scheduler'),
            checkpoint_dir=config.get('checkpoint_dir', 'models/checkpoints'),
            use_wandb=config.get('use_wandb', True),
            early_stopping=config.get('early_stopping', True),
            patience=config.get('patience', 10)
        )
    elif trainer_type == 'one_shot':
        return OneShot6DPoseTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=config.get('num_epochs', 100),
            scheduler=config.get('scheduler'),
            checkpoint_dir=config.get('checkpoint_dir', 'models/one_shot_checkpoints'),
            use_wandb=config.get('use_wandb', True),
            early_stopping=config.get('early_stopping', True),
            patience=config.get('patience', 10),
            feature_adaptation=config.get('feature_adaptation', True),
            domain_adaptation=config.get('domain_adaptation', True)
        )
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}") 
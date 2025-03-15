#!/usr/bin/env python3
"""
Main training script for 6DoF pose estimation models.

This script handles:
1. Configuration loading and argument parsing
2. Data loading and preprocessing
3. Model initialization and training
4. Evaluation and checkpointing
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from datetime import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.dataset import PoseEstimationDataset
from src.models.model_factory import create_model
from src.training.trainer import PoseEstimationTrainer, OneShot6DPoseTrainer
from src.training.losses import create_loss_function
from src.utils.logging_utils import setup_logger
from src.evaluation.metrics import calculate_mssd, calculate_mAP


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train 6DoF pose estimation model')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data_dir', type=str,
                        help='Path to dataset (overrides config)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory (overrides config)')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--one_shot', action='store_true',
                        help='Train in one-shot mode')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation, no training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_output_dir(output_dir: str, config: Dict[str, Any]) -> str:
    """Set up output directory for logs and checkpoints."""
    # Create timestamp for unique folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create output directory if not existing
    if output_dir is None:
        output_dir = config.get('output_dir', 'outputs')
    
    model_name = config['model']['name']
    output_path = os.path.join(output_dir, f"{model_name}_{timestamp}")
    os.makedirs(output_path, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(output_path, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'viz'), exist_ok=True)
    
    # Save config
    with open(os.path.join(output_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    return output_path


def create_datasets(config: Dict[str, Any], data_dir: Optional[str] = None) -> Tuple[PoseEstimationDataset, PoseEstimationDataset]:
    """Create training and validation datasets."""
    if data_dir is not None:
        data_path = data_dir
    else:
        data_path = config['data']['data_dir']
    
    # Create training dataset
    train_dataset = PoseEstimationDataset(
        root_dir=data_path,
        split=config['data']['train_split'],
        transform=config['data']['transform'],
        use_depth=config['data']['use_depth'],
        use_multiview=config['data']['use_multiview'],
        n_views=config['data'].get('n_views', 1)
    )
    
    # Create validation dataset
    val_dataset = PoseEstimationDataset(
        root_dir=data_path,
        split=config['data']['val_split'],
        transform=config['data'].get('val_transform', config['data']['transform']),
        use_depth=config['data']['use_depth'],
        use_multiview=config['data']['use_multiview'],
        n_views=config['data'].get('n_views', 1)
    )
    
    return train_dataset, val_dataset


def create_data_loaders(
    train_dataset: PoseEstimationDataset,
    val_dataset: PoseEstimationDataset,
    config: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    # Create training dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation dataloader
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training'].get('val_batch_size', config['training']['batch_size']),
        shuffle=False,
        num_workers=config['training'].get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


def setup_training(config: Dict[str, Any], device: torch.device, one_shot: bool = False) -> Tuple[nn.Module, optim.Optimizer, Any, Any]:
    """Set up model, optimizer, and scheduler."""
    # Create model
    model = create_model(config['model'])
    model = model.to(device)
    
    # Create optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
    elif config['training']['optimizer'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['training']['learning_rate'],
            momentum=config['training'].get('momentum', 0.9),
            weight_decay=config['training'].get('weight_decay', 0)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    # Create scheduler
    if config['training'].get('scheduler', None) == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['training'].get('scheduler_step_size', 30),
            gamma=config['training'].get('scheduler_gamma', 0.1)
        )
    elif config['training'].get('scheduler', None) == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training'].get('scheduler_eta_min', 0)
        )
    elif config['training'].get('scheduler', None) is None:
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {config['training']['scheduler']}")
    
    # Create loss function
    loss_config = config['training']['loss']
    loss_fn = create_loss_function(
        loss_type=loss_config['type'],
        config=loss_config
    )
    
    return model, optimizer, scheduler, loss_fn


def plot_training_curves(train_losses: List[float], val_losses: List[float], output_path: str):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_path, 'loss_curve.png'))
    plt.close()


def save_metrics(metrics: Dict[str, float], output_path: str, filename: str = 'metrics.yaml'):
    """Save metrics to YAML file."""
    with open(os.path.join(output_path, filename), 'w') as f:
        yaml.dump(metrics, f)


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up output directory
    output_path = setup_output_dir(args.output_dir, config)
    
    # Set up logging
    logger = setup_logger(os.path.join(output_path, 'logs', 'training.log'))
    logger.info(f"Starting training with config: {config}")
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.warning("CUDA not available, using CPU!")
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_datasets(config, args.data_dir)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, config)
    
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Set up model, optimizer, scheduler, and loss function
    model, optimizer, scheduler, loss_fn = setup_training(config, device, args.one_shot)
    
    # Create trainer
    if args.one_shot:
        trainer = OneShot6DPoseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            feature_adaptation_weight=config['training'].get('feature_adaptation_weight', 0.1),
            domain_adaptation_weight=config['training'].get('domain_adaptation_weight', 0.1)
        )
    else:
        trainer = PoseEstimationTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Resumed from checkpoint at epoch {start_epoch}")
    
    # Only run evaluation if requested
    if args.eval_only:
        logger.info("Running evaluation only")
        eval_metrics = trainer.validate(val_loader)
        save_metrics(eval_metrics, output_path, 'eval_metrics.yaml')
        logger.info(f"Evaluation complete: {eval_metrics}")
        return
    
    # Training loop
    logger.info("Starting training")
    
    # Initialize lists to store metrics
    train_losses = []
    val_losses = []
    
    # Number of epochs
    num_epochs = config['training']['epochs']
    
    for epoch in range(start_epoch, num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train for one epoch
        train_metrics = trainer.train_epoch(train_loader)
        train_losses.append(train_metrics['loss'])
        
        # Log training metrics
        logger.info(f"Training: {train_metrics}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        val_losses.append(val_metrics['loss'])
        
        # Log validation metrics
        logger.info(f"Validation: {val_metrics}")
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        # Save checkpoint
        if (epoch + 1) % config['training'].get('checkpoint_interval', 1) == 0:
            checkpoint_path = os.path.join(output_path, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }
            if scheduler is not None:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model
        if epoch == start_epoch or val_metrics['loss'] < min(val_losses[:-1]) if len(val_losses) > 1 else float('inf'):
            best_model_path = os.path.join(output_path, 'checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
        
        # Plot training curves
        plot_training_curves(train_losses, val_losses, os.path.join(output_path, 'viz'))
    
    # Save final model
    final_model_path = os.path.join(output_path, 'checkpoints', 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save final metrics
    final_metrics = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'final_train_metrics': train_metrics,
        'final_val_metrics': val_metrics
    }
    save_metrics(final_metrics, output_path, 'final_metrics.yaml')
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main() 
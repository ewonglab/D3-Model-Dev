#!/usr/bin/env python3
"""
DeepSTARR Training Script

This script provides training functionality specifically for the DeepSTARR dataset,
inheriting from the base training classes and implementing DeepSTARR-specific
model creation and data loading.
"""

import os
import sys
from pathlib import Path

# Package imports

from scripts.train import BaseD3LightningModule, BaseD3DataModule, BaseTrainer, parse_base_args
from model_zoo.deepstarr.models import create_model
from model_zoo.deepstarr.data import get_deepstarr_datasets, get_deepstarr_dataloaders
from omegaconf import OmegaConf


class DeepSTARRLightningModule(BaseD3LightningModule):
    """Lightning module specifically for DeepSTARR dataset."""
    
    def __init__(self, cfg, architecture: str = 'transformer'):
        super().__init__(cfg, dataset_name='deepstarr')
        self.architecture = architecture
        
    def create_model(self):
        """Create DeepSTARR-specific model."""
        return create_model(self.cfg, self.architecture)
        
    def process_batch(self, batch):
        """Process DeepSTARR batch data."""
        # DeepSTARR data comes as (inputs, targets) pairs
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            return inputs, targets
        else:
            raise ValueError(f"Expected (inputs, targets) pair, got {type(batch)}")


class DeepSTARRDataModule(BaseD3DataModule):
    """Data module specifically for DeepSTARR dataset."""
    
    def __init__(self, cfg):
        super().__init__(cfg, dataset_name='deepstarr')
        
    def setup(self, stage: str = None):
        """Setup DeepSTARR datasets."""
        # Use DeepSTARR-specific data loading
        self.train_ds, self.val_ds = get_deepstarr_datasets()
        print(f"DeepSTARR dataset loaded: {len(self.train_ds)} train, {len(self.val_ds)} val samples")


class DeepSTARRTrainer(BaseTrainer):
    """Trainer specifically for DeepSTARR dataset."""
    
    def __init__(self, architecture: str, config_path: str = None, work_dir: str = None):
        # Load DeepSTARR config
        if config_path:
            cfg = OmegaConf.load(config_path)
        else:
            # Use default DeepSTARR config
            config_file = Path(__file__).parent / 'configs' / f'{architecture}.yaml'
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            cfg = OmegaConf.load(config_file)
            
        super().__init__(cfg, 'deepstarr', work_dir)
        self.architecture = architecture
        
    def create_lightning_module(self):
        """Create DeepSTARR Lightning module."""
        return DeepSTARRLightningModule(self.cfg, self.architecture)
        
    def create_data_module(self):
        """Create DeepSTARR data module."""
        return DeepSTARRDataModule(self.cfg)


def main():
    """Main training function."""
    parser = parse_base_args()
    parser.description = 'DeepSTARR Training Script'
    args = parser.parse_args()
    
    # Create trainer
    trainer = DeepSTARRTrainer(
        architecture=args.architecture,
        config_path=args.config,
        work_dir=args.work_dir
    )
    
    # Override WandB settings if provided
    if args.wandb_project:
        trainer.cfg.wandb.project = args.wandb_project
    if args.wandb_name:
        trainer.cfg.wandb.name = args.wandb_name
    
    # Train
    try:
        trainer.train(resume_from=args.resume_from)
        return 0
    except Exception as e:
        print(f"Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
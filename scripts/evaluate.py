#!/usr/bin/env python3
"""
Base Evaluation Framework for D3-DNA Discrete Diffusion

This module provides the base evaluation framework that dataset-specific
evaluation scripts should inherit from. It provides common functionality
while allowing datasets to implement their own model and data loading.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json

import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm


class BaseEvaluator:
    """
    Base evaluator class that provides common evaluation functionality.
    
    Dataset-specific evaluation scripts should inherit from this class and
    implement the abstract methods for their specific needs.
    """
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer'):
        """
        Load model for the dataset. Must be implemented by subclasses.
        Each dataset implements its own simple model loading logic.
        
        Args:
            checkpoint_path: Path to specific checkpoint file
            config: Configuration object
            architecture: Architecture type ('transformer', 'convolutional')
            
        Returns:
            Tuple of (model, graph, noise) needed for sampling/evaluation
        """
        raise NotImplementedError("Subclasses must implement load_model()")
    
    def create_dataloader(self, config: OmegaConf, split: str = 'test', batch_size: Optional[int] = None):
        """
        Create dataloader for evaluation. Must be implemented by subclasses.
        
        Args:
            config: Configuration object
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size (if None, uses config default)
            
        Returns:
            DataLoader instance
        """
        raise NotImplementedError("Subclasses must implement create_dataloader()")
    
    
    def load_oracle_model(self, oracle_checkpoint: str, data_path: str):
        """
        Load oracle model for specialized evaluation metrics.
        Can be overridden by subclasses for dataset-specific oracle loading.
        
        Args:
            oracle_checkpoint: Path to oracle model checkpoint
            data_path: Path to data file needed by oracle
            
        Returns:
            Oracle model instance or None if not supported
        """
        print(f"Warning: Oracle evaluation not implemented for {self.dataset_name}")
        return None
    
    def sample_sequences_for_evaluation(self, checkpoint_path: str, config: OmegaConf, 
                                       dataloader, num_steps: int, architecture: str = 'transformer', 
                                       show_progress: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample sequences for evaluation using PC sampler.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object  
            dataloader: DataLoader for getting target labels
            num_steps: Number of sampling steps
            architecture: Architecture type
            show_progress: Whether to show progress bar during sampling
            
        Returns:
            Tuple of (sampled_sequences, target_labels)
        """
        from scripts import sampling
        
        # Load model using dataset-specific method  
        model, graph, noise = self.load_model(checkpoint_path, config, architecture)
        model.eval()
        
        # Get sequence length
        sequence_length = self.get_sequence_length(config)
        
        # Get batch size from dataloader
        batch_size = dataloader.batch_size
        
        # Create PC sampler once (will be reused for all batches)
        sampling_fn = sampling.get_pc_sampler(
            graph, noise, (batch_size, sequence_length), 'analytic', num_steps, device=self.device
        )
        
        sampled_sequences = []
        all_targets = []
        
        # Wrap dataloader with tqdm if show_progress is True
        if show_progress:
            dataloader = tqdm(dataloader, desc="Sampling sequences")
        
        for batch_idx, (batch, targets) in enumerate(dataloader):
            current_batch_size = batch.shape[0]
            
            # If last batch has different size, create new sampling function
            if current_batch_size != batch_size:
                sampling_fn = sampling.get_pc_sampler(
                    graph, noise, (current_batch_size, sequence_length), 'analytic', num_steps, device=self.device
                )
            
            # Sample sequences conditioned on targets
            sample = sampling_fn(model, targets.to(self.device))
            seq_pred_one_hot = F.one_hot(sample, num_classes=4).float()
            sampled_sequences.append(seq_pred_one_hot)
            all_targets.append(targets)
        
        # Concatenate all samples
        all_samples = torch.cat(sampled_sequences, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        return all_samples, all_targets
    
    def get_sequence_length(self, config: OmegaConf) -> int:
        """
        Get sequence length for the dataset. Can be overridden by subclasses.
        
        Args:
            config: Configuration object
            
        Returns:
            Sequence length
        """
        # Try common config locations
        if hasattr(config, 'dataset') and hasattr(config.dataset, 'sequence_length'):
            return config.dataset.sequence_length
        elif hasattr(config, 'model') and hasattr(config.model, 'length'):
            return config.model.length
        elif hasattr(config, 'data') and hasattr(config.data, 'sequence_length'):
            return config.data.sequence_length
        else:
            # Dataset-specific defaults - should be overridden by subclasses
            defaults = {
                'deepstarr': 249,
                'mpra': 200,
                'promoter': 1024,
                'atacseq': 1001
            }
            return defaults.get(self.dataset_name.lower(), 249)
    
    def compute_sp_mse(self, sampled_sequences: torch.Tensor, oracle_model, 
                      original_data: torch.Tensor) -> float:
        """
        Compute SP-MSE using oracle model.
        
        Args:
            sampled_sequences: Generated sequences (token indices)
            oracle_model: Oracle model for evaluation
            original_data: Original test data for comparison
            
        Returns:
            SP-MSE score
        """
        
        # Get oracle predictions for original and generated data
        val_score = oracle_model.predict_custom(original_data.to(self.device))
        val_pred_score = oracle_model.predict_custom(sampled_sequences.permute(0, 2, 1).to(self.device))
        
        # Compute SP-MSE
        sp_mse = (val_score - val_pred_score) ** 2
        mean_sp_mse = torch.mean(sp_mse).cpu().item()
        
        return mean_sp_mse
    
    def get_original_test_data(self, data_path: str) -> torch.Tensor:
        """
        Get original test data for SP-MSE comparison. Must be implemented by subclasses.
        
        Args:
            data_path: Path to data file
            
        Returns:
            Original test data tensor
        """
        raise NotImplementedError("Subclasses must implement get_original_test_data()")
    
    def evaluate_with_oracle(self, model, oracle_model, dataloader, config: OmegaConf) -> Dict[str, Any]:
        """
        Evaluate model using oracle model (for SP-MSE and similar metrics).
        Can be overridden by subclasses for dataset-specific oracle evaluation.
        
        Args:
            model: Trained diffusion model
            oracle_model: Oracle model for evaluation
            dataloader: Data loader
            config: Configuration object
            
        Returns:
            Dictionary of oracle-based evaluation metrics
        """
        if oracle_model is None:
            return {'oracle_evaluation': 'not_supported'}
        
        # Default implementation returns placeholder
        return {
            'oracle_evaluation': 'not_implemented',
            'sp_mse': 0.0
        }
    
    def evaluate_with_sampling(self, checkpoint_path: str, config: OmegaConf, 
                              oracle_checkpoint: str, data_path: str,
                              split: str = 'test', steps: Optional[int] = None, 
                              batch_size: Optional[int] = None, architecture: str = 'transformer',
                              show_progress: bool = False, save_sequences: bool = False) -> Dict[str, Any]:
        """
        Evaluate model by sampling sequences and computing SP-MSE with oracle.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            oracle_checkpoint: Path to oracle model checkpoint
            data_path: Path to data file needed by oracle
            split: Dataset split to evaluate on
            steps: Number of sampling steps (defaults to sequence length)
            batch_size: Batch size for evaluation (optional)
            architecture: Architecture type
            show_progress: Whether to show progress bar during sampling
            save_sequences: Whether to save sampled sequences as NPZ file
            
        Returns:
            Dictionary of evaluation results including SP-MSE
        """
        print(f"Evaluating {self.dataset_name} on {split} split with sampling...")
        
        # Set default steps to sequence length if not provided
        if steps is None:
            steps = self.get_sequence_length(config)
            print(f"Using default steps: {steps} (sequence length)")
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)
        
        # Sample sequences using PC sampler
        print(f"Sampling sequences with PC sampler ({steps} steps)...")
        sampled_sequences, target_labels = self.sample_sequences_for_evaluation(
            checkpoint_path, config, dataloader, steps, architecture, show_progress
        )
        
        # Save sequences as NPZ if requested
        if save_sequences:
            # Create output path based on checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_path)
            npz_path = os.path.join(checkpoint_dir, "sample.npz")
            self.save_sequences_as_npz(sampled_sequences, npz_path)
        
        # Load oracle model
        print("Loading oracle model for SP-MSE evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path)
        
        if oracle_model is None:
            return {
                'error': 'oracle_model_not_loaded',
                'num_samples': sampled_sequences.shape[0],
                'sequence_length': sampled_sequences.shape[1],
                'sampling_steps': steps
            }
        
        # Get original test data for comparison
        original_data = self.get_original_test_data(data_path)
        
        # Compute SP-MSE
        print("Computing SP-MSE...")
        sp_mse = self.compute_sp_mse(sampled_sequences, oracle_model, original_data)
        
        results = {
            'dataset': self.dataset_name,
            'split': split,
            'num_samples': sampled_sequences.shape[0],
            'sequence_length': sampled_sequences.shape[1],
            'sampling_steps': steps,
            'sp_mse': sp_mse,
            'oracle_evaluation': 'completed'
        }
        
        print(f"SP-MSE: {sp_mse:.6f}")
        
        return results
    
    def save_sequences_as_npz(self, sequences: torch.Tensor, save_path: str):
        """
        Save sampled sequences as NPZ file.
        
        Args:
            sequences: Sampled sequences tensor (batch_size, seq_len, 4)
            save_path: Path to save the NPZ file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez(save_path, sequences.cpu().numpy())
        print(f"✓ Saved {sequences.shape[0]} sequences to {save_path}")

    def evaluate(self, checkpoint_path: str, config: OmegaConf, architecture: str = 'transformer',
                 split: str = 'test', oracle_checkpoint: Optional[str] = None,
                 data_path: Optional[str] = None, batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Main evaluation method - now deprecated. Use evaluate_with_sampling instead.
        
        Args:
            checkpoint_path: Path to checkpoint file
            config: Configuration object
            architecture: Architecture type
            split: Dataset split to evaluate on
            oracle_checkpoint: Path to oracle model checkpoint (optional)
            data_path: Path to data file (optional, needed for oracle)
            batch_size: Batch size for evaluation (optional)
            
        Returns:
            Dictionary of evaluation results
        """
        print("Warning: evaluate() method is deprecated. Use evaluate_with_sampling() instead.")
        print("This method only performs oracle evaluation without sampling.")
        
        if not oracle_checkpoint:
            return {'error': 'oracle_checkpoint required for evaluation'}
        
        # Load model using dataset-specific method
        model, _, _ = self.load_model(checkpoint_path, config, architecture)
        
        # Create dataloader
        dataloader = self.create_dataloader(config, split, batch_size)
        
        print(f"Evaluating {self.dataset_name} on {split} split...")
        
        # Oracle evaluation only
        print("Loading oracle model for specialized evaluation...")
        oracle_model = self.load_oracle_model(oracle_checkpoint, data_path or "")
        metrics = self.evaluate_with_oracle(model, oracle_model, dataloader, config)
        
        return metrics
    
    def save_results(self, metrics: Dict[str, Any], output_path: str):
        """Save evaluation results to file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Results saved to: {output_path}")
    
    def print_results(self, metrics: Dict[str, Any]):
        """Print evaluation results in a formatted way."""
        print(f"\n{self.dataset_name} Evaluation Results:")
        print("=" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


def parse_base_args():
    """Parse common command line arguments for evaluation scripts."""
    parser = argparse.ArgumentParser(description='D3 Evaluation Script - Sampling + SP-MSE')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint file')
    parser.add_argument('--architecture', required=True, choices=['transformer', 'convolutional'], help='Model architecture')
    parser.add_argument('--oracle_checkpoint', required=True, help='Path to oracle model checkpoint (required for SP-MSE)')
    parser.add_argument('--data_path', required=True, help='Path to data file (required for oracle models)')
    parser.add_argument('--config', help='Path to config file (optional, dataset may provide default)')
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='test', help='Dataset split to evaluate on')
    parser.add_argument('--steps', type=int, help='Number of sampling steps (defaults to sequence length)')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--batch_size', type=int, help='Batch size for evaluation')
    parser.add_argument('--show_progress', action='store_true', help='Show progress bar during sampling')
    parser.add_argument('--save_sequences', action='store_true', help='Save sampled sequences as NPZ file')
    
    return parser


def main_evaluate(evaluator: BaseEvaluator, args):
    """
    Common main evaluation function that can be used by dataset-specific scripts.
    Now focuses on sampling + SP-MSE evaluation only.
    
    Args:
        evaluator: Dataset-specific evaluator instance
        args: Parsed command line arguments
    """
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        return 1
    
    if not args.oracle_checkpoint:
        print("Error: --oracle_checkpoint is required for SP-MSE evaluation")
        return 1
    
    if not args.data_path:
        print("Error: --data_path is required for SP-MSE evaluation")
        return 1
    
    # Load configuration (required)
    if not args.config:
        print("Error: Config file is required. Please provide --config path/to/config.yaml")
        return 1
    
    config = OmegaConf.load(args.config)
    
    # Handle progress bar settings
    show_progress = args.show_progress
    
    # Run evaluation with sampling (the main evaluation method now)
    metrics = evaluator.evaluate_with_sampling(
        checkpoint_path=args.checkpoint,
        config=config,
        oracle_checkpoint=args.oracle_checkpoint,
        data_path=args.data_path,
        split=args.split,
        steps=getattr(args, 'steps', None),
        batch_size=args.batch_size,
        architecture=args.architecture,
        show_progress=show_progress,
        save_sequences=args.save_sequences
    )
    
    # Print results
    evaluator.print_results(metrics)
    
    # Save results
    dataset_name = evaluator.dataset_name
    output_path = args.output or f"evaluation_results/{dataset_name}_{args.architecture}_{args.split}_results.json"
    evaluator.save_results(metrics, output_path)
    
    return 0
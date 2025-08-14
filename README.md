## ⚡ Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/ewonglab/D3-Model-Dev
cd D3-Model-Dev

# Install in development mode (recommended)
pip install -e .
```

## Development Commands

### Training Commands
- **2-stage training pipeline**: `bash scripts/run_training.sh` - Runs pre-training followed by fine-tuning
- **Pre-training only**: `python train.py --mode pre-training`
- **Fine-tuning only**: `python train.py --mode fine-tuning`
- **Run training**: `python run_train.py` - Direct training script with distributed support
- **Generate samples**: `python run_sample.py --model_path <path> --input_data <file.h5>` - Generate DNA sequences from trained models

### Key Parameters
- Training configurations are in `configs/pre-training.yaml` and `configs/fine-tuning.yaml`
- Batch sizes: 256 for pre-training, 64 for fine-tuning
- Learning rates: 0.0003 for pre-training, 0.00005 for fine-tuning
- Model checkpoints saved in `exp_local/<dataset>/<timestamp>/checkpoints-<mode>/`

## Architecture Overview

### Core Components
- **D3 Model (`src/models/d3_model.py`)**: Main diffusion transformer model for DNA sequence generation
- **SEDD Framework**: Score-based diffusion model using transformer architecture (768 hidden size, 12 blocks, 12 heads)
- **Graph Library (`src/utils/graph_lib.py`)**: Handles token graph construction for sequence modeling
- **Noise Library (`src/utils/noise_lib.py`)**: Geometric noise scheduling for diffusion process

### Data Pipeline
- **Input**: HDF5 files with one-hot encoded DNA sequences (200bp length)
- **Processing**: `src/data/data_processing.py` handles data loading and batching
- **Format**: Sequences stored as (batch, sequence_length) with 4-class tokens (A,C,G,T)
- **Main dataset**: `data/c5_human_heart_data.h5` for human heart regulatory sequences

### Training Flow
1. **Pre-training**: Uses `c5_human_heart` dataset with 25,600 iterations
2. **Fine-tuning**: Uses specialized fine-tuning dataset with 3,200 iterations
3. **Distributed**: Multi-GPU support via PyTorch DDP with configurable `ngpus`
4. **Monitoring**: Weights & Biases integration, EMA model averaging
5. **Sampling**: Euler predictor with 128 steps for sequence generation

### Key Files
- `train.py`: Main entry point with Hydra configuration management
- `run_train.py`: Core distributed training loop with logging
- `run_sample.py`: Inference script for generating DNA sequences
- `src/sampling.py`: Sampling algorithms for sequence generation
- `src/models/ema.py`: Exponential moving average for model parameters

### Configuration System
- Uses Hydra for configuration management
- Base configs in `configs/` with model-specific configs in `configs/model/`
- Override via command line: `python train.py wandb_project=my_project ngpus=4`
- Experiments logged to `exp_local/` with timestamped directories
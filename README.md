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
- **Generate samples**: `python run_sample.py --model_path <path> --input_data <file.h5>` - Generate DNA sequences from trained models with optional attention/PCA analysis

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
- **Processing**: `data_processing.py` handles data loading and batching
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

## 🧬 Sequence Generation & Analysis

The `run_sample.py` script generates DNA sequences from trained D3 models and supports advanced analysis features including attention visualization and PCA-based feature extraction.

### Basic Usage

```bash
# Generate sequences using default settings
python run_sample.py --model_path path/to/model.pth --input_data data/test_data.h5

# Generate with custom batch size and sampling steps
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --batch_size 256 \
    --steps 200 \
    --results_dir custom_results/
```

### Advanced Analysis Features

#### Attention Score Extraction
Extract and save attention matrices from transformer layers for sequence interpretability:

```bash
# Save attention scores from the last transformer layer (default)
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --save_attention

# Extract attention from the last layer (0-indexed)
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --save_attention \
    --attention_layer 11

# Here is an example command to extract attention from layer 12
python run_sample.py \
    --model_path outputs \
    --input_data data/c5_human_heart_data.h5 \
    --results_dir outputs/attention_results \
    --save_attention \
    --attention_layer 12
```

**Output**: `attention_{cell_type}.pt` files containing attention matrices for each cell type

#### PCA Feature Extraction
Perform Principal Component Analysis on transformer layer outputs to identify key sequence patterns:

```bash
# Extract PCA features with default 5 components
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --save_pca_features

# Customize number of PCA components
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --save_pca_features \
    --pca_components 10
```

**Output**: `pca_features_{cell_type}.pt` files containing PCA scores for each cell type

#### Combined Analysis
Extract both attention scores and PCA features simultaneously:

```bash
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --save_attention \
    --attention_layer 10 \
    --save_pca_features \
    --pca_components 8 \
    --batch_size 128 \
    --steps 250
```

### Reproducible Generation
Use random seeds for consistent sequence generation across runs:

```bash
# Generate with custom random seeds
python run_sample.py \
    --model_path path/to/model.pth \
    --input_data data/test_data.h5 \
    --random_seeds path/to/seeds.json \
    --save_attention \
    --save_pca_features
```

**Seeds file format**:
```json
{
    "seeds": [42, 123, 456, 789, 101112]
}
```

### Output Structure

Generated results are organized by cell type with the following structure:
```
results_dir/
├── sample_Endothelial.npz           # One-hot encoded sequences
├── final_Endothelial.txt            # DNA sequences as text
├── attention_Endothelial.pt         # Attention matrices (if enabled)
├── pca_features_Endothelial.pt      # PCA scores (if enabled)
├── sample_Fibroblast.npz
├── final_Fibroblast.txt
└── ...
```

### Complete Command Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model_path` | str | required | Path to trained model checkpoint |
| `--input_data` | str | required | Path to input HDF5 dataset |
| `--batch_size` | int | 500 | Batch size for processing |
| `--steps` | int | 200 | Number of sampling steps |
| `--results_dir` | str | `model_results/...` | Output directory for results |
| `--random_seeds` | str | None | Path to JSON file with random seeds |
| `--save_attention` | flag | False | Enable attention score extraction |
| `--attention_layer` | int | None | Specific layer for attention (default: last) |
| `--save_pca_features` | flag | False | Enable PCA feature extraction |
| `--pca_components` | int | 5 | Number of PCA components to extract |

### Performance Considerations

- **Memory Usage**: Attention and PCA extraction increase memory requirements. Reduce `--batch_size` if encountering OOM errors
- **Storage**: Feature extraction generates additional files (~10-50MB per cell type depending on dataset size)
- **Processing Time**: PCA computation adds ~10-30% overhead to sampling time
- **Attention Layer Selection**: Later layers (8-11) typically contain more semantic information for regulatory sequences
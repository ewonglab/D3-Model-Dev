#!/bin/bash

#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=2:00:00
#PBS -m ae
#PBS -j oe
module load cuda/11.8.0
source /srv/scratch/z5098304/miniconda3/bin/activate

conda create -n d3 python=3.10 -y

conda activate d3

pip install hydra-core omegaconf
pip install dm-tree biotite h5py scipy scikit-learn tqdm
pip install wandb
pip install numpy==1.25.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install pytorch-lightning==2.3.3
pip install transformers==4.38.1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.2.2/flash_attn-2.2.2+cu117torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
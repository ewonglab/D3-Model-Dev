#!/bin/bash

pip install hydra-core omegaconf
pip install dm-tree biotite h5py scipy scikit-learn tqdm
pip install wandb  
pip install numpy==1.25.1
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install pytorch-lightning==2.3.3
pip install transformers==4.38.1
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.2.2/flash_attn-2.2.2+cu117torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
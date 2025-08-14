import os
import torch
from models import SEDD
from utils import utils
from models.ema import ExponentialMovingAverage
from utils import graph_lib
from utils import noise_lib

from omegaconf import OmegaConf


def load_model_hf(dir, device):
    print(dir)
    score_model = SEDD.from_pretrained(dir).to(device)
    graph = graph_lib.get_graph(score_model.config, device)
    noise = noise_lib.get_noise(score_model.config).to(device)
    return score_model, graph, noise


def load_model_local(root_dir, device, mode="train"):
    cfg = utils.load_hydra_config_from_run(root_dir)
    graph = graph_lib.get_graph(cfg, device)
    noise = noise_lib.get_noise(cfg).to(device)
    score_model = SEDD(cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=cfg.training.ema)

    # Load the checkpoint using the input
    if mode == "train":
        ckpt_dir = os.path.join(root_dir, "checkpoints-train", "checkpoint.pth")
    elif mode == "finetune":
        ckpt_dir = os.path.join(root_dir, "checkpoints-finetune", "checkpoint.pth")
    else:
        raise ValueError(f"Invalid mode: {mode}")

    loaded_state = torch.load(ckpt_dir, map_location=device)

    score_model.load_state_dict(loaded_state["model"])
    ema.load_state_dict(loaded_state["ema"])

    ema.store(score_model.parameters())
    ema.copy_to(score_model.parameters())
    return score_model, graph, noise


def load_model(root_dir, device):
    return load_model_hf(root_dir, device)

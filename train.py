"""Training and evaluation"""

import hydra
import os
import numpy as np
import run_train
from src.utils import utils
import torch.multiprocessing as mp
import wandb
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict
import argparse


def get_config_name():
    """Parse command line arguments to get config name"""
    parser = argparse.ArgumentParser(description="Training script for 2-stage model")
    parser.add_argument(
        "--mode",
        choices=["pre-training", "fine-tuning"],
        default="pre-training",
        help="Config file to use: pre-training or fine-tuning (default: pre-training)",
    )

    # Parse only known args to avoid conflicts with Hydra
    args, _ = parser.parse_known_args()
    return args.mode


# Get config name from command line
CONFIG_NAME = get_config_name()


@hydra.main(version_base=None, config_path="configs", config_name=CONFIG_NAME)
def main(cfg):
    ngpus = cfg.ngpus
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, "hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)

        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        work_dir = (
            hydra_cfg.run.dir
            if hydra_cfg.mode == RunMode.RUN
            else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        )
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

    # Initialize wandb in the main process and store the run id
    wandb_run_id = None
    if cfg.get("wandb_project", None):
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
        logger.info(f"Initializing wandb project: {cfg.wandb_project}")
        wandb_run = wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_name,
            config=OmegaConf.to_container(cfg, resolve=True),
            dir=work_dir,
        )
        wandb_run_id = wandb_run.id
        logger.info(f"Created wandb run with ID: {wandb_run_id}")

        # Add wandb run ID to config so child processes can access it
        with open_dict(cfg):
            cfg.wandb_run_id = wandb_run_id

    # Log which training mode is being used
    logger = utils.get_logger(os.path.join(work_dir, "logs"))
    logger.info(f"Training mode: {cfg.mode}")
    logger.info(f"Using config: {CONFIG_NAME}.yaml")

    # Run the training pipeline
    port = int(np.random.randint(10000, 20000))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(
            run_train.run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True
        )
    except Exception as e:
        logger.critical(e, exc_info=True)
    finally:
        # Close wandb run if it was initialized
        if cfg.get("wandb_project", None):
            logger.info("Finishing wandb run")
            wandb.finish()


if __name__ == "__main__":
    main()

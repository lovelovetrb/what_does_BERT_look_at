import logging

import torch
import torch.distributed as dist
from transformers import set_seed
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - PID: %(process)d -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def init_config(config: dict):
    wandb.init(
        # set the wandb project where this run will be logged
        project="calc_attention",
        config=config,
    )

    set_seed(config["basic"]["seed"])


def init_gpu(args):
    logger.info("Initializing GPUs...")
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
    local_rank = dist.get_rank()
    args.local_rank = local_rank
    args.world_size = dist.get_world_size()
    args.is_master = local_rank == 0
    args.device = torch.device("cuda")

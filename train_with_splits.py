#!/usr/bin/env python3
"""
Training script with support for dataset splits
"""

import argparse
import os
import sys

# Add the metric_depth directory to Python path
sys.path.insert(0, '/app/metric_depth')

from zoedepth.utils.misc import count_parameters, parallelize
from zoedepth.utils.config import get_config
from zoedepth.utils.arg_utils import parse_unknown
from zoedepth.trainers.builder import get_trainer
from zoedepth.models.builder import build_model
from zoedepth.data.diode_splits import get_diode_train_loader, get_diode_val_loader
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch
import numpy as np
from pprint import pprint

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["WANDB_START_METHOD"] = "thread"


def fix_random_seed(seed: int):
    import random
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main_worker(gpu, ngpus_per_node, config):
    try:
        config.gpu = gpu
        config.multigpu = False
        if config.gpu is not None:
            config.multigpu = ngpus_per_node > 1

        config.rank = config.node_rank * ngpus_per_node + gpu
        print("Config:")
        pprint(config)

        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            config.batch_size = int(config.batch_size / ngpus_per_node)
            config.workers = int((config.workers + ngpus_per_node - 1) / ngpus_per_node)
        
        if config.distributed:
            torch.distributed.init_process_group(backend=config.dist_backend, init_method=config.dist_url,
                                                  world_size=config.world_size, rank=config.rank)

        model = build_model(config)
        if config.gpu is not None:
            torch.cuda.set_device(config.gpu)
            model = model.cuda(config.gpu)

        config.multigpu = ngpus_per_node > 1
        if config.multigpu:
            model = parallelize(config, model)

        total_params = f"{round(count_parameters(model)/1e6,2)}M"
        config.total_params = total_params
        print(f"Total parameters : {total_params}")

        # Load data using splits
        print(f"Loading data from splits directory: {config.splits_dir}")
        train_loader = get_diode_train_loader(
            config.splits_dir, 
            batch_size=config.batch_size,
            num_workers=config.workers
        )
        
        val_loader = get_diode_val_loader(
            config.splits_dir,
            batch_size=1,  # Use batch size 1 for validation
            num_workers=config.workers
        )

        trainer = get_trainer(config)(
            config, model, train_loader, val_loader, device=config.gpu)

        trainer.train()
    finally:
        import wandb
        wandb.finish()


def main():
    # Seed
    fix_random_seed(1234)

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="zoedepth")
    parser.add_argument("-d", "--dataset", type=str, default='diode_outdoor')
    parser.add_argument("--trainer", type=str, default=None)
    parser.add_argument("--splits_dir", type=str, required=True,
                       help="Directory containing dataset split files")

    args, unknown_args = parser.parse_known_args()
    unknown_args = parse_unknown(unknown_args)

    overwrite_kwargs = {**unknown_args}
    overwrite_kwargs["model"] = args.model
    overwrite_kwargs["splits_dir"] = args.splits_dir

    if args.trainer is not None:
        overwrite_kwargs["trainer"] = args.trainer

    config = get_config(args.model, "train", args.dataset, **overwrite_kwargs)
    
    # Add splits directory to config
    config.splits_dir = args.splits_dir
    
    config.batch_size = config.bs
    config.mode = 'train'
    if config.root != "." and not os.path.isdir(config.root):
        os.makedirs(config.root)

    try:
        node_str = os.environ['SLURM_JOB_NODELIST'].replace(
            '[', '').replace(']', '')
        nodes = node_str.split(',')

        config.world_size = len(nodes)
        config.rank = int(os.environ['SLURM_PROCID'])

    except KeyError as e:
        # We are NOT using SLURM
        config.world_size = 1
        config.rank = 0
        nodes = ["127.0.0.1"]

    if config.distributed:
        mp.set_start_method('forkserver')

        port = np.random.randint(15000, 15025)
        config.dist_url = 'tcp://{}:{}'.format(nodes[0], port)
        config.dist_backend = 'nccl'
        config.gpu = None

    config.num_workers = config.workers
    config.ngpus_per_node = torch.cuda.device_count()

    if config.gpu is not None:
        config.ngpus_per_node = 1

    if config.ngpus_per_node == 1:
        config.distributed = False

    config.num_workers = config.workers
    config.ngpus_per_node = ngpus_per_node = torch.cuda.device_count() if config.distributed else 1
    print("Device count: ", ngpus_per_node)
    config.node_rank = 0
    if config.distributed:
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, config))
    else:
        if ngpus_per_node == 1:
            config.gpu = 0
        main_worker(config.gpu, ngpus_per_node, config)


if __name__ == '__main__':
    main()

import datetime
import os
import random

import timm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from lark.config import Config
from lark.learner import Learner
from lark.ops import MixedSig2Spec


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_fn(fn, world_size):
    mp.spawn(fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


def make_model(cfg, rank):
    prep = MixedSig2Spec(cfg, rank)
    main_model = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    main_model.classifier = torch.nn.Linear(in_features=1280, out_features=len(cfg.labels), bias=True)
    model = torch.nn.Sequential(prep, main_model)
    return model.to(rank)


def do_work(rank, world_size):
    setup(rank, world_size)
    cfg = Config(
        sites=['COR', 'SSW'],
        use_neptune=True,
        n_epochs=10,
        bs=64,
        n_samples_per_label=2000,
        lr=1e-3,
        model='tf_efficientnet_b0_ns',
        scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        loss_fn='lark.ops.SigmoidFocalLossStar',
        use_pink_noise=0.1,
        use_recorded_noise=0.2,
        use_overlays=True,
        apply_filter=0.1,
        seed=231,
        n_workers=6,
    )
    model = make_model(cfg, rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    lrn = Learner("tf_efficientnet_b0_ns-full", cfg, rank, ddp_model)
    lrn.learn()
    cleanup()


if __name__ == "__main__":
    size = 2
    run_fn(do_work, size)

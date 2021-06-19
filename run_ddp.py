import os

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


class Backbone(torch.nn.Module):
    def __init__(self, name='resnet18', pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)
        if 'regnet' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'vit' in name:
            self.out_features = self.net.head.in_features
        elif 'csp' in name:
            self.out_features = self.net.head.fc.in_features
        elif 'res' in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif 'efficientnet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'densenet' in name:
            self.out_features = self.net.classifier.in_features
        elif 'senet' in name:
            self.out_features = self.net.fc.in_features
        elif 'inception' in name:
            self.out_features = self.net.last_linear.in_features
        else:
            self.out_features = self.net.classifier.in_features

        # remove unused parameters, otherwise DistributedDataParallel will balk
        # see https://rwightman.github.io/pytorch-image-models/feature_extraction/#remove-it-later
        self.net.reset_classifier(0, '')

        self.global_pool = torch.nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.net.forward_features(x)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        return x


def make_model(cfg: Config, rank: int):
    prep = MixedSig2Spec(cfg, rank)
    # backbone = timm.create_model('tf_efficientnet_b0_ns', pretrained=True)
    backbone = Backbone('tf_efficientnet_b0_ns', pretrained=True)

    embedding_size = 512
    neck = torch.nn.Sequential(
        torch.nn.Dropout(0.3),
        torch.nn.Linear(in_features=backbone.out_features, out_features=embedding_size, bias=True),
        torch.nn.BatchNorm1d(embedding_size),
        torch.nn.PReLU()
    )

    head = torch.nn.Linear(in_features=embedding_size, out_features=cfg.n_labels)

    # backbone.classifier = torch.nn.Linear(in_features=1280, out_features=len(cfg.labels), bias=True)
    model = torch.nn.Sequential(prep, backbone, neck, head)
    return model.to(rank)


def do_work(rank, world_size):
    setup(rank, world_size)
    cfg = Config(
        sites=['SSW', 'COR'],
        use_neptune=False,
        n_epochs=10,
        bs=32,
        n_samples_per_label=300,
        # n_samples_per_label=100,
        lr=1e-3,
        model='tf_efficientnet_b0_ns',
        scheduler='torch.optim.lr_scheduler.CosineAnnealingLR',
        loss_fn='lark.ops.SigmoidFocalLossStar',
        # loss_fn='lark.ops.SigmoidFocalLoss',
        use_pink_noise=0.1,
        use_recorded_noise=0.2,
        use_overlays=True,
        apply_filter=0.1,
        seed=231,
        n_workers=6,
    )
    model = make_model(cfg, rank)
    ddp_model = DistributedDataParallel(model, device_ids=[rank])
    lrn = Learner("tf_efficientnet_b0_ns-cor+ssw", cfg, rank, ddp_model)
    lrn.learn()
    cleanup()


if __name__ == "__main__":
    size = 2
    run_fn(do_work, size)

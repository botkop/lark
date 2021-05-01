import dataclasses
import itertools
from dataclasses import dataclass
from functools import partial
from typing import List

import pandas as pd


@dataclass
class Config:
    # data parameters
    site: str
    data_dir: str = 'data/birdclef-2021'
    checkpoint_dir: str = 'checkpoints'
    bs: int = 32
    n_workers: int = 12
    training_dataset_size: int = bs * n_workers * 28  # 28 = number of labels
    duration: float = 5

    # augmentation
    use_noise: bool = True
    noise_nsr_dbs: List[int] = dataclasses.field(default_factory=lambda: [20, 10, 3])
    noise_dir: str = 'data/noise/BirdVox-DCASE-20k/wav-32k'
    use_overlays: bool = True
    max_overlays: int = 5
    overlay_weights: List[float] = dataclasses.field(default_factory=lambda: [
        0.71986223,
        0.21010333,
        0.06314581,
        0.00574053,
        0.00114811])
    overlay_snr_dbs: List[int] = dataclasses.field(default_factory=lambda: [20, 10, 3])

    # logging
    use_neptune: bool = False

    # sig parameters
    sr: int = 32000
    n_frames: int = duration * sr
    n_fft: int = 512
    window_length: int = n_fft
    n_mels: int = 64
    hop_length: int = 312
    f_min: int = 150
    f_max: int = 15000

    # learner parameters
    lr: float = 1e-3
    n_epochs: int = 10
    model: str = 'Cnn14_DecisionLevelAtt'
    optimizer: str = 'torch.optim.Adam'
    loss_fn: str = 'torch.nn.BCEWithLogitsLoss'
    # loss_fn: str = 'PANNsLoss'
    scheduler: str = 'torch.optim.lr_scheduler.CosineAnnealingLR'
    # scheduler: str = 'torch.optim.lr_scheduler.OneCycleLR'

    default_scheduler_params = {
        "torch.optim.lr_scheduler.OneCycleLR": {
            "max_lr": lr * 10,
            "steps_per_epoch": training_dataset_size,
            "epochs": n_epochs
        },
        'torch.optim.lr_scheduler.CosineAnnealingLR': {
            "eta_min": 1e-5,
            "T_max":  n_epochs
        }
    }

    @property
    def scheduler_params(self):
        return self.default_scheduler_params[self.scheduler]

    @property
    def labels(self):
        df_ss = pd.read_csv(f"{self.data_dir}/train_soundscape_labels.csv")
        if self.site is not None:
            df_ss = df_ss[df_ss['site'] == self.site].reset_index(drop=True)
        labels = [x for x in sorted(set(itertools.chain(*df_ss['birds'].str.split(' ')))) if x != 'nocall']
        return labels

    def as_dict(self):
        d = dataclasses.asdict(self)
        d['labels'] = self.labels
        d['scheduler_params'] = self.scheduler_params
        return d

    @property
    def instantiate_model(self):
        import torch  # type: ignore
        from lark.pann import Cnn14_DecisionLevelAtt  # type: ignore
        return eval(self.model)

    @property
    def instantiate_loss(self):
        import torch  # type: ignore
        from lark.pann import PANNsLoss  # type: ignore
        return eval(self.loss_fn)

    @property
    def instantiate_optimizer(self):
        import torch  # type: ignore
        return partial(eval(self.optimizer), lr=self.lr)

    @property
    def instantiate_scheduler(self):
        import torch  # type: ignore
        return partial(eval(self.scheduler), **self.scheduler_params)


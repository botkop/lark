import dataclasses
import glob
import itertools
import os.path
from dataclasses import dataclass
from functools import partial
from typing import List
import torchaudio as ta

import pandas as pd


@dataclass
class Config:
    # data parameters
    sites: List[str]
    data_dir: str = 'data/birdclef-2021'
    checkpoint_dir: str = 'checkpoints'
    bs: int = 32
    n_samples_per_label: int = 1000
    n_workers: int = 12
    train_duration: float = 5
    valid_duration: float = 5

    seed: int = 231

    # augmentation
    use_pink_noise: float = 0.5
    use_recorded_noise: float = 0.5
    use_secondary_labels: bool = False
    noise_nsr_dbs: List[int] = dataclasses.field(default_factory=lambda: [20, 10, 3])
    pink_noise_nsr_dbs: List[int] = dataclasses.field(default_factory=lambda: [10, 5, 3])
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
    apply_filter: float = 0.6

    # logging
    use_neptune: bool = True
    log_batch_metrics: bool = False

    # sig parameters
    sr: int = 32000

    n_fft: int = 512
    window_length: int = n_fft

    n_mels: int = 64

    hop_length: int = 312

    f_min: int = 150
    f_max: int = 15000
    f1_threshold: float = 0.5

    # learner parameters
    lr: float = 1e-3
    n_epochs: int = 10
    model: str = 'Cnn14_DecisionLevelAtt'
    optimizer: str = 'torch.optim.Adam'
    loss_fn: str = 'torch.nn.BCEWithLogitsLoss'
    scheduler: str = 'torch.optim.lr_scheduler.CosineAnnealingLR'

    # scheduler: str = 'torch.optim.lr_scheduler.OneCycleLR'

    @property
    def df_train_meta(self):
        df_meta = pd.read_csv(f"{self.data_dir}/train_metadata.csv")
        df_meta['secondary_labels'] = df_meta['secondary_labels'].str.replace("[\[\]',]", '',
                                                                              regex=True).str.split()
        df_meta['filename'] = df_meta['filename'].str.replace(".ogg", '.wav', regex=False)
        df_meta['filename'] = df_meta.apply(
            lambda r: f"{self.data_dir}/train_short_audio.wav/{r['primary_label']}/{r['filename']}", axis=1)

        df_meta['file_exists'] = df_meta.apply(lambda r: os.path.exists(r['filename']), axis=1)
        df_meta = df_meta[df_meta['file_exists']]
        df_meta['n_frames'] = df_meta.apply(lambda r: ta.info(r['filename']).num_frames, axis=1)
        return df_meta

    @property
    def thr_dict(self):
        return {
            'high': 0.9,
            'median': 0.5,
            'low': 0.1,
            'corr': 0.6
        }

    @property
    def schedule_per_epoch(self) -> bool:
        d = {
            'torch.optim.lr_scheduler.CosineAnnealingLR': True,
            'torch.optim.lr_scheduler.OneCycleLR': False,
            'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts': False
        }
        return d[self.scheduler]

    @property
    def scheduler_params(self):
        default_params = {
            "torch.optim.lr_scheduler.OneCycleLR": {
                "max_lr": self.lr * 10,
                "steps_per_epoch": (self.training_dataset_size // self.bs) + self.training_dataset_size % self.bs,
                "epochs": self.n_epochs
            },
            'torch.optim.lr_scheduler.CosineAnnealingLR': {
                "eta_min": 1e-5,
                "T_max": 10
            },
            'torch.optim.lr_scheduler.CosineAnnealingWarmRestarts': dict(
                T_0=10, T_mult=2, eta_min=0.01, last_epoch=-1)
        }
        return default_params[self.scheduler]

    @property
    def labels(self):
        if self.sites is not None:
            df_ss = pd.read_csv(f"{self.data_dir}/train_soundscape_labels.csv")
            df_ss = df_ss[df_ss['site'].isin(self.sites)].reset_index(drop=True)
            labels = [x for x in sorted(set(itertools.chain(*df_ss['birds'].str.split(' ')))) if x != 'nocall']
        else:
            labels = sorted([os.path.basename(d) for d in glob.glob(f"{self.data_dir}/train_short_audio.wav/*")])
        return labels

    @property
    def n_labels(self):
        return len(self.labels)

    @property
    def training_dataset_size(self):
        return self.n_labels * self.n_samples_per_label

    def as_dict(self):
        d = dataclasses.asdict(self)
        d['labels'] = self.labels
        d['scheduler_params'] = self.scheduler_params
        d['training_dataset_size'] = self.training_dataset_size
        d['thr_dict'] = self.thr_dict
        return d

    @property
    def instantiate_model(self):
        import torch  # type: ignore
        from lark.pann import Cnn14_DecisionLevelAtt  # type: ignore
        return eval(self.model)

    @property
    def instantiate_loss(self):
        import torch  # type: ignore
        import lark  # type: ignore
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

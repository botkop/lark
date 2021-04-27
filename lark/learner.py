import dataclasses
import glob
import itertools
import math
import os
import random
from dataclasses import dataclass
from typing import List

import neptune
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as tat

from lark.Cnn14_DecisionLevelAtt import PANNsLoss


def f1(y_true: torch.Tensor, y_pred: torch.Tensor, thresh: float) -> float:
    tp = (y_pred[torch.where(y_true == 1)] >= thresh).sum()
    # tn = (y_pred[torch.where(y_true == 0)] < thresh).sum()
    fp = (y_pred[torch.where(y_true == 0)] >= thresh).sum()
    fn = (y_pred[torch.where(y_true == 1)] < thresh).sum()
    if tp + fp + fn == 0:
        f = 1.0
    else:
        f = tp / (tp + (fp + fn) / 2)
        f = f.item()
    # d = {"tp": tp.item(), "tn": tn.item(), "fp": fp.item(), "fn": fn.item(), "f1": f.item()}
    # return d
    return f


def read_random_sig(folder: str, n_frames: int) -> torch.Tensor:
    while True:
        f = random.choice(glob.glob(f"{folder}/*.wav"))
        tot_frames = ta.info(f).num_frames
        if tot_frames > n_frames:
            offset = random.randint(0, tot_frames - n_frames)
            sig, _ = ta.load(filepath=f, frame_offset=offset, num_frames=n_frames)
            if not torch.isnan(sig).any() and sig.norm(p=2).item() != 0.0:
                return sig
            else:
                print(f"skipping file {f} offset {offset}: invalid signal")


def merge_sigs(sig_a: torch.Tensor, sig_b: torch.Tensor, snr_db: int) -> torch.Tensor:
    power_a = sig_a.norm(p=2)
    power_b = sig_b.norm(p=2)
    snr = math.exp(snr_db / 10)
    scale = snr * power_b / power_a
    merged = (scale * sig_a + sig_b) / 2
    return merged


@dataclass
class Config:
    # data parameters
    site: str
    data_dir: str = 'data/birdclef-2021'
    bs: int = 32
    n_workers: int = 12
    training_dataset_size: int = bs * n_workers * 28  # 28 = number of labels
    duration: float = 5

    # augmentation
    use_noise: bool = True
    # noise_snr_dbs: List[int] = dataclasses.field(default_factory=lambda: [3, 1])
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
    window_length: int = 512
    n_mels: int = 64
    hop_length: int = 312
    f_min: int = 150
    f_max: int = 15000

    # learner parameters
    lr: float = 1e-4
    n_epochs: int = 10
    loss_fn: str = 'pann'
    optimizer: str = 'Adam'
    scheduler: str = 'OneCycleLR'

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
        return d

    @property
    def instantiate_loss(self):
        if self.loss_fn:
            return PANNsLoss

    @property
    def instantiate_optimizer(self):
        if self.optimizer == 'Adam':
            return torch.optim.Adam

    @property
    def instantiate_scheduler(self):
        if self.scheduler == 'OneCycleLR':
            return torch.optim.lr_scheduler.OneCycleLR


class ValidDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.site = cfg.site
        self.sss = glob.glob(f"{cfg.data_dir}/train_soundscapes.wav/*_{self.site}_*.wav")
        self.df_ss = pd.read_csv(f"{cfg.data_dir}/train_soundscape_labels.csv")
        self.df_ss = self.df_ss[self.df_ss['site'] == self.site].reset_index(drop=True)
        self.labels = [x for x in sorted(set(itertools.chain(*self.df_ss['birds'].str.split(' ')))) if x != 'nocall']
        self.indices = {b: self.labels.index(b) for b in self.labels}

    def __len__(self):
        return len(self.df_ss)

    def __getitem__(self, idx):
        row = self.df_ss.iloc[idx]
        fname = glob.glob(f"{self.cfg.data_dir}/train_soundscapes.wav/{row.audio_id}_{row.site}_*.wav")[0]
        sig, _ = ta.load(filepath=fname, frame_offset=self.cfg.sr * (row.seconds - self.cfg.duration),
                         num_frames=self.cfg.n_frames)

        label = torch.zeros(len(self.labels))
        if row.birds != 'nocall':
            birds = row.birds.split(' ')
            indices = [self.indices[b] for b in birds]
            label[indices] = 1.0

        return sig, label

    @property
    def loader(self):
        dl = DataLoader(self, batch_size=self.cfg.bs, shuffle=False, num_workers=self.cfg.n_workers)
        return dl


class TrainDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.labels = cfg.labels
        self.indices = {b: self.labels.index(b) for b in self.labels}
        self.size = cfg.training_dataset_size

    def __len__(self):
        return self.size

    def read_sig(self):
        b = random.choice(self.labels)
        sig = read_random_sig(f"{self.cfg.data_dir}/train_short_audio.wav/{b}", self.cfg.n_frames)
        return sig, self.indices[b]

    def assemble_sig(self):
        label = torch.zeros(len(self.labels))
        base_sig, b = self.read_sig()
        label[b] = 1.0
        if self.cfg.use_overlays:
            n_overlays = random.choices(range(self.cfg.max_overlays), weights=self.cfg.overlay_weights)[0]
            for _ in range(n_overlays):
                sig, b = self.read_sig()
                base_sig = merge_sigs(base_sig, sig, random.choice(self.cfg.overlay_snr_dbs))
                label[b] = 1.0
        if self.cfg.use_noise:
            noise = read_random_sig(self.cfg.noise_dir, self.cfg.n_frames)
            # base_sig = merge_sigs(base_sig, noise, random.choice(self.cfg.noise_snr_dbs))
            base_sig = merge_sigs(noise, base_sig, random.choice(self.cfg.noise_nsr_dbs))
        return base_sig, label

    def __getitem__(self, idx):
        return self.assemble_sig()

    @property
    def loader(self):
        dl = DataLoader(self, batch_size=self.cfg.bs, shuffle=True, num_workers=self.cfg.n_workers)
        return dl


class Sig2Spec(torch.nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.melspec = tat.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            win_length=cfg.window_length,
            hop_length=cfg.hop_length,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            pad=0,
            n_mels=cfg.n_mels,
            power=2.0,
            normalized=True, )
        self.p2db = tat.AmplitudeToDB(stype='power', top_db=80)

    @staticmethod
    def normalize(spec: torch.Tensor) -> torch.Tensor:
        spec -= spec.min()
        if spec.max() != 0:
            spec /= spec.max()
        else:
            spec = torch.clamp(spec, 0, 1)
        return spec

    def forward(self, sig: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        spec = self.melspec(sig)
        spec = self.p2db(spec)
        spec = self.normalize(spec)
        return spec


class Experiment:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def init(self, name: str, seed: int = 42):
        self.set_seed(seed)
        if self.cfg.use_neptune:
            neptune.init(project_qualified_name='botkop/lark')
            neptune.create_experiment(name, params=self.cfg.as_dict())

    def log_metric(self, mode: str = 'train', event: str = 'batch', name: str = 'loss', value: float = 0.0):
        if self.cfg.use_neptune:
            neptune.log_metric(f'{mode}_{event}_{name}', value)

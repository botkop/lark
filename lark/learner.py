import dataclasses
import glob
import itertools
import os
import random
from dataclasses import dataclass

import neptune
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from torchaudio import transforms as tat


def f1(y_true, y_pred, thresh: float) -> float:
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


@dataclass
class Config:
    # data parameters
    site: str
    data_dir: str = 'data/birdclef-2021'
    bs: int = 32
    n_workers: int = 12
    training_dataset_size: int = bs * n_workers * 10
    duration: float = 5

    # sig parameters
    sr: int = 32000
    n_frames: int = duration * sr
    n_fft: int = 512
    window_length: int = 512
    n_mels: int = 64
    hop_length: int = 312
    f_min: int = 150
    f_max: int = 15000

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

    def __getitem__(self, idx):
        b = random.choice(self.labels)
        f = random.choice(glob.glob(f"{self.cfg.data_dir}/train_short_audio.wav/{b}/*.wav"))
        n_frames = ta.info(f).num_frames
        offset = random.randint(0, n_frames - self.cfg.n_frames)
        sig, _ = ta.load(filepath=f, frame_offset=offset, num_frames=self.cfg.n_frames)
        label = torch.zeros(len(self.labels))
        label[self.indices[b]] = 1.0
        return sig, label

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
        neptune.init(project_qualified_name='botkop/lark')
        neptune.create_experiment(name, params=self.cfg.as_dict())


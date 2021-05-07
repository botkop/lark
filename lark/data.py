import glob
import math
import os
import random

import pandas as pd
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader

from lark.config import Config


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


class ValidDataset(Dataset):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.df_ss = pd.read_csv(f"{cfg.data_dir}/train_soundscape_labels.csv")
        if cfg.sites is not None:
            self.df_ss = self.df_ss[self.df_ss['site'].isin(cfg.sites)].reset_index(drop=True)
        self.labels = cfg.labels
        self.indices = {b: self.labels.index(b) for b in self.labels}
        self.n_frames = self.cfg.valid_duration * self.cfg.sr

    def __len__(self):
        return len(self.df_ss)

    def __getitem__(self, idx):
        row = self.df_ss.iloc[idx]
        fname = glob.glob(f"{self.cfg.data_dir}/train_soundscapes.wav/{row.audio_id}_{row.site}_*.wav")[0]
        sig, _ = ta.load(filepath=fname, frame_offset=self.cfg.sr * (row.seconds - self.cfg.valid_duration),
                         num_frames=self.n_frames)

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
        self.df_meta = pd.read_csv(f"{self.cfg.data_dir}/train_metadata.csv")
        self.df_meta['secondary_labels'] = self.df_meta['secondary_labels'].str.replace("[\[\]',]", '').str.split()
        self.df_meta['filename'] = self.df_meta['filename'].str.replace(".ogg", '.wav')
        self.n_frames = int(self.cfg.train_duration * self.cfg.sr)

    def __len__(self):
        return self.size

    def read_random_sig(self, f: str, tot_frames: int) -> torch.Tensor:
        offset = random.randint(0, tot_frames - self.n_frames)
        sig, _ = ta.load(filepath=f, frame_offset=offset, num_frames=self.n_frames)
        return sig

    def read_sig(self):
        b = random.choice(self.labels)
        while True:
            r = self.df_meta[self.df_meta['primary_label'] == b].sample().iloc[0]
            fname = f"{self.cfg.data_dir}/train_short_audio.wav/{b}/{r['filename']}"
            if not os.path.exists(fname):
                # print(f"{fname} does not exist")
                continue
            tot_frames = ta.info(fname).num_frames
            if tot_frames <= self.n_frames:
                # print(f"skipping file {fname}: not enough frames")
                continue
            sig = self.read_random_sig(fname, tot_frames)
            if torch.isnan(sig).any() or sig.norm(p=2).item() == 0.0:
                # print(f"skipping file {fname}: invalid signal")
                continue
            birds = r.secondary_labels
            birds.append(b)
            return sig, [self.indices[b] for b in birds if b in self.labels]

    def assemble_sig(self):
        label = torch.zeros(len(self.labels))
        base_sig, birds = self.read_sig()
        for b in birds:
            label[b] = 1.0
        if self.cfg.use_overlays:
            n_overlays = random.choices(range(self.cfg.max_overlays), weights=self.cfg.overlay_weights)[0]
            for _ in range(n_overlays):
                sig, bs = self.read_sig()
                base_sig = merge_sigs(base_sig, sig, random.choice(self.cfg.overlay_snr_dbs))
                for b in bs:
                    label[b] = 1.0
        if self.cfg.use_noise:
            noise = read_random_sig(self.cfg.noise_dir, self.n_frames)
            base_sig = merge_sigs(noise, base_sig, random.choice(self.cfg.noise_nsr_dbs))
        return base_sig, label

    def __getitem__(self, idx):
        return self.assemble_sig()

    @property
    def loader(self):
        dl = DataLoader(self, batch_size=self.cfg.bs, shuffle=True, num_workers=self.cfg.n_workers)
        return dl

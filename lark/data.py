import glob
import math
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
import colorednoise as cn

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


def merge_sigs(sig_a: torch.Tensor, sig_b: torch.Tensor, snr_db: int, ops_list: List) -> torch.Tensor:
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
        sig, _ = ta.load(filepath=fname,
                         frame_offset=self.cfg.sr * (row.seconds - self.cfg.valid_duration),
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
        self.n_frames = int(self.cfg.train_duration * self.cfg.sr)

        self.df_meta = pd.read_csv(f"{self.cfg.data_dir}/train_metadata.csv")
        self.df_meta['secondary_labels'] = self.df_meta['secondary_labels'].str.replace("[\[\]',]", '',
                                                                                        regex=True).str.split()
        self.df_meta['filename'] = self.df_meta['filename'].str.replace(".ogg", '.wav', regex=False)
        self.df_meta['filename'] = self.df_meta.apply(
            lambda r: f"{self.cfg.data_dir}/train_short_audio.wav/{r['primary_label']}/{r['filename']}", axis=1)

        self.df_meta['file_exists'] = self.df_meta.apply(lambda r: os.path.exists(r['filename']), axis=1)
        self.df_meta = self.df_meta[self.df_meta['file_exists']]

        self.df_meta['n_frames'] = self.df_meta.apply(lambda r: ta.info(r['filename']).num_frames, axis=1)
        self.df_meta = self.df_meta[self.df_meta['n_frames'] >= self.n_frames]

        self.pink_noise = torch.Tensor(cn.powerlaw_psd_gaussian(exponent=1, size=(1, self.n_frames)))

    def __len__(self):
        return self.size

    def read_random_sig(self, f: str, tot_frames: int, ops_list: List) -> torch.Tensor:
        offset = random.randint(0, tot_frames - self.n_frames)
        sig, _ = ta.load(filepath=f, frame_offset=offset, num_frames=self.n_frames)
        ops_list.append(("read_sig", f"{f},{offset}"))
        return sig

    def read_sig(self, ops_list):
        b = random.choice(self.labels)
        while True:
            r = self.df_meta[self.df_meta['primary_label'] == b].sample().iloc[0]
            sig = self.read_random_sig(r['filename'], r['n_frames'], ops_list)
            if sig.norm(p=2).item() == 0.0:
                # print(f"skipping file {fname}: invalid signal")
                continue
            if self.cfg.use_secondary_labels:
                birds = r.secondary_labels
                birds.append(b)
            else:
                birds = [b]
            return sig, [self.indices[b] for b in birds if b in self.labels]

    @classmethod
    def make_random_filter(cls, ops_list: List):
        # sinc 3k: high-pass
        # sinc -4k: low-pass
        # sinc 3k-4k: band-pass
        # sinc 4k-3k: band-reject
        filters = ['high-pass', 'low-pass', 'band-pass', 'band-reject']
        filter_type = filters[np.random.randint(len(filters))]
        if filter_type == 'high-pass':
            f = f"{np.random.randint(150, 2000)}"
        elif filter_type == 'low-pass':
            f = f"{- np.random.randint(6000, 12000)}"
        elif filter_type == 'band-pass':
            f = f"{np.random.randint(150, 3500)}-{np.random.randint(6000, 15000)}"
        else:  # filter_type  == 'band-reject':
            center = np.random.randint(1000, 12000)
            width = np.random.randint(500, 2000) // 2
            lo = center - width
            hi = center + width
            f = f"{hi}-{lo}"
        ops_list.append(("make_random_filter", f"{filter_type},{f}"))
        f = ["sinc", f]
        return f

    def apply_filter(self, sig: torch.Tensor, ops_list: List) -> torch.Tensor:
        f = self.make_random_filter(ops_list)
        t, _ = ta.sox_effects.apply_effects_tensor(sig, self.cfg.sr, [f])
        return t

    def assemble_sig(self):
        ops_list = []
        # label = torch.zeros(len(self.labels)) + 0.0025  # Label smoothing
        label = torch.zeros(len(self.labels))
        base_sig, birds = self.read_sig(ops_list)
        for b in birds:
            label[b] = 1.0
            # label[b] = 0.995
        if self.cfg.use_overlays:
            n_overlays = random.choices(range(self.cfg.max_overlays), weights=self.cfg.overlay_weights)[0]
            for _ in range(n_overlays):
                sig, bs = self.read_sig(ops_list)
                snr = random.choice(self.cfg.overlay_snr_dbs)
                base_sig = merge_sigs(base_sig, sig, snr, ops_list)
                for b in bs:
                    label[b] = 0.995
                    # label[b] = 1.0
        if self.cfg.use_recorded_noise > random.random():
            nsr = random.choice(self.cfg.noise_nsr_dbs)
            noise = read_random_sig(self.cfg.noise_dir, self.n_frames)
            ops_list.append(("apply_recorded_noise", f"{nsr}"))
            base_sig = merge_sigs(noise, base_sig, nsr, ops_list)
        elif self.cfg.use_pink_noise > random.random():
            nsr = random.choice(self.cfg.pink_noise_nsr_dbs)
            # import colorednoise as cn
            # noise = torch.Tensor(cn.powerlaw_psd_gaussian(exponent=1, size=base_sig.shape))
            ops_list.append(("apply_pink_noise", f"{nsr}"))
            # base_sig = merge_sigs(noise, base_sig, nsr, ops_list)
            base_sig = merge_sigs(self.pink_noise, base_sig, nsr, ops_list)

        if self.cfg.apply_filter > random.random():
            base_sig = self.apply_filter(base_sig, ops_list)

        return base_sig, label, ops_list

    def __getitem__(self, idx):
        sig, label, _ = self.assemble_sig()
        return sig, label

    @property
    def loader(self):
        dl = DataLoader(self, batch_size=self.cfg.bs, shuffle=False, num_workers=self.cfg.n_workers)
        return dl

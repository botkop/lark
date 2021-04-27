import glob
import itertools
import math
import os
import random
from datetime import datetime

import neptune
import numpy as np
import pandas as pd
import torch
import torchaudio as ta
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm

from lark.config import Config
from lark.ops import f1


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


class Learner:
    def __init__(self, exp_name: str, cfg: Config):
        self.cfg = cfg
        self.vdl = ValidDataset(cfg).loader
        self.tdl = TrainDataset(cfg).loader
        self.loss_fn = cfg.instantiate_loss().cuda()
        self.model = cfg.instantiate_model().load(cfg).cuda()
        self.optimizer = cfg.instantiate_optimizer(self.model.parameters())
        self.scheduler = cfg.instantiate_scheduler(self.optimizer)
        self.exp = Experiment(cfg)
        self.exp.init(f"{exp_name} {datetime.now()}")

    def tv_loop(self, dl, mode):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        tot_loss = 0
        tot_score = 0

        def inner_loop(pbar):
            nonlocal tot_loss
            nonlocal tot_score
            for x, y in pbar:
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                score = f1(y, pred, 0.5)

                pbar.set_description(f"{mode} loss: {loss:>8f} f1: {score:>8f}")
                self.exp.log_metric(mode, 'batch', 'loss', loss)
                self.exp.log_metric(mode, 'batch', 'f1', score)

                tot_loss += loss.item()
                tot_score += score
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

        with tqdm(dl, leave=False) as pbar:
            if mode == 'valid':
                with torch.no_grad():
                    inner_loop(pbar)
            else:
                inner_loop(pbar)

        n_batches = len(dl)
        tot_loss /= n_batches
        tot_score /= n_batches
        return tot_loss, tot_score

    def learn(self):
        with tqdm(range(self.cfg.n_epochs)) as pbar:
            for epoch in pbar:
                train_loss, train_score = self.tv_loop(self.tdl, 'train')
                valid_loss, valid_score = self.tv_loop(self.vdl, 'valid')
                msg = f"epoch: {epoch + 1:3d} train loss: {train_loss:>8f} train f1: {train_score:>8f} valid loss: {valid_loss:>8f} valid f1: {valid_score:>8f}"
                print(msg)
                self.exp.log_metric('train', 'epoch', 'loss', train_loss)
                self.exp.log_metric('valid', 'epoch', 'loss', valid_loss)
                self.exp.log_metric('train', 'epoch', 'f1', train_score)
                self.exp.log_metric('valid', 'epoch', 'f1', valid_score)


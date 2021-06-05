import numpy as np
import pandas as pd
import torch
from sklearn import metrics

from lark.config import Config


class CoOccurrence:
    def __init__(self, sites):
        self.cfg = Config(sites=sites)
        self.labels = self.cfg.labels
        self.indices = {b: self.labels.index(b) for b in self.labels}
        self.df_meta = pd.read_csv(f"{self.cfg.data_dir}/train_metadata.csv")
        self.df_meta = self.df_meta[self.df_meta.primary_label.isin(self.labels)]
        self.df_meta['secondary_labels'] = self.df_meta['secondary_labels'].str.replace("[\[\]',]", '',
                                                                                        regex=True).str.split()
        self.matrix = self.compute_matrix()

    def compute_matrix(self):
        occur = np.zeros((self.cfg.n_labels, self.cfg.n_labels), dtype='int')
        for primary, secondary in zip(self.df_meta.primary_label, self.df_meta.secondary_labels):
            for label in secondary:
                if label in self.labels and label != primary:
                    occur[self.indices[primary], self.indices[label]] = 1
                    occur[self.indices[label], self.indices[primary]] = 1
        return occur

    def save(self, fname: str):
        np.save(fname, self.matrix)


class PostProcessing:
    def __init__(self, occur: np.ndarray):
        self.occur = occur
        self.chunk_size = 120

    @staticmethod
    def get_thresholds(psc: torch.Tensor, thr_dict: dict, occur: np.ndarray):
        thresholds = np.ones_like(psc) * thr_dict['median']
        is_confident = np.sum(psc > thr_dict['high'], axis=0).astype(bool)
        thresholds[:, np.where(occur[is_confident])[0]] = thr_dict['corr']
        thresholds[:, is_confident] = thr_dict['low']
        return thresholds

    def get_chunk(self, ps, i):
        fr = i * self.chunk_size
        to = (i + 1) * self.chunk_size
        return ps[fr:to]

    def get_chunked_thresholds(self, ps, thr_dict):
        n_chunks = ps.shape[0] // self.chunk_size
        return np.concatenate([self.get_thresholds(self.get_chunk(ps, i), thr_dict, self.occur)
                               for i in range(n_chunks)])

    @staticmethod
    def compute_f1(ps, ys, ts, combined: bool = True):
        f1s = metrics.f1_score(ys, ps >= ts, average='micro', zero_division=1)
        if combined:
            no_call_f1s = metrics.f1_score(np.abs(ys - 1), ps < ts, average='micro', zero_division=1)
            f1s = no_call_f1s * 0.54 + f1s * 0.46
        return f1s

    def get_global_f1(self, ps, ys, td, combined=True):
        ts = self.get_chunked_thresholds(ps, td)
        return self.compute_f1(ps, ys, ts, combined)

    def get_individual_f1(self, ps, ys, td):
        n_chunks = ps.shape[0] // self.chunk_size
        scores = []
        for i in range(n_chunks):
            fr = i * self.chunk_size
            to = (i + 1) * self.chunk_size
            psc = ps[fr:to]
            ts = self.get_thresholds(psc, ps, td)
            f1s = metrics.f1_score(ys[fr:to], psc >= ts, average='micro', zero_division=1)
            scores.append(f1s)
        return scores

    def scan_thr_pars(self, ps, ys):
        if isinstance(ps, torch.Tensor):
            ps = ps.cpu().numpy()
        if isinstance(ys, torch.Tensor):
            ys = ys.cpu().numpy()
        max_f1 = -1
        max_td = {}
        step = 0.1
        for l in np.arange(0, 1, step):
            for m in np.arange(l + step, 1, step):
                for h in np.arange(m + step, 1, step):
                    for c in np.arange(0, 1, step):
                        td = {'high': h, 'median': m, 'low': l, 'corr': c}
                        fs = self.get_global_f1(ps, ys, td, combined=True)
                        if fs > max_f1:
                            max_f1 = fs
                            max_td = td
        return max_f1, max_td

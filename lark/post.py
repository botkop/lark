import numpy as np
from sklearn import metrics

from lark.config import Config
from lark.data import TrainDataset


class PostProcessing:
    def __init__(self, cfg: Config, tds: TrainDataset):
        self.cfg = cfg
        self.tds = tds
        self.df_meta = tds.df_meta[tds.df_meta.primary_label.isin(self.labels)]
        self.labels = tds.labels
        self.indices = tds.indices

        self.occur = self.co_occurrence
        self.chunk_size = 120

    @property
    def co_occurrence(self):
        occur = np.zeros((self.cfg.n_labels, self.cfg.n_labels), dtype='int')
        for primary, secondary in zip(self.df_meta.primary_label, self.df_meta.secondary_labels):
            for label in secondary:
                if label in self.labels and label != primary:
                    occur[self.indices[primary], self.indices[label]] = 1
                    occur[self.indices[label], self.indices[primary]] = 1
        return occur

    def get_chunk_thresholds(self, chunk_nr, ps, thr_dict):
        fr = chunk_nr * self.chunk_size
        to = (chunk_nr + 1) * self.chunk_size
        psc = ps[fr:to]
        thresholds = np.ones_like(psc) * thr_dict['median']
        is_confident = np.sum(psc > thr_dict['high'], axis=0).astype(bool)
        thresholds[:, is_confident] = thr_dict['low']
        thresholds[:, np.where(self.occur[is_confident])[0]] = thr_dict['corr']
        return thresholds

    def get_thresholds(self, ps, thr_dict):
        n_chunks = ps.shape[0] // self.chunk_size
        return np.concatenate([self.get_chunk_thresholds(i, ps, thr_dict) for i in range(n_chunks)])

    @staticmethod
    def compute_f1(ps, ys, ts, combined: bool = True):
        f1s = metrics.f1_score(ys, ps >= ts, average='micro', zero_division=1)
        if combined:
            no_call_f1s = metrics.f1_score(np.abs(ys - 1), ps < ts, average='micro', zero_division=1)
            f1s = no_call_f1s * 0.54 + f1s * 0.46
        return f1s

    def get_global_f1(self, ps, ys, td, combined=True):
        ts = self.get_thresholds(ps, td)
        return self.compute_f1(ps, ys, ts, combined)

    def get_individual_f1(self, ps, ys, ts):
        n_chunks = ps.shape[0] // self.chunk_size
        scores = []
        for i in range(n_chunks):
            fr = i * self.chunk_size
            to = (i + 1) * self.chunk_size
            ts = self.get_chunk_thresholds(i, ps, ts)
            f1s = metrics.f1_score(ys[fr:to], ps[fr:to] >= ts, average='micro', zero_division=1)
            scores.append(f1s)
        return scores

    def scan_thr_pars(self, ps, ys):
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






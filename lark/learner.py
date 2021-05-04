import gc
import os
import random
from datetime import datetime

import neptune
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm

from lark.config import Config
from lark.data import ValidDataset, TrainDataset


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

    def finish(self):
        if self.cfg.use_neptune:
            neptune.stop()


class Learner:
    def __init__(self, exp_name: str, cfg: Config, model: torch.nn.Module = None):
        self.cfg = cfg
        self.vdl = ValidDataset(cfg).loader
        self.tdl = TrainDataset(cfg).loader
        self.loss_fn = cfg.instantiate_loss().cuda()
        if model:
            self.model = model
        else:
            self.model = cfg.instantiate_model().load(cfg).cuda()
        self.optimizer = cfg.instantiate_optimizer(self.model.parameters())
        self.scheduler = cfg.instantiate_scheduler(self.optimizer)
        self.name = f"{exp_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.exp = Experiment(cfg)

    def epoch_loop(self, dl, mode, f1_threshold):
        from lark.ops import f1
        tl = 0
        ps = []
        ys = []
        n_batches = len(dl)
        with tqdm(dl, leave=False) as pbar:
            for x, y in pbar:
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                with torch.no_grad():
                    ps.append(pred.sigmoid())
                    ys.append(y)

                pbar.set_description(f"{mode} loss: {loss:>8f}")
                self.exp.log_metric(mode, 'batch', 'loss', loss)

                tl += loss.item()
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()

            ts = f1(torch.cat(ys), torch.cat(ps), f1_threshold)['f1']
            tl /= n_batches
            return tl, ts

    def tv_loop(self, dl, mode, f1_threshold=0.5):
        if mode == 'train':
            self.model.train()
            epoch_loss, epoch_score = self.epoch_loop(dl, mode, f1_threshold)
        else:
            self.model.eval()
            with torch.no_grad():
                epoch_loss, epoch_score = self.epoch_loop(dl, mode, f1_threshold)
        return epoch_loss, epoch_score

    def learn(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.exp.init(self.name)
        last_valid_loss = np.inf
        with tqdm(range(self.cfg.n_epochs)) as pbar:
            for epoch in pbar:
                train_loss, train_score = self.tv_loop(self.tdl, 'train')
                valid_loss, valid_score = self.tv_loop(self.vdl, 'valid')
                msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " \
                      f"epoch: {epoch + 1:3d} " \
                      f"train loss: {train_loss:>8f} train f1: {train_score:>8f} " \
                      f"valid loss: {valid_loss:>8f} valid f1: {valid_score:>8f}"
                print(msg)
                self.exp.log_metric('train', 'epoch', 'loss', train_loss)
                self.exp.log_metric('valid', 'epoch', 'loss', valid_loss)
                self.exp.log_metric('train', 'epoch', 'f1', train_score)
                self.exp.log_metric('valid', 'epoch', 'f1', valid_score)
                if valid_loss <= last_valid_loss:
                    self.save_checkpoint('best', epoch, valid_loss, valid_score)
                    last_valid_loss = valid_loss
                self.save_checkpoint('latest', epoch, valid_loss, valid_score)
        self.exp.finish()

    def evaluate(self):
        from lark.ops import f1
        self.model.eval()
        with torch.no_grad():
            ps = []
            ys = []
            for x, y in tqdm(self.vdl):
                x = x.cuda()
                y = y.cuda()
                pred = self.model(x).sigmoid()
                ps.append(pred)
                ys.append(y)
            ps = torch.cat(ps)
            ys = torch.cat(ys)
            ts = np.arange(0.0, 1.1, 0.1)
            rs = [f1(ys, ps, t) for t in ts]
            df = pd.DataFrame(rs)
            return df

    def save_checkpoint(self, kind: str, epoch: int, valid_loss: float, valid_score: float):
        fname = f"{self.cfg.checkpoint_dir}/{self.name}-{kind}.pt"
        torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'valid_score': valid_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, fname)

    def load_checkpoint(self, kind: str, name: str = None):
        if name is None:
            fname = f"{self.cfg.checkpoint_dir}/{self.name}-{kind}.pt"
        else:
            fname = f"{self.cfg.checkpoint_dir}/{name}.pt"
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        valid_loss = checkpoint['valid_loss']
        valid_score = checkpoint['valid_score']
        return dict(epoch=epoch, valid_loss=valid_loss, valid_score=valid_score)

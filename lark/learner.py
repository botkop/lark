import os
import random
from datetime import datetime

import neptune
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from lark.config import Config
from lark.data import ValidDataset, TrainDataset
from lark.post import PostProcessing


class Experiment:
    def __init__(self, cfg: Config, rank: int):
        self.cfg = cfg
        self.rank = rank
        self.use_neptune = cfg.use_neptune and rank == 0

    def set_seed(self):
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        os.environ["PYTHONHASHSEED"] = str(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    def init(self, name: str):
        self.set_seed()
        if self.use_neptune:
            neptune.init(project_qualified_name='botkop/lark')
            neptune.create_experiment(name, params=self.cfg.as_dict(), upload_stderr=False)

    def log_metric(self, mode: str = 'train', event: str = 'batch', name: str = 'loss', value: float = 0.0):
        if self.use_neptune:
            neptune.log_metric(f'{mode}_{event}_{name}', value)

    def finish(self):
        if self.use_neptune:
            neptune.stop()


class Learner:
    def __init__(self, exp_name: str, cfg: Config, rank: int, model: torch.nn.Module = None):
        self.name = f"{exp_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        print(f"{self.name}:{rank}")
        self.cfg = cfg
        self._vdl = None
        self._tdl = None
        self.loss_fn = cfg.instantiate_loss()
        if model:
            self.model = model
        else:
            self.model = cfg.instantiate_model().load(cfg).to(rank)
        self.optimizer = cfg.instantiate_optimizer(self.model.parameters())
        self.scheduler = cfg.instantiate_scheduler(self.optimizer)
        self.schedule_per_epoch = cfg.schedule_per_epoch
        self.schedule_per_batch = not cfg.schedule_per_epoch
        self.exp = Experiment(cfg, rank)
        self.rank = rank
        self.popr = PostProcessing(cfg)

    @property
    def vdl(self):
        if self._vdl is None:
            self._vdl = ValidDataset(self.cfg).loader
        return self._vdl

    @property
    def tdl(self):
        if self._tdl is None:
            self._tdl = TrainDataset(self.cfg).loader
        return self._tdl

    def epoch_loop(self, dl, mode):
        from lark.ops import f1
        tl = 0
        ps = []
        ys = []
        n_batches = len(dl)
        with tqdm(dl, desc=f"{self.rank}:{mode}", leave=False, ascii=None, position=self.rank+2) as epoch_bar:
            for x, y in epoch_bar:
                # y = y.to(self.rank)
                pred = self.model(x)
                smooth_y = y.to(self.rank)
                smooth_y[smooth_y == 0] = 0.01
                loss = self.loss_fn(pred, smooth_y)
                with torch.no_grad():
                    ps.append(pred.sigmoid())
                    ys.append(y)
                epoch_bar.set_description(f"{self.rank}:{mode} loss: {loss:>8f}")

                tl += loss.item()
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.schedule_per_batch:
                        self.scheduler.step()

        if mode == 'train' and self.schedule_per_epoch:
            self.scheduler.step()
        ts = f1(torch.cat(ys), torch.cat(ps), self.cfg.f1_threshold)['f1']
        tl /= n_batches
        xf1 = self.popr.get_f1(torch.cat(ps), torch.cat(ys), self.cfg.thr_dict)
        return tl, ts, xf1

    def tv_loop(self, dl, mode):
        if mode == 'train':
            self.model.train()
            epoch_loss, epoch_score, epoch_x_score = self.epoch_loop(dl, mode)
        else:
            self.model.eval()
            with torch.no_grad():
                epoch_loss, epoch_score, epoch_x_score = self.epoch_loop(dl, mode)
        return epoch_loss, epoch_score, epoch_x_score

    def learn(self):
        # gc.collect()
        # torch.cuda.empty_cache()
        self.exp.init(self.name)
        last_valid_loss = np.inf
        last_valid_x_score = -1
        with tqdm(range(self.cfg.n_epochs), desc="epochs", leave=False, ascii=None, position=0) as pbar:
            for epoch in pbar:
                train_loss, train_score, _ = self.tv_loop(self.tdl, 'train')
                valid_loss, valid_score, valid_x_score = self.tv_loop(self.vdl, 'valid')
                if self.rank == 0:
                    msg = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " \
                          f"epoch: {epoch + 1:3d} " \
                          f"train loss: {train_loss:>8f} train f1: {train_score:>8f} " \
                          f"valid loss: {valid_loss:>8f} valid f1: {valid_score:>8f} valid xf1: {valid_x_score:>8f}"
                    pbar.write(msg)
                    self.exp.log_metric('train', 'epoch', 'loss', train_loss)
                    self.exp.log_metric('valid', 'epoch', 'loss', valid_loss)
                    self.exp.log_metric('train', 'epoch', 'f1', train_score)
                    self.exp.log_metric('valid', 'epoch', 'f1', valid_score)
                    self.exp.log_metric('valid', 'epoch', 'xf1', valid_x_score)
                    if valid_loss <= last_valid_loss:
                        self.save_checkpoint('best_loss', epoch, valid_loss, valid_score, valid_x_score)
                        last_valid_loss = valid_loss
                    if valid_x_score >= last_valid_x_score:
                        self.save_checkpoint('best_score', epoch, valid_loss, valid_score, valid_x_score)
                        last_valid_x_score = valid_x_score
                    self.save_checkpoint('latest', epoch, valid_loss, valid_score, valid_x_score)

        self.exp.finish()

    def validation_inference(self):
        self.model.eval()
        with torch.no_grad():
            ps = []
            ys = []
            for x, y in tqdm(self.vdl):
                x = x.to(self.rank)
                y = y.to(self.rank)
                pred = self.model(x).sigmoid()
                ps.append(pred)
                ys.append(y)
            ps = torch.cat(ps)
            ys = torch.cat(ys)
        return ps, ys

    def evaluate(self):
        from lark.ops import f1
        ps, ys = self.validation_inference()
        ts = np.arange(0.0, 1.1, 0.1)
        rs = [f1(ys, ps, t) for t in ts]
        df = pd.DataFrame(rs)
        return df

    def save_checkpoint(self, kind: str, epoch: int, valid_loss: float, valid_score: float, valid_x_score: float):
        fname = f"{self.cfg.checkpoint_dir}/{self.name}-{kind}.pt"
        torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'valid_score': valid_score,
            'valid_x_score': valid_x_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, fname)

    def load_checkpoint(self, kind: str, name: str = None):
        if name is None:
            fname = f"{self.cfg.checkpoint_dir}/{self.name}-{kind}.pt"
        else:
            fname = f"{self.cfg.checkpoint_dir}/{name}.pt"
        checkpoint = torch.load(fname, map_location=torch.device(self.rank))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        valid_loss = checkpoint['valid_loss']
        valid_score = checkpoint['valid_score']
        return dict(epoch=epoch, valid_loss=valid_loss, valid_score=valid_score)

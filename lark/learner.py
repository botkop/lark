import os
import random
from datetime import datetime

import neptune
import numpy as np
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
    def __init__(self, exp_name: str, cfg: Config):
        self.cfg = cfg
        self.vdl = ValidDataset(cfg).loader
        self.tdl = TrainDataset(cfg).loader
        self.loss_fn = cfg.instantiate_loss().cuda()
        self.model = cfg.instantiate_model().load(cfg).cuda()
        self.optimizer = cfg.instantiate_optimizer(self.model.parameters())
        self.scheduler = cfg.instantiate_scheduler(self.optimizer)
        self.name = f"{exp_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.exp = Experiment(cfg)

    def work_loop(self, pbar, mode, f1_threshold):
        from lark.ops import f1
        tl = 0
        ts = 0
        for x, y in pbar:
            x = x.cuda()
            y = y.cuda()
            pred = self.model(x)
            loss = self.loss_fn(pred, y)
            score = f1(y, pred, f1_threshold)

            pbar.set_description(f"{mode} loss: {loss:>8f} f1: {score:>8f}")
            self.exp.log_metric(mode, 'batch', 'loss', loss)
            self.exp.log_metric(mode, 'batch', 'f1', score)

            tl += loss.item()
            ts += score
            if mode == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
        return tl, ts

    def tv_loop(self, dl, mode, f1_threshold=0.5):
        if mode == 'train':
            self.model.train()
        else:
            self.model.eval()

        with tqdm(dl, leave=False) as pbar:
            if mode == 'train':
                tot_loss, tot_score = self.work_loop(pbar, mode, f1_threshold)
            else:
                with torch.no_grad():
                    tot_loss, tot_score = self.work_loop(pbar, mode, f1_threshold)

        n_batches = len(dl)
        tot_loss /= n_batches
        tot_score /= n_batches
        return tot_loss, tot_score

    def learn(self):
        self.exp.init(self.name)
        last_valid_loss = np.inf
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
                if valid_loss <= last_valid_loss:
                    self.save_checkpoint(epoch, valid_loss, valid_score)
                    last_valid_loss = valid_loss
        self.exp.finish()

    def valid_loop(self, f1_threshold: float):
        oun = self.cfg.use_neptune
        self.cfg.use_neptune = False
        r = self.tv_loop(self.vdl, 'valid', f1_threshold)
        self.cfg.use_neptune = oun
        return r

    def save_checkpoint(self, epoch: int, valid_loss: float, valid_score: float):
        fname = f"{self.cfg.checkpoint_dir}/{self.name}.pt"
        torch.save({
            'epoch': epoch,
            'valid_loss': valid_loss,
            'valid_score': valid_score,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, fname)

    def load_checkpoint(self, name: str = None):
        if name is None:
            fname = f"{self.cfg.checkpoint_dir}/{self.name}.pt"
        else:
            fname = f"{self.cfg.checkpoint_dir}/{name}.pt"
        checkpoint = torch.load(fname)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        valid_loss = checkpoint['valid_loss']
        valid_score = checkpoint['valid_score']
        return dict(epoch=epoch, valid_loss=valid_loss, valid_score=valid_score)

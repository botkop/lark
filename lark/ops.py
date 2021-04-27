import torch
from torchaudio import transforms as tat

from lark.learner import Config


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


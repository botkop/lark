from typing import Dict

import torch
import torchvision
from torchaudio import transforms as tat
from torchvision import transforms

from lark.config import Config


def f1(y_true: torch.Tensor, y_pred: torch.Tensor, thresh: float) -> Dict[str, float]:
    # note f score calculated here is identical to:
    # f = sklearn.metrics.f1_score(y_true.cpu().numpy(), (y_pred >= thresh).cpu().numpy(), average='micro')

    tp = (y_pred[torch.where(y_true == 1)] >= thresh).sum()
    tn = (y_pred[torch.where(y_true == 0)] < thresh).sum()
    fp = (y_pred[torch.where(y_true == 0)] >= thresh).sum()
    fn = (y_pred[torch.where(y_true == 1)] < thresh).sum()
    if tp + fp + fn == 0:
        f = 1.0
    else:
        f = tp / (tp + (fp + fn) / 2)
        f = f.item()
    d = {"thresh": thresh, "tp": tp.item(), "tn": tn.item(), "fp": fp.item(), "fn": fn.item(), "f1": f}
    return d


def normalize(spec: torch.Tensor) -> torch.Tensor:
    spec -= spec.min()
    if spec.max() != 0:
        spec /= spec.max()
    else:
        spec = torch.clamp(spec, 0, 1)
    return spec


class Sig2Spec(torch.nn.Module):
    def __init__(self, cfg: Config, forward_as_image: bool = False):
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
            normalized=False,
        )
        self.p2db = tat.AmplitudeToDB(stype='power', top_db=80)
        self.forward_as_image = forward_as_image

    @classmethod
    def scale_minmax(cls, x, min=0.0, max=1.0):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (max - min) + min
        return x_scaled

    def forward(self, sig: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        spec = self.melspec(sig)
        spec = self.p2db(spec)
        spec = normalize(spec)
        if self.forward_as_image:
            # change channel axis from 1 to 3 (rgb)
            spec = torch.cat([spec.transpose(0, 1)]*3).transpose(0, 1)
        return spec


class MixedSig2Spec(torch.nn.Module):
    def __init__(self, cfg: Config, forward_as_image: bool = False):
        super().__init__()

        # based on https://arxiv.org/pdf/2007.11154.pdf
        window_lengths = [800, 1600, 3200]
        hop_lengths = [320, 800, 1600]

        self.melspecs = [tat.MelSpectrogram(
            sample_rate=32000,
            n_fft=window_lengths[i],
            win_length=window_lengths[i],
            hop_length=hop_lengths[i],
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            pad=0,
            n_mels=128,
            power=2.0,
            normalized=False,
        ).cuda() for i in range(3)]

        self.p2db = tat.AmplitudeToDB(stype='power', top_db=80)
        self.forward_as_image = forward_as_image
        self.tf_resize = torchvision.transforms.Resize((128, 250))
        # self.tf_resize = torchvision.transforms.Resize((224, 224))
        self.tf_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def forward(self, sig: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # note: assuming batch input
        with torch.no_grad():
            imgs = [self.tf_resize(self.p2db(ms(sig))) for ms in self.melspecs]  # 3 * [bs x 1 x H x W]
            spec = torch.cat([x.transpose(0, 1) for x in imgs]).transpose(0, 1)
            # note: tf_norm(spec) == tf_norm(normalize(spec))
            if self.forward_as_image:
                spec = normalize(spec)
            else:
                spec = self.tf_norm(spec)
            return spec

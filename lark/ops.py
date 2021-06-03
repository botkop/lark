from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
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
        self.tf_resize = torchvision.transforms.Resize((224, 224))
        self.tf_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    @classmethod
    def scale_minmax(cls, x, min=0.0, max=1.0):
        x_std = (x - x.min()) / (x.max() - x.min())
        x_scaled = x_std * (max - min) + min
        return x_scaled

    def forward(self, sig: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            spec = self.melspec(sig)
            spec = self.p2db(spec)
            spec = torch.cat([spec.transpose(0, 1)] * 3).transpose(0, 1)
            spec = self.tf_resize(spec)
            if self.forward_as_image:
                spec = normalize(spec)
            else:
                spec = self.tf_norm(spec)
        return spec


class MixedSig2Spec(torch.nn.Module):
    def __init__(self, cfg: Config, rank, forward_as_image: bool = False):
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
        ).to(rank) for i in range(3)]

        self.p2db = tat.AmplitudeToDB(stype='power', top_db=80)
        self.forward_as_image = forward_as_image
        # self.tf_resize = torchvision.transforms.Resize((128, 250))
        self.tf_resize = torchvision.transforms.Resize((224, 224))
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


class FocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.alpha = alpha
        self.gamma = gamma
        if reduction == 'mean':
            self.reduction = torch.mean
        else:
            self.reduction = torch.sum

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        probas = torch.sigmoid(preds)
        loss = torch.where(targets >= 0.5,
                           self.alpha * (1. - probas) ** self.gamma * bce_loss,
                           probas ** self.gamma * bce_loss)
        loss = self.reduction(loss)
        return loss


def sigmoid_focal_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 2,
        reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SigmoidFocalLoss(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        return sigmoid_focal_loss(preds, targets, self.alpha, self.gamma, self.reduction)


def sigmoid_focal_loss_star(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = -1,
        gamma: float = 1,
        reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class SigmoidFocalLossStar(nn.Module):

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, preds, targets):
        return sigmoid_focal_loss_star(preds, targets, self.alpha, self.gamma, self.reduction)


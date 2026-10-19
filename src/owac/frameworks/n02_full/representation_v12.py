"""RAW spatial conditioning and local self-supervision for the r8 repair.

VICReg terms follow Bardes et al., ICLR 2022, facebookresearch/vicreg main_vicreg.py
(MIT). RAW coordinate matching and patch prediction are declared adaptations;
this module does not claim to reproduce the full VICRegL training recipe.
"""

import torch
from torch import nn
from torch.nn import functional as F


def spatial_contrast(x: torch.Tensor, valid=None, *, eps=1e-4):
    """Remove a per-camera DC vector, then normalize actual spatial signal.

    Input B,N,C. No learned gain can let one pyramid level dominate another.
    Empty support and constant features stay zero; invalid NaNs are excluded.
    """
    if valid is None:
        valid = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
    valid = valid.to(device=x.device, dtype=torch.bool)
    weights = valid.unsqueeze(-1).to(x.dtype)
    clean = x.masked_fill(~valid.unsqueeze(-1), 0)
    mass = weights.sum(1, keepdim=True).clamp_min(1)
    mean = clean.sum(1, keepdim=True) / mass
    centred = (clean - mean).masked_fill(~valid.unsqueeze(-1), 0)
    # sqrt(0) has an undefined derivative even if a subsequent clamp is used.
    rms = (centred.square().sum((1, 2), keepdim=True) / (mass * x.shape[-1]) + eps**2).sqrt()
    return centred / rms


def local_vicreg(x, y):
    """Local VICReg with 25/25/1 weights and sample covariance.

    Covariance is averaged across views here; the official implementation sums
    them. Our covariance coefficient is therefore 0.5 in the official convention.
    """
    x, y = x.float(), y.float()
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("VICReg requires matching N,D tensors with N >= 2")
    invariance = F.mse_loss(x, y)
    sx = (x.var(0) + 1e-4).sqrt()
    sy = (y.var(0) + 1e-4).sqrt()
    variance = (F.relu(1 - sx).mean() + F.relu(1 - sy).mean()) / 2
    left, right = x - x.mean(0), y - y.mean(0)
    cx, cy = left.T @ left / (len(x) - 1), right.T @ right / (len(y) - 1)
    off = ~torch.eye(x.shape[1], device=x.device, dtype=torch.bool)
    covariance = (cx[off].square().sum() + cy[off].square().sum()) / (2 * x.shape[1])
    return 25 * invariance + 25 * variance + covariance, {
        "invariance": invariance,
        "variance": variance,
        "covariance": covariance,
    }


def scale_rms(x: torch.Tensor, valid=None, *, eps=1e-4):
    """Balance a scale's energy while preserving its image-dependent mean vector.

    Unlike spatial_contrast, this does not prescribe a spatial/global energy ratio.
    The learned projection can retain both scene-wide content and local geometry.
    """
    if valid is None:
        valid = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
    valid = valid.to(device=x.device, dtype=torch.bool)
    clean = x.masked_fill(~valid.unsqueeze(-1), 0)
    mass = valid.sum(1, keepdim=True).unsqueeze(-1).clamp_min(1) * x.shape[-1]
    rms = (clean.square().sum((1, 2), keepdim=True) / mass + eps**2).sqrt()
    return clean / rms


class RawPatchHead(nn.Module):
    """Training-only linear RAW patch predictor, as in MAE/FCMAE pixel heads.

    Predicts normalized 16x16x3 RAW patches from stride16 features. It is never
    installed as a policy image encoder and never produces RGB targets.
    """

    def __init__(self, channels=512, patch=16):
        super().__init__()
        self.patch = patch
        self.head = nn.Conv2d(channels, 3 * patch * patch, 1)

    def forward(self, features, raw):
        height, width = raw.shape[-2:]
        ph, pw = height // self.patch, width // self.patch
        prediction = self.head(features)[..., :ph, :pw].flatten(2).transpose(1, 2)
        target = F.unfold(raw, kernel_size=self.patch, stride=self.patch).transpose(1, 2)
        mean = target.mean(-1, keepdim=True)
        variance = target.var(-1, keepdim=True)
        target = (target - mean) / (variance + 1e-6).sqrt()
        return F.mse_loss(prediction.float(), target.float())

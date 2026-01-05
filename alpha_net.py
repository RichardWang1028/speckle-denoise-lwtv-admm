# alpha_net.py
from __future__ import annotations

from pathlib import Path
import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1) Utilities: image I/O + synthetic speckle + alpha-target
# ============================================================
def load_gray01(path: str | Path, eps: float = 1e-6) -> np.ndarray:
    img = Image.open(path)
    if img.mode != "L":
        raise ValueError(f"Expected grayscale (mode 'L'), got {img.mode} for {path}")
    u = np.asarray(img, dtype=np.float32) / 255.0
    u = np.clip(u, eps, 1.0)
    return u


def add_speckle_lognormal(
    u01: np.ndarray,
    var: float = 0.01,
    seed: int | None = 0,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Positive multiplicative speckle using lognormal noise.
    We set sigma so Var(noise) ~= var.
    """
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log(1.0 + var))
    noise = rng.lognormal(
        mean=-0.5 * sigma * sigma, sigma=sigma, size=u01.shape
    ).astype(np.float32)
    b = u01 * noise
    b = np.maximum(b, eps)  # keep positive; do NOT clip upper bound for the model
    return b.astype(np.float32)


def gradmag_np(u: np.ndarray) -> np.ndarray:
    gx = np.zeros_like(u, dtype=np.float32)
    gy = np.zeros_like(u, dtype=np.float32)
    gx[:, :-1] = u[:, 1:] - u[:, :-1]
    gy[:-1, :] = u[1:, :] - u[:-1, :]
    return np.sqrt(gx * gx + gy * gy)


def make_alpha_target_from_clean(
    u_clean01: np.ndarray,
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    k: float = 25.0,
) -> np.ndarray:
    """
    Target rule: alpha SMALL near edges, LARGE in flat regions.
      alpha = alpha_min + (alpha_max-alpha_min) * exp(-k * |∇u|)
    """
    g = gradmag_np(u_clean01)
    a = alpha_min + (alpha_max - alpha_min) * np.exp(-k * g)
    return a.astype(np.float32)


# ============================================================
# 2) CNN: small U-Net-ish network for alpha map
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AlphaUNetSmall(nn.Module):
    """
    Input:  (B,1,H,W) noisy image
    Output: (B,1,H,W) alpha logits -> map through sigmoid to [alpha_min, alpha_max]
    """

    def __init__(self, base=32):
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.bot = ConvBlock(base * 2, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bot(self.pool2(e2))
        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out(d1)


def alpha_from_logits(logits: torch.Tensor, alpha_min: float, alpha_max: float) -> torch.Tensor:
    s = torch.sigmoid(logits)
    return alpha_min + (alpha_max - alpha_min) * s


# ============================================================
# 3) Dataset for training alpha net (synthetic)
# ============================================================
class SpeckleAlphaDataset(Dataset):
    def __init__(
        self,
        clean_dir: str | Path,
        patch: int = 256,
        n_patches_per_image: int = 16,
        var_range=(0.005, 0.03),
        alpha_min=0.2,
        alpha_max=2.0,
        k=25.0,
        seed: int = 0,
    ):
        self.clean_paths = sorted(Path(clean_dir).rglob("*.png"))
        if not self.clean_paths:
            raise RuntimeError(f"No PNGs found under {clean_dir}")
        self.patch = int(patch)
        self.nppi = int(n_patches_per_image)
        self.var_range = var_range
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.k = float(k)
        self.rng = np.random.default_rng(seed)

        self.N = len(self.clean_paths) * self.nppi

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img_path = self.clean_paths[idx // self.nppi]
        u = load_gray01(img_path)

        H, W = u.shape
        p = self.patch
        if H < p or W < p:
            pad_y = max(0, p - H)
            pad_x = max(0, p - W)
            u = np.pad(u, ((0, pad_y), (0, pad_x)), mode="reflect")
            H, W = u.shape

        y0 = self.rng.integers(0, H - p + 1)
        x0 = self.rng.integers(0, W - p + 1)
        u_patch = u[y0 : y0 + p, x0 : x0 + p].astype(np.float32)

        var = float(self.rng.uniform(self.var_range[0], self.var_range[1]))
        seed = int(self.rng.integers(0, 1_000_000_000))
        b_patch = add_speckle_lognormal(u_patch, var=var, seed=seed)

        alpha_target = make_alpha_target_from_clean(
            u_patch, alpha_min=self.alpha_min, alpha_max=self.alpha_max, k=self.k
        )

        x = torch.from_numpy(b_patch[None, ...].astype(np.float32))
        y = torch.from_numpy(alpha_target[None, ...].astype(np.float32))
        return x, y


# ============================================================
# 4) Train CNN
# ============================================================
def train_alpha_net(
    clean_dir: str | Path,
    out_ckpt: str | Path,
    epochs: int = 3,
    batch: int = 8,
    lr: float = 1e-3,
    patch: int = 256,
    n_patches_per_image: int = 16,
    var_range=(0.005, 0.03),
    alpha_min: float = 0.2,
    alpha_max: float = 2.0,
    k: float = 25.0,
    num_workers: int | None = 2,
    amp: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Torch device:", device)
    if device.type == "cuda":
        print("CUDA GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    ds = SpeckleAlphaDataset(
        clean_dir=clean_dir,
        patch=patch,
        n_patches_per_image=n_patches_per_image,
        var_range=var_range,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
        k=k,
    )

    if num_workers is None:
        # conservative default (won't explode RAM)
        num_workers = 2

    dl_kwargs = dict(
        batch_size=batch,
        shuffle=True,
        num_workers=int(num_workers),
        pin_memory=(device.type == "cuda"),
    )
    if int(num_workers) > 0:
        dl_kwargs.update(dict(persistent_workers=True, prefetch_factor=2))
    dl = DataLoader(ds, **dl_kwargs)

    net = AlphaUNetSmall(base=32).to(device)
    print("model device:", next(net.parameters()).device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    device_type = "cuda" if device.type == "cuda" else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and device.type == "cuda"))

    net.train()
    for ep in range(1, epochs + 1):
        running = 0.0
        for x, y in dl:
            x = x.to(device, non_blocking=True).float()
            y = y.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type, enabled=(amp and device.type == "cuda")):
                logits = net(x)
                pred = alpha_from_logits(logits, alpha_min, alpha_max)
                loss = F.mse_loss(pred, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.detach().cpu())

        print(f"Epoch {ep}/{epochs} | avg_loss={running / max(1, len(dl)):.6f}")

    out_ckpt = Path(out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "alpha_min": float(alpha_min),
            "alpha_max": float(alpha_max),
            "k": float(k),
        },
        out_ckpt,
    )
    print("Saved checkpoint:", out_ckpt.resolve())

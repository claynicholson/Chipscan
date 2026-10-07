"""A compact UNet for inspection.

Inputs : 1 channel (image) or 2 channels (image + rendered design prior).
Outputs: 1 channel  -> metal probability (plain segmenter), or
         3 channels -> [metal, extra_metal_defect, missing_metal_defect].

With 3 outputs the network is a *detector*: the defect channels are trained on
design-aware injected defects (see dataset.py) and are used directly by the
pipeline instead of the tolerance-band XOR.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 2, base: int = 24, depth: int = 4, out_channels: int = 3):
        super().__init__()
        self.in_channels, self.base, self.depth, self.out_channels = in_channels, base, depth, out_channels
        chs = [base * (2**i) for i in range(depth + 1)]
        self.enc = nn.ModuleList()
        c = in_channels
        for ch in chs[:-1]:
            self.enc.append(_Block(c, ch))
            c = ch
        self.bottleneck = _Block(c, chs[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = chs[-1]
        for ch in reversed(chs[:-1]):
            self.up.append(nn.ConvTranspose2d(c, ch, 2, stride=2))
            self.dec.append(_Block(ch * 2, ch))
            c = ch
        self.head = nn.Conv2d(c, out_channels, 1)

    def forward(self, x):
        skips = []
        for e in self.enc:
            x = e(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips), strict=True):
            x = up(x)
            if x.shape[-2:] != s.shape[-2:]:
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, s], dim=1))
        return self.head(x)

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "base": self.base,
                "depth": self.depth,
                "out_channels": self.out_channels,
                **(extra or {}),
            },
            str(path),
        )

    @staticmethod
    def load(path: str | Path, device: torch.device | None = None) -> UNet:
        ck = torch.load(str(path), map_location=device or "cpu", weights_only=False)
        m = UNet(ck["in_channels"], ck["base"], ck["depth"], ck.get("out_channels", 1))
        m.load_state_dict(ck["state_dict"])
        return m


class UNetSegmenter:
    """Pipeline adapter. `predict` gives metal probability; with a 3-channel model
    `predict_defects` gives (extra, missing) probabilities and `predicts_defects` is True."""

    def __init__(self, model_path: str | Path, device: str = "auto", batch_size: int = 4):
        self.device = pick_device(device)
        self.model = UNet.load(model_path, self.device).to(self.device).eval()
        self.uses_design_prior = self.model.in_channels == 2
        self.predicts_defects = self.model.out_channels >= 3
        self.name = f"unet:{Path(model_path).name}"
        self.batch_size = batch_size
        self._cache: tuple[int, np.ndarray] | None = None

    @torch.no_grad()
    def _forward(self, tiles: np.ndarray, priors: np.ndarray | None) -> np.ndarray:
        key = (id(tiles), tiles.shape)
        if self._cache is not None and self._cache[0] == key:
            return self._cache[1]
        out = np.empty((len(tiles), self.model.out_channels, *tiles.shape[1:]), dtype=np.float32)
        for i in range(0, len(tiles), self.batch_size):
            x = torch.from_numpy(tiles[i : i + self.batch_size]).float()[:, None]
            if self.uses_design_prior:
                if priors is None:
                    raise ValueError("this UNet expects a design prior channel")
                p = torch.from_numpy(priors[i : i + self.batch_size].astype(np.float32))[:, None]
                x = torch.cat([x, p], dim=1)
            x = _pad_to_multiple(x, 2**self.model.depth)
            y = torch.sigmoid(self.model(x.to(self.device))).cpu().numpy()
            out[i : i + len(y)] = y[:, :, : tiles.shape[1], : tiles.shape[2]]
        self._cache = (key, out)
        return out

    def predict(self, tiles: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
        return self._forward(tiles, priors)[:, 0]

    def predict_defects(self, tiles: np.ndarray, priors: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if not self.predicts_defects:
            raise ValueError("model has no defect head")
        y = self._forward(tiles, priors)
        return y[:, 1], y[:, 2]


def _pad_to_multiple(x: torch.Tensor, m: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    ph, pw = (-h) % m, (-w) % m
    return F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x

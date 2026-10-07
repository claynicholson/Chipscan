"""Learn the user's definition of a defect from their reviews.

Candidates come from the high-recall detectors (die-to-die, learned anomaly map, faint
specks).  Each review in the viewer (real / false alarm) becomes a labelled crop; a small
CNN on two-scale crops (full resolution + 4x context) learns to rank the rest.  With a
few hundred labels this turns "what Clay calls dirt" into a probability per candidate,
and the viewer can then show the most confident ones first or ask about the most
uncertain ones (active learning).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from chipscan.io.readers import ImageSource

CROP = 64


def crop_pair(source: ImageSource, cx: float, cy: float, size: int = CROP) -> np.ndarray:
    """(2, size, size) float32: full-res crop and 4x-downsampled context around (cx, cy)."""
    h = size // 2
    x, y = int(round(cx)), int(round(cy))
    fine = source.read_region(x - h, y - h, x + h, y + h, level=0).astype(np.float32) / 255.0
    lvl = min(2, source.num_levels - 1)
    s = 2**lvl
    hh = h * s
    coarse = source.read_region(x - hh, y - hh, x + hh, y + hh, level=lvl).astype(np.float32) / 255.0
    if coarse.shape != (size, size):
        import cv2

        coarse = cv2.resize(coarse, (size, size), interpolation=cv2.INTER_AREA)
    return np.stack([fine, coarse]).astype(np.float32)


class CropNet(nn.Module):
    def __init__(self, in_ch: int = 2, width: int = 32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(w, 2 * w, 3, padding=1), nn.BatchNorm2d(2 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(2 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(4 * w, 4 * w, 3, padding=1), nn.BatchNorm2d(4 * w), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(4 * w, 1),
        )

    def forward(self, x):
        return self.net(x)[:, 0]


def _augment(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = int(rng.integers(0, 4))
    x = np.rot90(x, k, axes=(1, 2))
    if rng.uniform() < 0.5:
        x = x[:, :, ::-1]
    # appearance is the whole point of this classifier: no polarity inversion, only mild gain/offset
    g = float(rng.uniform(0.9, 1.1))
    b = float(rng.uniform(-0.04, 0.04))
    x = np.clip(x * g + b, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)


def train_reviewer(
    source: ImageSource,
    labelled: list[tuple[float, float, int]],
    epochs: int = 40,
    device: str | None = None,
    seed: int = 0,
    progress=None,
) -> tuple[CropNet, dict]:
    """labelled: [(cx_px, cy_px, label 1=real 0=false)]. Returns (model, metrics from a held-out fifth)."""
    from chipscan.segmentation.unet import pick_device

    dev = pick_device(device or "auto")
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    X = np.stack([crop_pair(source, cx, cy) for cx, cy, _ in labelled])
    y = np.array([lab for _, _, lab in labelled], dtype=np.float32)
    idx = rng.permutation(len(X))
    n_val = max(1, len(X) // 5) if len(X) >= 10 else 0
    va, tr = idx[:n_val], idx[n_val:]
    model = CropNet().to(dev)
    pos = max(1.0, float((y[tr] == 0).sum()) / max(1.0, float((y[tr] == 1).sum())))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    bs = 32
    for ep in range(epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), bs):
            j = order[i : i + bs]
            xb = torch.from_numpy(np.stack([_augment(X[k], rng) for k in j])).to(dev)
            yb = torch.from_numpy(y[j]).to(dev)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=torch.tensor(pos, device=dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        if progress:
            progress(ep + 1, epochs, float(loss))
    metrics = {"n_train": int(len(tr)), "n_val": int(len(va)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum())}
    if n_val:
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(X[va]).to(dev))).cpu().numpy()
        pred = p > 0.5
        metrics.update(
            {
                "val_acc": float((pred == (y[va] > 0.5)).mean()),
                "val_precision": float(((pred) & (y[va] > 0.5)).sum() / max(1, pred.sum())),
                "val_recall": float(((pred) & (y[va] > 0.5)).sum() / max(1, (y[va] > 0.5).sum())),
            }
        )
    return model, metrics


@torch.no_grad()
def score_candidates(model: CropNet, source: ImageSource, cands: list[dict], device: str | None = None, batch: int = 128) -> np.ndarray:
    from chipscan.segmentation.unet import pick_device

    dev = pick_device(device or "auto")
    model = model.to(dev).eval()
    out = np.zeros(len(cands), np.float32)
    for i in range(0, len(cands), batch):
        xb = np.stack([crop_pair(source, *_cxy(c)) for c in cands[i : i + batch]])
        out[i : i + len(xb)] = torch.sigmoid(model(torch.from_numpy(xb).to(dev))).cpu().numpy()
    return out


def _cxy(d: dict) -> tuple[float, float]:
    c = d.get("centroid_px")
    if c is not None:
        return float(c[0]), float(c[1])
    b = d["bbox_px"]
    return 0.5 * (b["x0"] + b["x1"]), 0.5 * (b["y0"] + b["y1"])


def save_reviewer(model: CropNet, path: str | Path, metrics: dict) -> None:
    torch.save({"state_dict": model.state_dict(), "metrics": metrics}, str(path))
    Path(path).with_suffix(".json").write_text(json.dumps(metrics, indent=2))


def load_reviewer(path: str | Path) -> CropNet:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    m = CropNet()
    m.load_state_dict(ck["state_dict"])
    return m

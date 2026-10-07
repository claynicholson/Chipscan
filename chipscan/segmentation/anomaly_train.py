"""Train the design-free anomaly detector."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from torch.utils.data import DataLoader

from chipscan.segmentation.anomaly import AnomalyTiles
from chipscan.segmentation.unet import UNet, pick_device

console = Console()


def anomaly_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 3.0) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.full_like(target, pos_weight))
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean()


@torch.no_grad()
def evaluate(model: UNet, ds: AnomalyTiles, dev: torch.device, batch_size: int = 8, thr: float = 0.5) -> dict[str, float]:
    model.eval()
    inter = union = tp = fp = fn = 0.0
    n_tiles_fa = n_clean = 0
    for x, y in DataLoader(ds, batch_size=batch_size):
        p = torch.sigmoid(model(x.to(dev))).cpu() > thr
        t = y > 0.5
        inter += float((p & t).sum())
        union += float((p | t).sum())
        tp += float((p & t).sum())
        fp += float((p & ~t).sum())
        fn += float((~p & t).sum())
        for k in range(len(x)):  # tiles without any injected anomaly: does the model stay quiet?
            if not t[k].any():
                n_clean += 1
                n_tiles_fa += int(p[k].sum() > 4)
    return {
        "iou": inter / max(union, 1.0),
        "recall": tp / max(tp + fn, 1.0),
        "precision": tp / max(tp + fp, 1.0),
        "clean_tile_fa_rate": n_tiles_fa / max(n_clean, 1),
    }


def train_anomaly(
    dataset: AnomalyTiles,
    out_path: str | Path,
    epochs: int = 12,
    batch_size: int = 16,
    lr: float = 2e-3,
    base: int = 24,
    depth: int = 4,
    device: str = "auto",
    val_dataset: AnomalyTiles | None = None,
    num_workers: int = 0,
    pos_weight: float = 3.0,
    init_from: str | Path | None = None,
) -> tuple[UNet, list[dict]]:
    dev = pick_device(device)
    model = UNet.load(init_from, dev) if init_from else UNet(1, base, depth, out_channels=1)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * max(1, len(dataset) // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.15)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, persistent_workers=num_workers > 0)
    console.print(f"[bold]training anomaly detector[/] UNet(in=1, base={model.base}, depth={model.depth}, out=1) on {dev}: {len(dataset)} tiles/epoch x {epochs} epochs from {len(dataset.scenes)} scenes")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hist: list[dict] = []
    best = -1.0
    for ep in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for x, y in dl:
            x, y = x.to(dev), y.to(dev)
            loss = anomaly_loss(model(x), y, pos_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(x)
            n += len(x)
        rec = {"epoch": ep + 1, "loss": tot / max(n, 1), "seconds": time.time() - t0}
        if val_dataset is not None:
            rec.update(evaluate(model, val_dataset, dev, batch_size))
        hist.append(rec)
        console.print(
            f"epoch {ep + 1}/{epochs} loss {rec['loss']:.4f}"
            + (f"  val iou {rec['iou']:.3f} recall {rec['recall']:.3f} precision {rec['precision']:.3f} clean-tile FA {rec['clean_tile_fa_rate']:.2%}" if val_dataset else "")
            + f"  ({rec['seconds']:.0f}s)"
        )
        score = rec.get("iou", -rec["loss"])
        if score > best:
            best = score
            model.save(out_path, {"epoch": ep + 1, "metrics": rec, "task": "anomaly"})
    Path(out_path).with_suffix(".history.json").write_text(json.dumps(hist, indent=2))
    return model, hist

"""Train the pipeline's three networks (layout, speck, trace) on real scans.

    uv run python scripts/train_stages.py --which layout speck trace --epochs 10

Scenes are any pyramids; the held-out validation scene is never trained on.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch  # noqa: F401  (load torch before pyvips/zarr users)

from chipscan.io.readers import PyramidSource
from chipscan.stages.nets import DefectTiles, LayoutTiles, train

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ["AFT5_01_A2", "AFT5_04_B2", "AHQ1_B1_reticle", "PCB_boards_batch1"]
VAL = ["AFT5_07_C1"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["layout", "speck", "trace"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--val-samples", type=int, default=256)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--init", default=None, help="fine-tune from models/<which>_<init>.pt")
    ap.add_argument("--lr", type=float, default=2e-3)
    a = ap.parse_args()
    scenes = [PyramidSource(ROOT / "data/pyramids" / f"{n}.zarr") for n in TRAIN]
    val = [PyramidSource(ROOT / "data/pyramids" / f"{n}.zarr") for n in VAL]
    for which in a.which:
        t0 = time.time()
        if which == "layout":
            ds, vs = LayoutTiles(scenes, n_samples=a.samples, seed=1), LayoutTiles(val, n_samples=a.val_samples, seed=99, augment=False)
        else:
            ds = DefectTiles(scenes, which, n_samples=a.samples, seed=2 if which == "speck" else 3)
            vs = DefectTiles(val, which, n_samples=a.val_samples, seed=98, augment=False)
        init = ROOT / "models" / f"{which}_{a.init}.pt" if a.init else None
        train(which, ds, ROOT / "models" / f"{which}_{a.version}.pt", vs, epochs=a.epochs, lr=a.lr, init_from=init, log=lambda s: print(s, flush=True))
        print(f"{which} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

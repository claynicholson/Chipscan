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
    ap.add_argument("--init-path", default=None, help="fine-tune from this checkpoint (any task with the same output size)")
    ap.add_argument("--pos-weight", type=float, default=3.0)
    ap.add_argument("--hard-negatives", default=None, help="JSON from mine_negatives.py: sample tiles around these (unlabelled)")
    ap.add_argument("--p-hard", type=float, default=0.4)
    ap.add_argument("--refs", action="store_true", help="label real specks from reference maps (data/refmaps, runs/<scan>/reference.zarr) and sample around them")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args()
    def scene(n):  # a scan whose reference map was made on its straightened copy is read from there
        c = ROOT / "runs" / n / "corrected.zarr"
        if a.refs and (ROOT / "runs" / n / "reference.zarr").exists() and not (ROOT / "data/refmaps" / f"{n}.reference.zarr").exists():
            return PyramidSource(c)
        return PyramidSource(ROOT / "data/pyramids" / f"{n}.zarr")

    scenes = [scene(n) for n in TRAIN]
    val = [PyramidSource(ROOT / "data/pyramids" / f"{n}.zarr") for n in VAL]
    for which in a.which:
        t0 = time.time()
        if which == "layout":
            ds, vs = LayoutTiles(scenes, n_samples=a.samples, seed=1), LayoutTiles(val, n_samples=a.val_samples, seed=99, augment=False)
        else:
            ds = DefectTiles(scenes, which, n_samples=a.samples, seed=2 if which == "speck" else 3)
            if a.hard_negatives:
                import json

                import numpy as np

                hn = json.loads(Path(a.hard_negatives).read_text())
                ds.hard = [np.asarray(hn.get(n, []), dtype=np.int64).reshape(-1, 2) for n in TRAIN]
                ds.p_hard = a.p_hard
                print("hard negatives:", {n: len(hn.get(n, [])) for n in TRAIN}, flush=True)
            if a.refs and which == "speck":
                import json

                import numpy as np
                import zarr

                refs, centres = [], []
                for n in TRAIN:
                    pth = next((p for p in (ROOT / "data/refmaps" / f"{n}.reference.zarr", ROOT / "runs" / n / "reference.zarr") if p.exists()), None)
                    refs.append(zarr.open_group(str(pth), mode="r")["0"] if pth else None)
                hv = ROOT / "models" / "speck_harvest.json"
                hj = json.loads(hv.read_text()) if hv.exists() else {}
                ds.refs = refs
                ds.hard = [np.asarray(hj.get(n, []), dtype=np.int64).reshape(-1, 2) for n in TRAIN]
                ds.p_hard = a.p_hard
                print("reference labels:", {n: (r is not None, len(hj.get(n, []))) for n, r in zip(TRAIN, refs)}, flush=True)
            vs = DefectTiles(val, which, n_samples=a.val_samples, seed=98, augment=False)
        init = a.init_path or (ROOT / "models" / f"{which}_{a.init}.pt" if a.init else None)
        train(which, ds, ROOT / "models" / f"{which}_{a.version}.pt", vs, epochs=a.epochs, lr=a.lr, base=a.base, batch_size=a.batch_size, init_from=init, pos_weight=a.pos_weight, log=lambda s: print(s, flush=True))
        print(f"{which} done in {(time.time() - t0) / 60:.1f} min", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()

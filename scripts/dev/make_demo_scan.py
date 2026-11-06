"""A synthetic demo scan for figures: generic CPW resonator chips (chipscan.synth, nothing from a
lab design) tiled into a small wafer, with dust laid down in a known pattern so the findings
heatmap has something to show: a diagonal smear, one dense cluster, a fibre and scattered specks.

usage: uv run python scripts/dev/make_demo_scan.py OUT.png [--grid 4] [--chip-um 2500] [--px-um 1.6]
then:  uv run chipscan add OUT.png --name DEMO_synthetic_wafer --pixel-size 1.6
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from chipscan.design import Design
from chipscan.geometry import Affine2D, BBox
from chipscan.synth import build_design


def chip_image(size_um: float, px_um: float, seed: int) -> np.ndarray:
    lib, _ = build_design(size_um, n_res=8, seed=seed)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.gds"
        lib.write_gds(str(p))
        des = Design.load(p, polarity="drawn_is_etch")
    k = 1 / px_um
    n = int(np.ceil(size_um * k))
    cov = des.render(Affine2D.from_similarity(k, 0, 0, 0), BBox(0, 0, n, n), coverage=True)
    return cov.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--grid", type=int, default=4)
    ap.add_argument("--chip-um", type=float, default=2500.0)
    ap.add_argument("--px-um", type=float, default=1.6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    chips = [chip_image(a.chip_um, a.px_um, seed=s) for s in range(3)]
    n = chips[0].shape[0]
    street = int(120 / a.px_um)
    margin = int(300 / a.px_um)
    W = a.grid * n + (a.grid - 1) * street + 2 * margin
    cov = np.zeros((W, W), np.float32)
    for gy in range(a.grid):
        for gx in range(a.grid):
            x0, y0 = margin + gx * (n + street), margin + gy * (n + street)
            c = chips[(gx + 2 * gy) % 3]
            if (gx + gy) % 2:
                c = c[:, ::-1]
            cov[y0 : y0 + n, x0 : x0 + n] = c
    # wafer: everything outside a circle is bare (darker, no metal)
    yy, xx = np.mgrid[0:W, 0:W].astype(np.float32)
    r = np.hypot(xx - W / 2, yy - W / 2)
    on = r < W * 0.5
    img = np.where(on, 0.30 + 0.38 * cov, 0.22).astype(np.float32)
    img *= 1.0 + 0.06 * (xx / W - 0.5) + 0.04 * (yy / W - 0.5)

    # dust: centres drawn from a density with a smear, a cluster and a uniform floor
    def sample(kind, m):
        out = []
        while len(out) < m:
            if kind == "smear":
                t = rng.uniform(0.15, 0.85)
                x, y = W * t, W * (0.95 - t) + rng.normal(0, W * 0.05)
            elif kind == "cluster":
                x, y = rng.normal(W * 0.27, W * 0.04), rng.normal(W * 0.30, W * 0.04)
            else:
                x, y = rng.uniform(0, W, 2)
            if 0 < x < W and 0 < y < W and np.hypot(x - W / 2, y - W / 2) < W * 0.48:
                out.append((x, y))
        return out

    truth = []
    for kind, m in (("smear", 900), ("cluster", 350), ("floor", 300)):
        for x, y in sample(kind, m):
            d_um = float(np.clip(rng.lognormal(np.log(4.0), 0.55), 1.5, 30))
            rad = max(1.0, d_um / a.px_um / 2)
            dark = rng.random() < 0.8
            val = rng.uniform(0.05, 0.14) if dark else rng.uniform(0.85, 0.95)
            ax = (rad, rad * rng.uniform(0.6, 1.0))
            cv2.ellipse(img, (int(x), int(y)), (max(1, int(round(ax[0]))), max(1, int(round(ax[1])))), float(rng.uniform(0, 180)), 0, 360, float(val), -1, cv2.LINE_AA)
            truth.append({"kind": kind, "x_px": float(x), "y_px": float(y), "size_um": d_um})
    # a fibre: a long thin curve
    t = np.linspace(0, 1, 400)
    fx = W * (0.58 + 0.18 * t)
    fy = W * (0.22 + 0.10 * np.sin(t * 3.0) + 0.05 * t)
    pts = np.stack([fx, fy], 1).astype(np.int32)
    cv2.polylines(img, [pts], False, 0.12, max(2, int(3 / a.px_um)), cv2.LINE_AA)
    truth.append({"kind": "fibre", "x_px": float(fx.mean()), "y_px": float(fy.mean())})

    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    cv2.imwrite(str(a.out), np.clip(img * 255, 0, 255).astype(np.uint8))
    a.out.with_suffix(".truth.json").write_text(json.dumps({"px_um": a.px_um, "dust": truth}))
    print(f"{a.out}: {W} x {W} px, {len(truth)} dust items")


if __name__ == "__main__":
    main()

"""Make a microscope-like image from a *real* GDS cell, tile by tile, with injected defects.

    uv run python scripts/synth_from_gds.py "data/designs/NFT1_26_1_wafer(1) - Copy.gds" \
        --cell RetA_5x40_00_AFT4_RETA_SPLIT1 --layer 5/0 --pixel-size 0.93 --rotation 1.2 --out data/nft1_chip

Writes a zarr pyramid directly (no giant PNG), a truth.json, and a pipeline config.yaml.
Defect truth classes are 'extra_metal' / 'missing_metal' (kind-level), since topology
classes depend on the surrounding geometry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
import zarr

from chipscan.design import Design
from chipscan.geometry import Affine2D, BBox
from chipscan.io.pyramid import _build_levels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("gds")
    ap.add_argument("--cell", required=True)
    ap.add_argument("--layer", default="5/0")
    ap.add_argument("--polarity", default="drawn_is_metal")
    ap.add_argument("--pixel-size", type=float, default=0.93)
    ap.add_argument("--rotation", type=float, default=1.2)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--flip", action="store_true")
    ap.add_argument("--n-defects", type=int, default=60)
    ap.add_argument("--tile", type=int, default=2048)
    ap.add_argument("--margin", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    print("loading design cell", a.cell)
    des = Design.load(a.gds, cell=a.cell, layers=[a.layer], polarity=a.polarity)
    print(f"  {des.num_polygons} polygons, bbox {des.bbox.width:.0f} x {des.bbox.height:.0f} um")
    k = a.scale / a.pixel_size
    a0 = Affine2D.from_similarity(k, a.rotation, 0, 0, flip_x=a.flip)
    bb = a0.apply_bbox(des.bbox)
    world_to_px = Affine2D.translation(-bb.x0 + a.margin, -bb.y0 + a.margin).compose(a0)
    W, H = int(np.ceil(bb.width)) + 2 * a.margin, int(np.ceil(bb.height)) + 2 * a.margin
    print(f"  image {W} x {H} px at {a.pixel_size} um/px, rotation {a.rotation} deg, scale {a.scale}, flip {a.flip}")

    zpath = a.out / "image.zarr"
    root = zarr.open_group(str(zpath), mode="w")
    lvl0 = root.create_array("0", shape=(H, W), chunks=(1024, 1024), dtype="uint8")
    root.attrs.update({"source": "synth_from_gds", "pixel_size_um": a.pixel_size, "channel": "gray", "downsample": 2, "chunk_px": 1024, "fingerprint": {"synthetic": True}})

    # defect budget spread over tiles with content
    truth: list[dict] = []
    n_tiles = (-(-H // a.tile)) * (-(-W // a.tile))
    per_tile = a.n_defects / max(n_tiles, 1)
    yy_grad = np.linspace(-0.04, 0.04, H).astype(np.float32)
    xx_grad = np.linspace(-0.04, 0.04, W).astype(np.float32)
    mosaic = 1000
    tiles_b = rng.uniform(-0.03, 0.03, (H // mosaic + 1, W // mosaic + 1)).astype(np.float32)
    done = 0
    for ty in range(0, H, a.tile):
        for tx in range(0, W, a.tile):
            th, tw = min(a.tile, H - ty), min(a.tile, W - tx)
            box = BBox(tx, ty, tx + tw, ty + th)
            cov = des.render(world_to_px, box, coverage=True, supersample=4)
            mask = (cov >= 0.5).astype(np.uint8)
            if 0.001 < mask.mean() < 0.999:
                n_here = int(rng.poisson(per_tile * 1.3))
                for _ in range(n_here):
                    want_extra = rng.uniform() < 0.5
                    cand = np.argwhere((mask == 0) if want_extra else (mask > 0))
                    if len(cand) < 50:
                        continue
                    cy, cx = cand[int(rng.integers(0, len(cand)))]
                    r = int(rng.integers(3, 8))
                    if cx < r + 2 or cy < r + 2 or cx >= tw - r - 2 or cy >= th - r - 2:
                        continue
                    blob = np.zeros_like(mask)
                    if rng.uniform() < 0.4:
                        L = int(r * rng.uniform(3, 6))
                        pts = cv2.boxPoints(((float(cx), float(cy)), (float(L), float(max(2, r // 2))), float(rng.uniform(0, 180)))).astype(np.int32)
                        cv2.fillPoly(blob, [pts], 1)
                    else:
                        cv2.circle(blob, (int(cx), int(cy)), r, 1, -1)
                    region = blob.astype(bool) & ((mask == 0) if want_extra else (mask > 0))
                    if region.sum() < 6:
                        continue
                    mask[region] = 1 if want_extra else 0
                    cov[region] = 1.0 if want_extra else 0.0
                    wx, wy = world_to_px.inverse().apply(np.array([[tx + cx + 0.5, ty + cy + 0.5]]))[0]
                    truth.append({"class": "extra_metal" if want_extra else "missing_metal", "x_um": float(wx), "y_um": float(wy), "size_um": float(2 * r * a.pixel_size), "area_px": int(region.sum())})
            img = 0.30 + 0.38 * cov
            img *= 1.0 + xx_grad[tx : tx + tw][None, :] + yy_grad[ty : ty + th][:, None]
            ys, xs = np.mgrid[ty : ty + th, tx : tx + tw]
            img += tiles_b[ys // mosaic, xs // mosaic]
            img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.8)
            img += rng.normal(0, 0.03, img.shape).astype(np.float32)
            lvl0[ty : ty + th, tx : tx + tw] = np.clip(img * 255, 0, 255).astype(np.uint8)
            done += 1
        print(f"  rendered {done}/{n_tiles} tiles, {len(truth)} defects so far", flush=True)
    _build_levels(root, 1024, 512)
    root.attrs["levels"] = len(list(root.array_keys()))

    (a.out / "truth.json").write_text(json.dumps({"world_to_px": world_to_px.to_dict(), "defects": truth, "cell": a.cell, "layer": a.layer}, indent=2))
    cfg = {
        "name": a.cell,
        "image": {"path": "image.zarr", "pixel_size_um": a.pixel_size, "channel": "gray"},
        "design": {"path": str(Path(a.gds).resolve()), "cell": a.cell, "layers": [a.layer], "polarity": a.polarity, "metal_bright": True},
        "registration": {"coarse": {"target_pixel_size_um": 8.0, "allow_flip": True, "allow_scale": True}, "fine": {"enabled": True, "level_pixel_size_um": 2.0, "tile_px": 512, "grid": 8}},
        "segmentation": {"method": "classical", "classical_method": "design_prior", "tile_px": 1024, "margin_px": 64},
        "detection": {"tolerance_px": 2, "min_area_um2": 4.0},
        "export": {"out_dir": "outputs", "overlay_max_px": 3000},
    }
    (a.out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"wrote {zpath}, truth.json ({len(truth)} defects), config.yaml")


if __name__ == "__main__":
    main()

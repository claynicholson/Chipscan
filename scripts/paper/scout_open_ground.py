"""Find findings on bare ground plane: nothing patterned anywhere in a wide window around them.

    uv run python scripts/paper/scout_open_ground.py RUN_DIR OUT_DIR [--window-um 600] [--top 24]

A window counts as structure-free when, outside the findings it contains, no pixel's local
gradient is above what the ground plane's own noise gives. Writes a contact sheet of the best
candidates (and candidates.json) for a person to check before any is used in a figure.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from chipscan.io.readers import PyramidSource


def structure_score(img: np.ndarray, masks: list[tuple[int, int, int, int]], pad: int) -> float:
    """Fraction of pixels with an edge-like gradient outside the findings (0 = flat ground)."""
    g = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.0)
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    keep = np.ones(img.shape, bool)
    for x0, y0, x1, y1 in masks:
        keep[max(0, y0 - pad) : y1 + pad, max(0, x0 - pad) : x1 + pad] = False
    m = mag[keep]
    if m.size < 100:
        return 1.0
    noise = np.median(m) * 1.4826 + 1e-6
    return float((m > 8 * noise).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--window-um", type=float, default=600.0)
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--top", type=int, default=24)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    src = PyramidSource(a.run / "corrected.zarr")
    px = src.pixel_size_um or 1.6285
    fs = json.loads((a.run / "findings.json").read_text())
    fs = fs if isinstance(fs, list) else fs["findings"]
    # far from every trace and device the layout net found, and big enough to see
    cand = [d for d in fs if d.get("d_trace_um", 0) >= 40 and d.get("d_jj_um", 0) >= 80 and 6 <= d.get("size_um", 0) <= 80]
    rng = np.random.default_rng(0)
    rng.shuffle(cand)
    cand = cand[:4000]
    s = 2**a.level
    half = int(a.window_um / px / 2)
    cent = np.array([d["centroid_px"] for d in fs])
    results = []
    for d in cand:
        cx, cy = d["centroid_px"]
        x0, y0, x1, y1 = int(cx - half), int(cy - half), int(cx + half), int(cy + half)
        if x0 < 0 or y0 < 0 or x1 >= src.shape[1] or y1 >= src.shape[0]:
            continue
        img = src.read_region(x0, y0, x1, y1, level=a.level)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        near = np.where((np.abs(cent[:, 0] - cx) < half) & (np.abs(cent[:, 1] - cy) < half))[0]
        boxes = []
        for i in near:
            b = fs[i]["bbox_px"]
            boxes.append((int((b["x0"] - x0) / s), int((b["y0"] - y0) / s), int((b["x1"] - x0) / s) + 1, int((b["y1"] - y0) / s) + 1))
        sc = structure_score(img, boxes, pad=4)
        results.append({"id": d["id"], "score": sc, "size_um": d["size_um"], "risk": d["risk"], "n_in_window": len(near), "centroid_px": [cx, cy]})
    results.sort(key=lambda r: r["score"])
    best = results[: a.top]
    (a.out / "candidates.json").write_text(json.dumps(best, indent=1))
    tiles = []
    for r in best:
        cx, cy = r["centroid_px"]
        img = src.read_region(int(cx - half), int(cy - half), int(cx + half), int(cy + half), level=a.level)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        t = cv2.resize(img, (240, 240), interpolation=cv2.INTER_AREA)
        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        cv2.putText(t, f"#{r['id']} {r['score']:.4f}", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        tiles.append(t)
    while len(tiles) % 6:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + 6]) for i in range(0, len(tiles), 6)]
    cv2.imwrite(str(a.out / "contact_sheet.png"), np.vstack(rows))
    print(f"{len(cand)} candidates, {len(results)} scored; best scores", [round(r["score"], 4) for r in best[:8]])


if __name__ == "__main__":
    main()

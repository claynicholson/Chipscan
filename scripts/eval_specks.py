"""Speck recall benchmark on a held-out scan: plant specks, run the real pipeline, count what it finds.

Specks are painted (same painters as training, tones from the scan itself) into windows of a real
scan that no model was trained on.  The image is never copied: an `InjectedSource` overlays the
painted patches on every level-0 read, so the full detection stack - speck net, anomaly model,
pad rule, reference residual against twins, text / component / lettering filters, twin
verification - runs exactly as in a scan, including twin searches across the whole scan.

Every window is run twice, clean and planted.  Reported:
  recall      planted specks overlapped by a finding, by size and contrast
  alarms      findings in the clean windows (real dirt included - the scan is not clean - so this
              is an upper bound on false alarms; compare stacks, do not read it as precision)

    uv run python scripts/eval_specks.py --run runs/AFT5_07_C1_1009_161048 --windows 200
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch  # noqa: F401  (before pyvips on macOS)
import numpy as np
import zarr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from chipscan.io.readers import PyramidSource  # noqa: E402

WIN = 768
PATCH = 96


class InjectedSource(PyramidSource):
    """A pyramid with painted patches overlaid on level-0 reads (coarse levels untouched)."""

    def __init__(self, path, patches: list[tuple[int, int, np.ndarray]] | None = None):
        super().__init__(path)
        self.patches = patches or []

    def read_region(self, x0, y0, x1, y1, level=0):
        a = super().read_region(x0, y0, x1, y1, level)
        if level or not self.patches:
            return a
        a = a.copy()
        for px, py, p in self.patches:
            h, w = p.shape
            ix0, iy0, ix1, iy1 = max(x0, px), max(y0, py), min(x1, px + w), min(y1, py + h)
            if ix0 < ix1 and iy0 < iy1:
                a[iy0 - y0 : iy1 - y0, ix0 - x0 : ix1 - x0] = p[iy0 - py : iy1 - py, ix0 - px : ix1 - px]
        return a


def plant(src: PyramidSource, n_windows: int, per_window: int, seed: int):
    from chipscan.stages.inject import Tones, paint_speck

    rng = np.random.default_rng(seed)
    H, W = src.shape
    wins, patches, truth = [], [], []
    tries = 0
    while len(wins) < n_windows and tries < 50 * n_windows:
        tries += 1
        x = int(rng.integers(1, (W - WIN) // 384)) * 384
        y = int(rng.integers(1, (H - WIN) // 384)) * 384
        if any(abs(x - a) < WIN and abs(y - b) < WIN for a, b in wins):
            continue
        img = src.read_region(x, y, x + WIN, y + WIN).astype(np.float32) / 255.0
        if img.std() < 0.02:  # off the chip
            continue
        wins.append((x, y))
        spots = []
        for _ in range(per_window):
            for _ in range(20):
                sx, sy = int(rng.integers(96, WIN - 96 - PATCH)), int(rng.integers(96, WIN - 96 - PATCH))
                if all(abs(sx - a) > 200 or abs(sy - b) > 200 for a, b in spots):
                    break
            spots.append((sx, sy))
            ctx = img[max(0, sy - 80) : sy + PATCH + 80, max(0, sx - 80) : sx + PATCH + 80]
            tones = Tones(ctx)
            base = img[sy : sy + PATCH, sx : sx + PATCH]
            for _ in range(10):
                out, vis = paint_speck(base.copy(), rng, tones)
                if vis.sum() >= 2:
                    break
            if vis.sum() < 2:
                continue
            out8 = np.clip(np.rint(out * 255), 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(vis)
            contrast = float(np.abs(out[vis] - base[vis]).mean())
            patches.append((x + sx, y + sy, out8))
            truth.append(
                {
                    "x": x + sx,
                    "y": y + sy,
                    "mask": vis,
                    "area_px": int(vis.sum()),
                    "contrast": contrast,
                    "bbox": (int(x + sx + xs.min()), int(y + sy + ys.min()), int(x + sx + xs.max() + 1), int(y + sy + ys.max() + 1)),
                }
            )
    return wins, patches, truth


# ----------------------------------------------------------------------------- maps per window
def _ref_init(path, patches):
    from chipscan.stages import reference

    reference._SRC[path] = InjectedSource(path, patches)


def _ref_tile(args):
    from chipscan.stages.reference import _tile_residual

    return _tile_residual(args)


def build_maps(run: Path, out: Path, src: InjectedSource, wins, spec, workers: int):
    from chipscan.stages.nets import TiledNet, real_dirt

    out.mkdir(parents=True, exist_ok=True)
    (out / "layout.zarr").symlink_to((run / "layout.zarr").resolve())
    H, W = src.shape
    arrs = {}
    for name in ("speck", "anomaly", "paddirt", "reference", "trace"):
        g = zarr.open_group(str(out / f"{name}.zarr"), mode="w")
        arrs[name] = g.create_array("0", shape=(H, W), chunks=(1024, 1024), dtype="uint8", fill_value=0)
    nets = {"speck": TiledNet(spec.speck_model, spec.device), "anomaly": TiledNet(spec.anomaly_model, spec.device)}
    m = 48
    for x, y in wins:
        tile = src.read_region(x - m, y - m, x + WIN + m, y + WIN + m).astype(np.float32) / 255.0
        for k, net in nets.items():
            p = net.predict(tile[None])[0]
            arrs[k][y : y + WIN, x : x + WIN] = np.rint(p[m:-m, m:-m] * 255).astype(np.uint8)
        pm = 160
        big = src.read_region(x - pm, y - pm, x + WIN + pm, y + WIN + pm).astype(np.float32) / 255.0
        dirt, _ = real_dirt(big)
        arrs["paddirt"][y : y + WIN, x : x + WIN] = (dirt[pm:-pm, pm:-pm] * 255).astype(np.uint8)
    jobs = [(str(src.path), x + dx, y + dy, 384, 32) for x, y in wins for dy in (0, 384) for dx in (0, 384)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_ref_init, initargs=(str(src.path), src.patches)) as ex:
        for x0, y0, a, _n in ex.map(_ref_tile, jobs, chunksize=4):
            arrs["reference"][y0 : y0 + 384, x0 : x0 + 384] = a


# ----------------------------------------------------------------------------- scoring
def _hits(findings, truth):
    import cv2

    hit = np.zeros(len(truth), bool)
    claimed = set()
    for i, t in enumerate(truth):
        bx0, by0, bx1, by1 = t["bbox"]
        for j, d in enumerate(findings):
            b = d["bbox_px"]
            if b["x1"] < bx0 - 2 or b["x0"] > bx1 + 2 or b["y1"] < by0 - 2 or b["y0"] > by1 + 2:
                continue
            ph, pw = t["mask"].shape
            canvas = np.zeros((ph + 8, pw + 8), np.uint8)
            polys = [np.rint(np.asarray(q) - [t["x"] - 4, t["y"] - 4]).astype(np.int32) for q in d.get("polygons_px") or [] if len(q) >= 3]
            if polys:
                cv2.fillPoly(canvas, polys, 1)
            mk = np.pad(t["mask"], 4)
            mk = cv2.dilate(mk.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            if (canvas.astype(bool) & mk).any():
                hit[i] = True
                claimed.add(j)
    return hit, claimed


def score(findings_inj, findings_clean, truth, wins):
    hit, claimed = _hits(findings_inj, truth)
    area = np.array([t["area_px"] for t in truth])
    con = np.array([t["contrast"] for t in truth])
    out = {"planted": len(truth), "recall": round(float(hit.mean()), 3)}
    bins = {}
    for lo, hi, name in ((0, 12, "tiny <12px"), (12, 60, "small 12-60px"), (60, 400, "medium 60-400px"), (400, 1e9, "large >400px")):
        s = (area >= lo) & (area < hi)
        if s.any():
            bins[name] = {"n": int(s.sum()), "recall": round(float(hit[s].mean()), 3)}
    out["by_size"] = bins
    cb = {}
    for lo, hi, name in ((0, 0.06, "faint <0.06"), (0.06, 0.15, "moderate 0.06-0.15"), (0.15, 1, "strong >0.15")):
        s = (con >= lo) & (con < hi)
        if s.any():
            cb[name] = {"n": int(s.sum()), "recall": round(float(hit[s].mean()), 3)}
    out["by_contrast"] = cb

    def in_wins(fs):
        return [d for d in fs if any(x <= d["centroid_px"][0] < x + WIN and y <= d["centroid_px"][1] < y + WIN for x, y in wins)]

    clean = in_wins(findings_clean)
    extra = [d for j, d in enumerate(in_wins(findings_inj)) if j not in claimed]
    out["alarms_clean"] = len(clean)
    out["alarms_clean_per_window"] = round(len(clean) / max(1, len(wins)), 3)
    out["by_source_clean"] = {s: sum(1 for d in clean if d.get("source") == s) for s in sorted({d.get("source") for d in clean})}
    return out, hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(ROOT / "runs" / "AFT5_07_C1_1009_161048"))
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--per-window", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--speck-model", default=None)
    ap.add_argument("--out", default=str(ROOT / "runs" / "_eval"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stacks", default="current,reference")
    ap.add_argument("--reuse", action="store_true", help="keep the maps already in --out (re-score filter changes only)")
    args = ap.parse_args()

    from chipscan.stages.flow import ScanSpec, extract_findings

    run = Path(args.run)
    meta = json.loads((run / "scan.json").read_text())
    spec = ScanSpec(**{k: v for k, v in meta["spec"].items() if k in ScanSpec.__dataclass_fields__})
    if args.speck_model:
        spec.speck_model = args.speck_model
    base = PyramidSource(run / "corrected.zarr")
    t0 = time.time()
    wins, patches, truth = plant(base, args.windows, args.per_window, args.seed)
    print(f"planted {len(truth)} specks in {len(wins)} windows", flush=True)
    out = Path(args.out)
    srcs = {"clean": InjectedSource(run / "corrected.zarr"), "planted": InjectedSource(run / "corrected.zarr", patches)}
    if not args.reuse:
        if out.exists():
            shutil.rmtree(out)
        for k, s in srcs.items():
            build_maps(run, out / k, s, wins, spec, args.workers)
            print(f"maps {k} {time.time() - t0:.0f}s", flush=True)
    report = {"run": str(run), "speck_model": spec.speck_model, "windows": len(wins), "window_px": WIN}
    for stack in args.stacks.split(","):
        sp = replace(spec, reference_threshold=0.0) if stack == "current" else spec
        fs = {k: extract_findings(out / k, sp, srcs[k])[0] for k in srcs}
        for k, f in fs.items():
            (out / f"findings_{stack}_{k}.json").write_text(json.dumps([{kk: d.get(kk) for kk in ("centroid_px", "bbox_px", "source", "score", "area_px", "verified")} for d in f]))
        r, hit = score(fs["planted"], fs["clean"], truth, wins)
        report[stack] = r
        print(stack, json.dumps(r), f"{time.time() - t0:.0f}s", flush=True)
        np.save(out / f"hits_{stack}.npy", hit)
    (out / f"report_{'_'.join(args.stacks.split(','))}.json").write_text(json.dumps(report, indent=2))
    json.dump([{k: v for k, v in t.items() if k != "mask"} for t in truth], open(out / "truth.json", "w"))


if __name__ == "__main__":
    main()

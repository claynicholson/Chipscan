"""Synthetic chip generator: a CPW-resonator style GDS plus a tilted, noisy mosaic
image with injected, ground-truthed defects.

Used by the tests and by `chipscan synth` so the pipeline can be exercised and
benchmarked without any lab data.  The design is drawn *negative* (etched gaps
in a ground plane) because that is how most superconducting-circuit layouts are
drawn; the positive path is covered by unit tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import gdstk
import numpy as np
import yaml

from chipscan.design import Design
from chipscan.geometry import Affine2D, BBox

GAP, CENTER = 10.0, 20.0  # CPW geometry, um


def _rect(x0, y0, x1, y1):
    return gdstk.rectangle((x0, y0), (x1, y1))


def _cpw_gaps(points: list[tuple[float, float]], gap: float = GAP, center: float = CENTER) -> list[gdstk.Polygon]:
    outer = gdstk.FlexPath(points, center + 2 * gap, ends="flush").to_polygons()
    inner = gdstk.FlexPath(points, center, ends="flush").to_polygons()
    return gdstk.boolean(outer, inner, "not")


def build_design(size_um: float = 3000.0, n_res: int = 6, seed: int = 0) -> tuple[gdstk.Library, dict]:
    rng = np.random.default_rng(seed)
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("SYNTH_CHIP")
    polys: list[gdstk.Polygon] = []
    s = size_um
    # chip outline frame (etched border) so the design has a clear extent
    polys += [_rect(0, 0, s, 8), _rect(0, s - 8, s, s), _rect(0, 0, 8, s), _rect(s - 8, 0, s, s)]
    # feedline across the middle with launch pads
    fy = s / 2
    polys += _cpw_gaps([(300, fy), (s - 300, fy)])
    for px in (150, s - 150):
        # pad: 200x200 centre, 50 um gap ring made of four rectangles
        polys += [_rect(px - 150, fy - 150, px + 150, fy - 100), _rect(px - 150, fy + 100, px + 150, fy + 150),
                  _rect(px - 150, fy - 100, px - 100, fy + 100), _rect(px + 100, fy - 100, px + 150, fy + 100)]
        # taper region: simple straight CPW stub to the feedline
        polys += _cpw_gaps([(px + (100 if px < s / 2 else -100), fy), (300 if px < s / 2 else s - 300, fy)])
    # hanger resonators (meanders) above and below the feedline
    resonators = []
    xs = np.linspace(500, s - 500, n_res)
    for i, x in enumerate(xs):
        up = i % 2 == 0
        sign = -1 if up else 1
        y0 = fy + sign * (GAP + CENTER / 2 + 30)
        pts = [(x, y0)]
        n_turns = 5
        pitch = 70.0
        length = 400.0 + 60.0 * (i % 3)
        y = y0
        for k in range(n_turns):
            y = y + sign * length / n_turns
            pts.append((x + (k % 2) * pitch, y))
            pts.append((x + ((k + 1) % 2) * pitch, y))
        polys += _cpw_gaps(pts)
        resonators.append({"x": float(x), "y0": float(y0), "up": up})
    # alignment crosses in the corners
    for cx, cy in ((120, 120), (s - 120, 120), (120, s - 120), (s - 120, s - 120)):
        polys += [_rect(cx - 60, cy - 6, cx + 60, cy + 6), _rect(cx - 6, cy - 60, cx + 6, cy + 60)]
    # test-structure arrays (grids of small holes), a different pitch in each block
    for bx, by, pitch in ((400, 400, 40), (s - 700, 400, 55), (400, s - 700, 48), (s - 700, s - 700, 36)):
        for i in range(6):
            for j in range(6):
                x, y = bx + i * pitch, by + j * pitch
                polys.append(_rect(x, y, x + 14, y + 14))
    # some random isolated holes to break symmetry (helps registration tests)
    for _ in range(30):
        x, y = rng.uniform(250, s - 250, 2)
        if abs(y - fy) > 120:
            w = rng.uniform(10, 30)
            polys.append(_rect(x, y, x + w, y + w * rng.uniform(0.5, 2)))
    for p in polys:
        p.layer, p.datatype = 1, 0
        cell.add(p)
    return lib, {"size_um": s, "feedline_y": fy, "resonators": resonators}


# ---------------------------------------------------------------------------
def inject_defects(mask: np.ndarray, world_to_px: Affine2D, info: dict, n: int, rng: np.random.Generator) -> list[dict]:
    """Modify the *metal* mask in place; return ground-truth records in world um."""
    h, w = mask.shape
    s, fy = info["size_um"], info["feedline_y"]
    px_um = world_to_px.scale_factor
    # direction of world +y in pixel space (handles rotation *and* mirroring)
    dxy = world_to_px.linear @ np.array([0.0, 1.0])
    across_deg = float(np.degrees(np.arctan2(dxy[1], dxy[0])))
    truth = []
    kinds = ["particle", "short", "open", "void", "protrusion"]
    for i in range(n):
        kind = kinds[i % len(kinds)]
        for _ in range(50):
            if kind in ("short", "open"):
                x = rng.uniform(400, s - 400)
                y = fy + (rng.choice([-1, 1]) * (CENTER / 2 + GAP / 2) if kind == "short" else 0.0)
            elif kind == "protrusion":
                # on the ground-plane edge of the feedline gap, bulging into the gap
                x = rng.uniform(400, s - 400)
                y = fy + rng.choice([-1, 1]) * (CENTER / 2 + GAP)
            elif kind == "particle":
                # inside a launch-pad gap ring (50 um wide) with clearance from both edges
                px_c = rng.choice([150.0, s - 150.0])
                side = rng.integers(0, 4)
                if side == 0:
                    x, y = rng.uniform(px_c - 140, px_c + 140), rng.uniform(fy - 140, fy - 110)
                elif side == 1:
                    x, y = rng.uniform(px_c - 140, px_c + 140), rng.uniform(fy + 110, fy + 140)
                elif side == 2:
                    x, y = rng.uniform(px_c - 140, px_c - 110), rng.uniform(fy - 90, fy + 90)
                else:
                    x, y = rng.uniform(px_c + 110, px_c + 140), rng.uniform(fy - 90, fy + 90)
            else:
                x, y = rng.uniform(60, s - 60, 2)
            cx, cy = world_to_px.apply(np.array([[x, y]]))[0]
            if not (10 < cx < w - 10 and 10 < cy < h - 10):
                continue
            cxi, cyi = int(round(cx)), int(round(cy))
            if kind == "particle":
                r = max(3, int(round(rng.integers(3, 6) * px_um)))
                if mask[cyi - r - 6 : cyi + r + 7, cxi - r - 6 : cxi + r + 7].any():  # clearance from metal
                    continue
                cv2.circle(mask, (cxi, cyi), r, 1, -1)
                size = 2 * r / px_um
            elif kind == "short":
                # bar bridging one gap strip (centre conductor <-> ground), ~2-4 um wide
                wpx = int(max(2, rng.integers(2, 5) * px_um))
                L = int(GAP * px_um) + 2
                _bar(mask, cx, cy, L, wpx, across_deg, 1)
                size = GAP
            elif kind == "open":
                wpx = int(max(2, rng.integers(3, 6) * px_um))
                L = int(CENTER * px_um) + 2
                _bar(mask, cx, cy, L, wpx, across_deg, 0)
                size = CENTER
            elif kind == "protrusion":
                # must protrude further than the tolerance band (+2 px) to be detectable
                r = max(4, int(round(rng.integers(4, 6) * px_um)))
                # half-disc hanging off the ground-plane edge into the gap (never reaches the centre)
                cv2.circle(mask, (cxi, cyi), r, 1, -1)
                size = 2 * r / px_um
            else:  # void in ground plane
                r = max(2, int(round(rng.integers(2, 5) * px_um)))
                if not mask[cyi, cxi] or abs(y - fy) < 60:
                    continue
                cv2.circle(mask, (cxi, cyi), r, 0, -1)
                size = 2 * r / px_um
            truth.append({"class": kind, "x_um": float(x), "y_um": float(y), "size_um": float(size)})
            break
    return truth


def _bar(mask, cx, cy, length, width, angle_deg, value):
    rect = ((float(cx), float(cy)), (float(length), float(width)), float(angle_deg))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], value)


def render_image(
    design: Design,
    rotation_deg: float,
    scale: float,
    flip: bool,
    pixel_size_um: float,
    rng: np.random.Generator,
    margin_px: int = 120,
    n_defects: int = 40,
    info: dict | None = None,
    mosaic_px: int = 700,
) -> tuple[np.ndarray, Affine2D, list[dict]]:
    k = scale / pixel_size_um
    a0 = Affine2D.from_similarity(k, rotation_deg, 0, 0, flip_x=flip)
    bb = a0.apply_bbox(design.bbox)
    a = Affine2D.translation(-bb.x0 + margin_px, -bb.y0 + margin_px).compose(a0)
    W, H = int(np.ceil(bb.width)) + 2 * margin_px, int(np.ceil(bb.height)) + 2 * margin_px
    cov = design.render(a, BBox(0, 0, W, H), coverage=True)  # fractional metal coverage per pixel
    mask = (cov >= 0.5).astype(np.uint8)
    before = mask.copy()
    truth = inject_defects(mask, a, info or {"size_um": design.bbox.width, "feedline_y": design.bbox.height / 2}, n_defects, rng) if n_defects else []
    changed = mask != before
    cov = np.where(changed, mask.astype(np.float32), cov)
    img = (0.30 + 0.38 * cov).astype(np.float32)
    # illumination gradient + mosaic tile brightness steps (stitched-microscope look)
    yy, xx = np.mgrid[0:H, 0:W]
    img *= 1.0 + 0.08 * (xx / W - 0.5) + 0.05 * (yy / H - 0.5)
    tiles_b = rng.uniform(-0.03, 0.03, (H // mosaic_px + 1, W // mosaic_px + 1))
    img += tiles_b[yy // mosaic_px, xx // mosaic_px]
    # a couple of faint scratches
    for _ in range(3):
        p0 = rng.uniform(0, [W, H]).astype(int)
        p1 = rng.uniform(0, [W, H]).astype(int)
        cv2.line(img, tuple(p0), tuple(p1), float(rng.uniform(0.35, 0.5)), 1, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    return np.clip(img * 255, 0, 255).astype(np.uint8), a, truth


def make_synthetic(
    out_dir: str | Path,
    rotation_deg: float = 3.7,
    scale: float = 1.0,
    flip: bool = False,
    pixel_size_um: float = 1.0,
    size_um: float = 3000.0,
    n_defects: int = 40,
    seed: int = 0,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lib, info = build_design(size_um, seed=seed)
    gds_path = out / "design.gds"
    lib.write_gds(str(gds_path))
    design = Design.load(gds_path, polarity="drawn_is_etch")
    img, a, truth = render_image(design, rotation_deg, scale, flip, pixel_size_um, rng, n_defects=n_defects, info=info)
    img_path = out / "image.png"
    cv2.imwrite(str(img_path), img)
    truth_path = out / "truth.json"
    truth_path.write_text(json.dumps({"world_to_px": a.to_dict(), "defects": truth, "info": info}, indent=2))
    cfg = {
        "name": "synthetic",
        "image": {"path": "image.png", "pixel_size_um": pixel_size_um, "channel": "gray"},
        "design": {"path": "design.gds", "layers": ["1/0"], "polarity": "drawn_is_etch", "metal_bright": True},
        "registration": {"coarse": {"target_pixel_size_um": 4.0, "allow_flip": True, "allow_scale": True},
                         "fine": {"enabled": True, "level_pixel_size_um": 1.0, "tile_px": 384, "grid": 6}},
        "segmentation": {"method": "classical", "classical_method": "design_prior", "tile_px": 512, "margin_px": 32},
        "detection": {"tolerance_px": 1, "min_area_um2": 3.0},
        "metrology": {"cuts": [
            {"name": "feedline_A", "p0_um": [0.35 * size_um, info["feedline_y"] - 40], "p1_um": [0.35 * size_um, info["feedline_y"] + 40], "expect_um": [GAP, CENTER, GAP]},
            {"name": "feedline_B", "p0_um": [0.65 * size_um, info["feedline_y"] - 40], "p1_um": [0.65 * size_um, info["feedline_y"] + 40], "expect_um": [GAP, CENTER, GAP]},
        ]},
        "stage": {"enabled": True, "origin_stage_um": [50000.0, 80000.0]},
        "export": {"out_dir": "outputs"},
    }
    cfg_path = out / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return {"gds": gds_path, "image": img_path, "truth": truth_path, "config": cfg_path}

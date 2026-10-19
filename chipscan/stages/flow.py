"""The inspection flow for one scan, stage by stage, with progress reporting.

    image ─► straighten + fix tiling ─► layout (GDS or layout net) ─┬─► speck UNet ────┐
                                                                    ├─► trace UNet ────┤─► risk ─► findings.json
                                                                    └─► component lines┘

Everything lands in ``runs/<name>/``:
  raw.zarr          pyramid of the uploaded image (or a link to an existing pyramid)
  corrected.zarr    straightened, seam-corrected pyramid (what every detector sees)
  layout.zarr       class map (0 background, 1 trace, 2 component, 3 text) + display levels
  speck.zarr, trace.zarr, heat.zarr   detector probability maps (uint8)
  findings.json     all findings in three groups, each weighted by risk
  scan.json         the request, stage timings, straighten report, summary
"""

from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import zarr

from chipscan.io.readers import ImageSource, PyramidSource

ROOT = Path(__file__).resolve().parents[2]
STAGES = [
    ("ingest", "Import image"),
    ("straighten", "Straighten and fix tiling"),
    ("layout", "Find traces and components"),
    ("specks", "Speck detector"),
    ("traces", "Trace-error detector"),
    ("components", "Compare components"),
    ("risk", "Weight by risk"),
    ("export", "Write results"),
]
GROUPS = ("speck", "trace", "component")


@dataclass
class ScanSpec:
    name: str
    image: str
    pixel_size_um: float | None = None
    gds: str | None = None
    gds_cell: str | None = None
    trace_layers: list[str] = field(default_factory=list)
    component_layers: list[str] = field(default_factory=list)
    straighten: bool = True
    speck_threshold: float = 0.35
    trace_threshold: float = 0.5
    component_z: float = 12.0
    report_traces: bool = False
    """Trace-error findings are off until the detector is reliable (on C1/B2 nearly all were designed corners)."""
    component_missing: bool = False  # missing-member reports need regular lines; off until validated per design
    min_area_px: int = 4
    layout_model: str = str(ROOT / "models" / "layout_v1.pt")
    speck_model: str = str(ROOT / "models" / "speck_v3.pt")
    trace_model: str = str(ROOT / "models" / "trace_v2.pt")
    reference_threshold: float = 0.06
    """Reference residual (grey levels outside every twin's tolerance envelope) that counts as dirt; <=0 disables."""
    anomaly_model: str = str(ROOT / "models" / "anomaly_v6.pt")
    """Second opinion for dirt: the general anomaly model finds things the speck net misses (hairs,
    odd particles). Its candidates go through the same text and twin filters."""
    device: str = "auto"


Progress = Callable[[str, float, str], None]


def _noop(stage: str, frac: float, msg: str) -> None:
    pass


# ----------------------------------------------------------------------------- helpers
_PRIORITY = np.array([0, 2, 3, 1], np.uint8)  # class -> display priority (component > trace > text > bg)
_UNPRI = np.array([0, 3, 1, 2], np.uint8)


def _label_levels(group: zarr.Group, chunk: int = 1024, min_px: int = 256) -> None:
    """Display pyramid for a class map: each 2x2 block keeps its highest-priority class."""
    level = 0
    while True:
        prev = group[str(level)]
        h, w = prev.shape
        if min(h, w) // 2 < min_px:
            break
        nxt = group.create_array(str(level + 1), shape=(h // 2, w // 2), chunks=(chunk, chunk), dtype="uint8")
        step = 2 * chunk
        for y in range(0, (h // 2) * 2, step):
            for x in range(0, (w // 2) * 2, step):
                b = _PRIORITY[np.asarray(prev[y : min(y + step, (h // 2) * 2), x : min(x + step, (w // 2) * 2)])]
                hh, ww = b.shape[0] // 2, b.shape[1] // 2
                m = b[: 2 * hh, : 2 * ww].reshape(hh, 2, ww, 2).max(axis=(1, 3))
                nxt[y // 2 : y // 2 + hh, x // 2 : x // 2 + ww] = _UNPRI[m]
        level += 1


def _max_levels(group: zarr.Group, chunk: int = 1024, min_px: int = 256) -> None:
    level = 0
    while True:
        prev = group[str(level)]
        h, w = prev.shape
        if min(h, w) // 2 < min_px:
            break
        nxt = group.create_array(str(level + 1), shape=(h // 2, w // 2), chunks=(chunk, chunk), dtype="uint8")
        step = 2 * chunk
        for y in range(0, (h // 2) * 2, step):
            for x in range(0, (w // 2) * 2, step):
                b = np.asarray(prev[y : min(y + step, (h // 2) * 2), x : min(x + step, (w // 2) * 2)])
                hh, ww = b.shape[0] // 2, b.shape[1] // 2
                nxt[y // 2 : y // 2 + hh, x // 2 : x // 2 + ww] = b[: 2 * hh, : 2 * ww].reshape(hh, 2, ww, 2).max(axis=(1, 3))
        level += 1


def _new_map(path: Path, shape: tuple[int, int], chunk: int = 1024) -> tuple[zarr.Group, zarr.Array]:
    g = zarr.open_group(str(path), mode="w")
    return g, g.create_array("0", shape=shape, chunks=(chunk, chunk), dtype="uint8", fill_value=0)


def _iou(a: dict, b: dict) -> float:
    ax0, ay0, ax1, ay1 = a["x0"], a["y0"], a["x1"], a["y1"]
    bx0, by0, bx1, by1 = b["x0"], b["y0"], b["x1"], b["y1"]
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / u if u > 0 else 0.0


# ----------------------------------------------------------------------------- dirt candidates
def pad_dirt_map(source: ImageSource, store, tile: int = 1024, margin: int = 160, progress=None) -> None:
    """Rule-based dirt on pads (anything dark inside a pad's interior), written as 0/255."""
    from chipscan.stages.nets import real_dirt
    from chipscan.tiling import make_tiles

    tiles = make_tiles(source.shape, tile, margin)
    for i, t in enumerate(tiles):
        x0, y0, x1, y1 = t.read.to_int()
        img = source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0
        dirt, _ = real_dirt(img)
        cx0, cy0, cx1, cy1 = t.core.to_int()
        store[cy0:cy1, cx0:cx1] = (dirt[t.core_slice_in_read] * 255).astype(np.uint8)
        if progress:
            progress(i + 1, len(tiles))


def speck_union_map(run_dir: Path, spec: ScanSpec) -> zarr.Array:
    """All dirt candidates in one map: speck net, anomaly model, pad rule (max of calibrated scores)."""
    run_dir = Path(run_dir)
    sp = zarr.open_group(str(run_dir / "speck.zarr"), mode="r")["0"]
    extra = []
    srcs = [("anomaly", 0.5), ("paddirt", 0.5)]
    if spec.reference_threshold > 0:
        srcs.append(("reference", spec.reference_threshold * 1000.0 / 255.0))
    for name, thr in srcs:
        pth = run_dir / f"{name}.zarr"
        if pth.exists():
            extra.append((zarr.open_group(str(pth), mode="r")["0"], thr))
    g, u = _new_map(run_dir / "speck_union.zarr", sp.shape)
    H, W = sp.shape
    t_sp = spec.speck_threshold
    for y in range(0, H, 2048):
        for x in range(0, W, 2048):
            a = np.asarray(sp[y : y + 2048, x : x + 2048]).astype(np.float32) / 255.0
            # rescale so each source's own threshold lands on 0.5
            out = np.where(a >= t_sp, 0.5 + 0.5 * (a - t_sp) / max(1e-6, 1 - t_sp), 0.5 * a / max(1e-6, t_sp))
            for arr, thr in extra:
                b = np.asarray(arr[y : y + 2048, x : x + 2048]).astype(np.float32) / 255.0
                out = np.maximum(out, np.where(b >= thr, 0.5 + 0.5 * (b - thr) / max(1e-6, 1 - thr), 0.5 * b / max(1e-6, thr)))
            u[y : y + 2048, x : x + 2048] = np.rint(out * 255).astype(np.uint8)
    return u


# ----------------------------------------------------------------------------- findings from maps
def extract_findings(run_dir: Path, spec: ScanSpec, source: ImageSource | None = None) -> tuple[list[dict], dict]:
    """Speck + trace findings from the stored probability maps (cheap: no network run)."""
    from chipscan.segmentation.anomaly import anomalies_from_map
    from chipscan.stages.risk import suppress_component_shapes, suppress_text, suppress_text_lines

    source = source or PyramidSource(run_dir / "corrected.zarr")
    px = spec.pixel_size_um or getattr(source, "pixel_size_um", None) or 1.0
    cls = zarr.open_group(str(run_dir / "layout.zarr"), mode="r")["0"]
    info: dict = {}
    out: list[dict] = []
    dropped: list[tuple[dict, str]] = []
    groups = [("speck", spec.speck_threshold, 6)] + ([("trace", spec.trace_threshold, 8)] if spec.report_traces else [])
    for group, thr, merge in groups:
        if group == "speck":
            amap, thr = speck_union_map(run_dir, spec), 0.5
        else:
            amap = zarr.open_group(str(run_dir / f"{group}.zarr"), mode="r")["0"]
        recs = anomalies_from_map(amap, thr, spec.min_area_px, px, merge_px=merge, solid=True)
        n0 = len(recs)
        if group == "speck":
            # where did each candidate come from? anomaly-model-only finds are the least specific
            sp_net = zarr.open_group(str(run_dir / "speck.zarr"), mode="r")["0"]
            pad = zarr.open_group(str(run_dir / "paddirt.zarr"), mode="r")["0"] if (run_dir / "paddirt.zarr").exists() else None
            ref = zarr.open_group(str(run_dir / "reference.zarr"), mode="r")["0"] if spec.reference_threshold > 0 and (run_dir / "reference.zarr").exists() else None
            for d in recs:
                b = d["bbox_px"]
                ys, xs = slice(int(b["y0"]), int(b["y1"])), slice(int(b["x0"]), int(b["x1"]))
                net = float(np.asarray(sp_net[ys, xs]).max()) / 255.0
                rule = float(np.asarray(pad[ys, xs]).max()) / 255.0 if pad is not None else 0.0
                rr = float(np.asarray(ref[ys, xs]).max()) / 1000.0 if ref is not None else 0.0
                d["source"] = "speck_net" if net >= spec.speck_threshold else ("pad_rule" if rule >= 0.5 else ("reference" if rr >= spec.reference_threshold else "anomaly_model"))
            # the scan's own border (black fill past the last camera frame / from straightening) is not dirt
            edge = [d for d in recs if _on_border(source, d)]
            recs = [d for d in recs if not any(d is e for e in edge)]
            dropped += [(d, "scan_border") for d in edge]
            # the reference residual alone speaks only for specks isolated on flat field; next to a
            # junction pad or a trace corner it is more often a design variant than dust
            lone = [d for d in recs if d["source"] != "reference" or _isolated(source, d)]
            dropped += [(d, "reference_not_isolated") for d in recs if d["source"] == "reference" and not any(d is e for e in lone)]
            recs = lone
            big = [d for d in recs if d["source"] == "anomaly_model" and d["area_px"] > 3000]
            recs = [d for d in recs if not (d["source"] == "anomaly_model" and d["area_px"] > 3000)]
            dropped += [(d, "anomaly_too_big") for d in big]
        # small specks sit right on letter strokes the layout net only partly calls text: wider margin
        if group == "speck":
            # compact blobs survive inside "text" for the speck net and the pad rule (the layout net
            # calls dark particles text); the anomaly model fires on letters, so its finds do not
            a, on_a = suppress_text([d for d in recs if d.get("source") != "anomaly_model"], cls, grow_px=10, keep_compact=True)
            b, on_b = suppress_text([d for d in recs if d.get("source") == "anomaly_model"], cls, grow_px=10, keep_compact=False)
            recs, on_text = a + b, on_a + on_b
            dropped += [(d, "on_text") for d in on_text]
            recs, is_comp = suppress_component_shapes(recs, cls)
            dropped += [(d, "component_shape") for d in is_comp]
            info["speck_is_component"] = len(is_comp)
            recs, glyphs = suppress_text_lines(recs, source)
            dropped += [(d, "lettering") for d in glyphs]
            info["speck_text_lines"] = len(glyphs)
        else:
            recs, on_text = suppress_text(recs, cls, grow_px=4)
        if recs:
            # reference check: is the candidate explained by the same spot on its twins (other
            # chips, other cells)? designed structure is; a defect is not
            from chipscan.stages.verify import verify

            recs, explained = verify(source, recs)
            dropped += [(d, "stitch_ghost" if d.get("stitch_ghost") else "explained_by_twins") for d in explained]
            info[f"{group}_explained_by_twins"] = sum(1 for d in explained if not d.get("stitch_ghost"))
            info[f"{group}_stitch_ghosts"] = sum(1 for d in explained if d.get("stitch_ghost"))
        if group == "speck" and recs:
            recs = [_grow_on_pads(source, d, px) for d in recs]
        for d in recs:
            d["group"] = group
            d["kind"] = group
            d["class"] = group
            if group == "speck" and (d.get("source") == "pad_rule" or d.get("grown_on_pad")):
                d["class"] = "probe_mark"  # dark mark inside a bond/probe pad: usually the probe needle's scrub
        info[f"{group}_raw"] = n0
        info[f"{group}_on_text"] = len(on_text)
        out += recs
    # one physical thing seen by both nets: keep the more confident
    specks = [d for d in out if d["group"] == "speck"]
    traces = [d for d in out if d["group"] == "trace"]
    drop = set()
    if specks and traces:
        tc = np.array([d["centroid_px"] for d in traces])
        for s in specks:
            dd = np.hypot(*(tc - s["centroid_px"]).T)
            reach = 30 + 0.5 * max(s["bbox_px"]["x1"] - s["bbox_px"]["x0"], s["bbox_px"]["y1"] - s["bbox_px"]["y0"])
            for j in np.nonzero(dd < reach)[0]:
                t = traces[j]
                sb, tb = s["bbox_px"], t["bbox_px"]
                inside = sb["x0"] - 3 <= t["centroid_px"][0] <= sb["x1"] + 3 and sb["y0"] - 3 <= t["centroid_px"][1] <= sb["y1"] + 3
                if inside:  # part of an object already reported as dirt (e.g. a probe mark's edge)
                    drop.add(id(t))
                elif _iou(sb, tb) > 0.3:
                    drop.add(id(t) if s["score"] >= t["score"] else id(s))
    out = [d for d in out if id(d) not in drop]
    info["dedup_dropped"] = len(drop)
    info["dropped"] = [{"centroid_px": d["centroid_px"], "bbox_px": d["bbox_px"], "reason": r, "source": d.get("source")} for d, r in dropped]
    return out, info


def _on_border(source: ImageSource, d: dict, reach: int = 12) -> bool:
    b = d["bbox_px"]
    Hs, Ws = source.shape
    if b["x0"] <= 2 or b["y0"] <= 2 or b["x1"] >= Ws - 2 or b["y1"] >= Hs - 2:
        return True
    win = source.read_region(int(b["x0"]) - reach, int(b["y0"]) - reach, int(b["x1"]) + reach, int(b["y1"]) + reach)
    return int((win <= 1).sum()) >= 20


def _isolated(source: ImageSource, d: dict, ring_in: int = 3, ring_out: int = 8) -> bool:
    """No designed edge within a few px of the finding (its 3-8 px ring is featureless)."""
    import cv2
    from scipy import ndimage as ndi

    b = d["bbox_px"]
    x0, y0, x1, y1 = int(b["x0"]) - 12, int(b["y0"]) - 12, int(b["x1"]) + 12, int(b["y1"]) + 12
    win = source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0
    own = np.zeros(win.shape, np.uint8)
    polys = [np.rint(np.asarray(q) - [x0, y0]).astype(np.int32) for q in d.get("polygons_px") or [] if len(q) >= 3]
    if polys:
        cv2.fillPoly(own, polys, 1)
    else:
        own[12:-12, 12:-12] = 1
    own = own.astype(bool)
    ring = ndi.binary_dilation(own, iterations=ring_out) & ~ndi.binary_dilation(own, iterations=ring_in)
    sm = ndi.gaussian_filter(win, 1.0)
    busy = np.hypot(ndi.sobel(sm, 0), ndi.sobel(sm, 1)) > 0.12
    return bool(ring.any()) and float(busy[ring].mean()) < 0.1


def _grow_on_pads(source: ImageSource, d: dict, px: float, pad: int = 160) -> dict:
    """A detection on a pad is usually one corner of a larger probe mark: take the whole dark mark.

    Uses the same rule that labelled real pad dirt for training (anything dark inside a pad's
    interior), restricted to the marks this detection actually touches."""
    import cv2
    from scipy import ndimage as ndi

    from chipscan.stages.nets import real_dirt

    b = d["bbox_px"]
    x0, y0 = int(b["x0"]) - pad, int(b["y0"]) - pad
    x1, y1 = int(b["x1"]) + pad, int(b["y1"]) + pad
    win = source.read_region(x0, y0, x1, y1).astype(np.float32) / 255.0
    dirt, _ = real_dirt(win)
    if not dirt.any():
        return d
    own = np.zeros(win.shape, np.uint8)
    polys = [np.rint(np.asarray(q) - [x0, y0]).astype(np.int32) for q in d.get("polygons_px") or [] if len(q) >= 3]
    if polys:
        cv2.fillPoly(own, polys, 1)
    own = own.astype(bool)
    lab, n = ndi.label(dirt)
    touch = np.unique(lab[ndi.binary_dilation(own, iterations=2) & (lab > 0)])
    if not len(touch):
        return d
    grown = own | np.isin(lab, touch)
    grown = ndi.binary_fill_holes(ndi.binary_closing(grown, iterations=3))
    if grown.sum() <= own.sum() * 1.05 or grown.sum() > max(4 * own.sum(), 400) or grown.sum() > 6000:
        return d  # nothing to add, or the rule ran away (a dim pad read as one big mark)
    cs, _ = cv2.findContours(grown.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ys, xs = np.nonzero(grown)
    bb = {"x0": float(xs.min() + x0), "y0": float(ys.min() + y0), "x1": float(xs.max() + 1 + x0), "y1": float(ys.max() + 1 + y0)}
    area = int(grown.sum())
    cx, cy = float(xs.mean() + x0), float(ys.mean() + y0)
    d.update(
        {
            "polygons_px": [(c.reshape(-1, 2).astype(float) + [x0 + 0.5, y0 + 0.5]).round(2).tolist() for c in cs if len(c) >= 3],
            "bbox_px": bb,
            "area_px": area,
            "area_um2": area * px * px,
            "centroid_px": [cx, cy],
            "centroid_um": [cx * px, cy * px],
            "bbox_um": {k: v * px for k, v in bb.items()},
            "grown_on_pad": True,
        }
    )
    return d


# ----------------------------------------------------------------------------- the flow
def run_scan(spec: ScanSpec, runs_root: str | Path = ROOT / "runs", progress: Progress = _noop) -> Path:
    from chipscan.io.pyramid import build_pyramid
    from chipscan.stages.components import component_errors, find_instances
    from chipscan.stages.nets import TiledNet
    from chipscan.stages.risk import assess, suppress_text
    from chipscan.stages.straighten import straighten

    run_dir = Path(runs_root) / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"spec": asdict(spec), "stages": {}, "started": time.time(), "status": "running"}
    meta_p = run_dir / "scan.json"

    def save_meta():
        meta_p.write_text(json.dumps(meta, indent=2, default=str))

    def stage(key: str):
        meta["stages"][key] = {"t0": time.time()}
        meta["current"] = key
        save_meta()
        progress(key, 0.0, dict(STAGES)[key])

    def done(key: str, **extra):
        s = meta["stages"][key]
        s["seconds"] = round(time.time() - s["t0"], 1)
        s.update(extra)
        save_meta()
        progress(key, 1.0, dict(STAGES)[key])

    def sub(key: str):
        return lambda i, n, key=key: progress(key, i / max(n, 1), dict(STAGES)[key])

    try:
        # 1. ingest ------------------------------------------------------------
        stage("ingest")
        img = Path(spec.image)
        if img.suffix == ".zarr" or (img.is_dir() and (img / "zarr.json").exists()):
            raw = PyramidSource(img)
        else:
            raw = build_pyramid(img, run_dir / "raw.zarr", spec.pixel_size_um)
        px = spec.pixel_size_um or raw.pixel_size_um or 1.0
        spec.pixel_size_um = px
        meta["spec"]["pixel_size_um"] = px
        H, W = raw.shape
        done("ingest", width=W, height=H, pixel_size_um=px)

        # 2. straighten + tiling --------------------------------------------------
        stage("straighten")
        if spec.straighten:
            src, rep = straighten(raw, run_dir / "corrected.zarr", progress=sub("straighten"))
            done("straighten", angle_deg=rep.angle_deg, seams=rep.tiles.get("found"), frame_px=rep.tiles.get("period_px"), matrix_out_to_in=rep.matrix_out_to_in)
        else:
            src, rep = straighten(raw, run_dir / "corrected.zarr", angle_deg=0.0, fix_seams=False, progress=sub("straighten"))
            done("straighten", angle_deg=0.0, seams=False)
        src.pixel_size_um = px

        # 3. layout -----------------------------------------------------------------
        stage("layout")
        lg, cls = _new_map(run_dir / "layout.zarr", src.shape)
        comp = lg.create_array("component_prob", shape=src.shape, chunks=(1024, 1024), dtype="uint8", fill_value=0)
        TiledNet(spec.layout_model, spec.device).run(src, (cls, comp), progress=sub("layout"))
        design_info = {}
        if spec.gds:
            design_info = _apply_design(spec, src, cls, px)
        _label_levels(lg)
        done("layout", **design_info)

        # 4/5. detectors ------------------------------------------------------------
        for key, model, group in (("specks", spec.speck_model, "speck"), ("traces", spec.trace_model, "trace")):
            stage(key)
            g, arr = _new_map(run_dir / f"{group}.zarr", src.shape)
            TiledNet(model, spec.device).run(src, arr, progress=(lambda i, n, key=key: progress(key, 0.5 * i / max(n, 1), dict(STAGES)[key])) if key == "specks" else sub(key))
            if key == "specks":
                if spec.anomaly_model and Path(spec.anomaly_model).exists():
                    ga, aa = _new_map(run_dir / "anomaly.zarr", src.shape)
                    TiledNet(spec.anomaly_model, spec.device).run(src, aa, progress=lambda i, n, key=key: progress(key, 0.5 + 0.4 * i / max(n, 1), dict(STAGES)[key]))
                gp, pa = _new_map(run_dir / "paddirt.zarr", src.shape)
                pad_dirt_map(src, pa, progress=lambda i, n, key=key: progress(key, 0.9 + 0.1 * i / max(n, 1), dict(STAGES)[key]))
            done(key)
        hg, heat = _new_map(run_dir / "heat.zarr", src.shape)
        sp, tr = zarr.open_group(str(run_dir / "speck.zarr"))["0"], zarr.open_group(str(run_dir / "trace.zarr"))["0"]
        for y in range(0, H, 2048):
            for x in range(0, W, 2048):
                heat[y : y + 2048, x : x + 2048] = np.maximum(np.asarray(sp[y : y + 2048, x : x + 2048]), np.asarray(tr[y : y + 2048, x : x + 2048]))
        _max_levels(hg)

        # 6. components --------------------------------------------------------------
        stage("components")
        insts = find_instances(cls)
        comp_recs, csum = component_errors(src, insts, px, z_min=spec.component_z, progress=sub("components"), class_map=cls, report_missing=spec.component_missing)
        comp_recs, csum["twin_dropped"] = _verify_components(src, comp_recs)
        comp_recs, on_text = suppress_text(comp_recs, cls)
        csum["on_text_dropped"] = len(on_text)
        done("components", **csum)

        # 7. risk ---------------------------------------------------------------------
        stage("risk")
        st_recs, info = extract_findings(run_dir, spec, src)
        (run_dir / "dropped.json").write_text(json.dumps(info.pop("dropped", [])))
        findings = st_recs + comp_recs
        assess(findings, cls, px)
        findings.sort(key=lambda d: -d["risk"])
        for i, d in enumerate(findings):
            d["id"] = i
        done("risk", **info)

        # 8. export ---------------------------------------------------------------------
        stage("export")
        summary = _summary(findings)
        (run_dir / "findings.json").write_text(json.dumps({"summary": summary, "findings": findings}))
        _write_csv(run_dir / "findings.csv", findings)
        meta["summary"] = summary
        meta["status"] = "done"
        meta["finished"] = time.time()
        done("export")
    except Exception as e:
        meta["status"] = "failed"
        meta["error"] = f"{type(e).__name__}: {e}"
        meta["traceback"] = traceback.format_exc()
        save_meta()
        raise
    return run_dir


def _verify_components(source: ImageSource, recs: list[dict]) -> tuple[list[dict], int]:
    from chipscan.stages.verify import verify

    kept, explained = verify(source, recs)
    for d in explained:
        if d.get("stitch_ghost"):
            print(f"component finding at {d['centroid_px']} is a stitching ghost: {d['stitch_ghost']}", flush=True)
    return kept, len(explained)


def twin_check_components(source: ImageSource, recs: list[dict]) -> tuple[list[dict], int]:
    """Keep a component finding only if the same spot on the identical neighbouring chips differs.

    A position that is empty on every chip is a gap in the design, not a missing junction; a
    feature that every chip shares is design, not a defect."""
    from chipscan.segmentation.repeats import estimate_pitches, suppress_repeats

    if not recs:
        return recs, 0
    pitches = estimate_pitches(source)
    kept = []
    for d in recs:
        b = d.get("component_bbox_px") or d["bbox_px"]
        half = int(max(b["x1"] - b["x0"], b["y1"] - b["y0"]) / 2 + 8)
        probe = dict(d)
        probe["centroid_px"] = [(b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2]
        k, _ = suppress_repeats(source, [probe], pitches, half_px=half, search_px=40, match_ncc=0.9 if d["class"] != "component_missing" else 0.8)
        if k:
            d["repeat_ncc"] = k[0].get("repeat_ncc")
            kept.append(d)
    return kept, len(recs) - len(kept)


def reprocess(run_dir: Path) -> dict:
    """Redo everything after the networks (components, extraction, risk, export) from stored maps."""
    from chipscan.stages.components import component_errors, find_instances
    from chipscan.stages.risk import assess, suppress_text

    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "scan.json").read_text())
    spec = ScanSpec(**{k: v for k, v in meta["spec"].items() if k in ScanSpec.__dataclass_fields__})
    src = PyramidSource(run_dir / "corrected.zarr")
    px = spec.pixel_size_um or 1.0
    cls = zarr.open_group(str(run_dir / "layout.zarr"), mode="r")["0"]
    insts = find_instances(cls)
    comp_recs, csum = component_errors(src, insts, px, z_min=spec.component_z, class_map=cls, report_missing=spec.component_missing)
    comp_recs, csum["twin_dropped"] = _verify_components(src, comp_recs)
    comp_recs, on_text = suppress_text(comp_recs, cls)
    csum["on_text_dropped"] = len(on_text)
    st_recs, info = extract_findings(run_dir, spec, src)
    (run_dir / "dropped.json").write_text(json.dumps(info.pop("dropped", [])))
    findings = st_recs + comp_recs
    assess(findings, cls, px)
    findings.sort(key=lambda d: -d["risk"])
    for i, d in enumerate(findings):
        d["id"] = i
    summary = _summary(findings)
    (run_dir / "findings.json").write_text(json.dumps({"summary": summary, "findings": findings}))
    _write_csv(run_dir / "findings.csv", findings)
    meta["summary"] = summary
    meta["stages"].setdefault("components", {}).update(csum)
    meta["stages"].setdefault("risk", {}).update(info)
    (run_dir / "scan.json").write_text(json.dumps(meta, indent=2, default=str))
    return {"summary": summary, "components": csum, "extract": info}


def refresh_findings(run_dir: Path, spec: ScanSpec) -> list[dict]:
    """Re-threshold speck/trace maps (sensitivity change) and re-weight; components are kept."""
    from chipscan.stages.risk import assess

    run_dir = Path(run_dir)
    old = json.loads((run_dir / "findings.json").read_text())["findings"]
    comps = [d for d in old if d.get("group") == "component"]
    src = PyramidSource(run_dir / "corrected.zarr")
    recs, _ = extract_findings(run_dir, spec, src)
    cls = zarr.open_group(str(run_dir / "layout.zarr"), mode="r")["0"]
    assess(recs, cls, spec.pixel_size_um)
    findings = sorted(recs + comps, key=lambda d: -d["risk"])
    for i, d in enumerate(findings):
        d["id"] = i
    (run_dir / "findings.json").write_text(json.dumps({"summary": _summary(findings), "findings": findings}))
    _write_csv(run_dir / "findings.csv", findings)
    meta = json.loads((run_dir / "scan.json").read_text())
    meta["spec"].update({k: getattr(spec, k) for k in ("speck_threshold", "trace_threshold", "min_area_px")})
    meta["summary"] = _summary(findings)
    (run_dir / "scan.json").write_text(json.dumps(meta, indent=2, default=str))
    return findings


def _summary(findings: list[dict]) -> dict:
    s: dict = {"n": len(findings)}
    for g in GROUPS:
        fs = [d for d in findings if d.get("group") == g]
        s[g] = {"n": len(fs), **{lvl: sum(1 for d in fs if d.get("risk_level") == lvl) for lvl in ("critical", "high", "medium", "low")}}
    return s


def _write_csv(path: Path, findings: list[dict]) -> None:
    cols = ["id", "group", "class", "risk", "risk_level", "area_um2", "size_um", "d_jj_um", "d_trace_um", "score"]
    lines = [",".join(cols + ["x_px", "y_px", "x_um", "y_um"])]
    for d in findings:
        row = [str(d.get(c, "")) if not isinstance(d.get(c), float) else f"{d[c]:.3f}" for c in cols]
        cx, cy = d["centroid_px"]
        cu = d.get("centroid_um") or [cx, cy]
        lines.append(",".join(row + [f"{cx:.1f}", f"{cy:.1f}", f"{cu[0]:.2f}", f"{cu[1]:.2f}"]))
    path.write_text("\n".join(lines) + "\n")


def _apply_design(spec: ScanSpec, src: ImageSource, cls: zarr.Array, px: float) -> dict:
    """When a GDS is given: register it and paint its traces/components into the class map.

    The layout net's map is kept where the design is silent (a design revision that does not
    match the scan everywhere still helps where it does)."""
    import cv2

    from chipscan.config import CoarseRegistrationConfig, FineRegistrationConfig
    from chipscan.design import Design
    from chipscan.geometry import BBox
    from chipscan.registration import coarse_register, fine_register
    from chipscan.tiling import make_tiles

    layers = list(dict.fromkeys(spec.trace_layers + spec.component_layers)) or None
    des_all = Design.load(spec.gds, spec.gds_cell, layers)
    co = coarse_register(src, des_all, CoarseRegistrationConfig(), px)
    fine = fine_register(src, des_all, co.affine, FineRegistrationConfig(), co.pixel_size_um)
    warp = fine.warp
    info = {"design_coarse_score": round(float(co.score), 3), "design_fine_rms_px": round(float(fine.rms_residual_px), 2)}
    if co.score < 0.15:
        info["design_used"] = False
        return info
    roles = [(spec.trace_layers, 1), (spec.component_layers, 2)]
    designs = [(Design.load(spec.gds, spec.gds_cell, ls), code) for ls, code in roles if ls]
    region = warp.affine.apply_bbox(des_all.bbox).intersection(src.bbox_px)
    if region is None:
        info["design_used"] = False
        return info
    for t in make_tiles(src.shape, 2048, 0, region):
        x0, y0, x1, y1 = t.core.to_int()
        a = warp.local_affine(t.core)
        blk = np.asarray(cls[y0:y1, x0:x1])
        for des, code in designs:
            m = des.render(a, BBox(x0, y0, x1, y1), supersample=2)
            if code == 1:  # traces matter at their edges (gaps): a band around every polygon outline
                m = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8))
            else:
                m = cv2.dilate(m, np.ones((5, 5), np.uint8))
            blk[(m > 0) & (blk != 3)] = code
        cls[y0:y1, x0:x1] = blk
    info["design_used"] = True
    return info

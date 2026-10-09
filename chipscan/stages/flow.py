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
    speck_threshold: float = 0.5
    trace_threshold: float = 0.5
    component_z: float = 6.0
    min_area_px: int = 4
    layout_model: str = str(ROOT / "models" / "layout_v1.pt")
    speck_model: str = str(ROOT / "models" / "speck_v1.pt")
    trace_model: str = str(ROOT / "models" / "trace_v1.pt")
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


# ----------------------------------------------------------------------------- findings from maps
def extract_findings(run_dir: Path, spec: ScanSpec, source: ImageSource | None = None) -> tuple[list[dict], dict]:
    """Speck + trace findings from the stored probability maps (cheap: no network run)."""
    from chipscan.segmentation.anomaly import anomalies_from_map
    from chipscan.segmentation.repeats import estimate_pitches, suppress_repeats
    from chipscan.stages.risk import suppress_text

    source = source or PyramidSource(run_dir / "corrected.zarr")
    px = spec.pixel_size_um or getattr(source, "pixel_size_um", None) or 1.0
    cls = zarr.open_group(str(run_dir / "layout.zarr"), mode="r")["0"]
    info: dict = {}
    out: list[dict] = []
    for group, thr, merge in (("speck", spec.speck_threshold, 6), ("trace", spec.trace_threshold, 8)):
        amap = zarr.open_group(str(run_dir / f"{group}.zarr"), mode="r")["0"]
        recs = anomalies_from_map(amap, thr, spec.min_area_px, px, merge_px=merge)
        n0 = len(recs)
        recs, on_text = suppress_text(recs, cls)
        if group == "speck" and recs:
            # designed things that recur at the reticle pitch are not dirt; tight patches so that a
            # real speck dominates its own patch and does not match its twin
            pitches = estimate_pitches(source)
            kept = []
            for d in recs:
                half = int(max(6, np.sqrt(d["area_px"]) + 5))
                k, _ = suppress_repeats(source, [d], pitches, half_px=half, search_px=4, match_ncc=0.85)
                kept += k
            info["speck_repeats_dropped"] = len(recs) - len(kept)
            recs = kept
        for d in recs:
            d["group"] = group
            d["kind"] = group
            d["class"] = group
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
            for j in np.nonzero(dd < 30)[0]:
                t = traces[j]
                if _iou(s["bbox_px"], t["bbox_px"]) > 0.3:
                    drop.add(id(t) if s["score"] >= t["score"] else id(s))
    out = [d for d in out if id(d) not in drop]
    info["dedup_dropped"] = len(drop)
    return out, info


# ----------------------------------------------------------------------------- the flow
def run_scan(spec: ScanSpec, runs_root: str | Path = ROOT / "runs", progress: Progress = _noop) -> Path:
    from chipscan.io.pyramid import build_pyramid
    from chipscan.stages.components import component_errors, find_instances
    from chipscan.stages.nets import TiledNet
    from chipscan.stages.risk import assess
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
        return lambda i, n: progress(key, i / max(n, 1), dict(STAGES)[key])

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
            TiledNet(model, spec.device).run(src, arr, progress=sub(key))
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
        comp_recs, csum = component_errors(src, insts, px, z_min=spec.component_z, progress=sub("components"))
        done("components", **csum)

        # 7. risk ---------------------------------------------------------------------
        stage("risk")
        st_recs, info = extract_findings(run_dir, spec, src)
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

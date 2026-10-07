"""End-to-end orchestration: ingest -> register -> segment -> compare -> metrology -> export.

Each stage is a plain function on the shared data model so you can run them one
at a time from a notebook, swap a segmenter, or reuse a saved registration.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from rich.console import Console
from rich.progress import Progress

from chipscan.config import PipelineConfig
from chipscan.design import Design
from chipscan.detection import Defect, RuleClassifier, compare_tile, measure_cut, merge_defects
from chipscan.detection.metrology import CutResult
from chipscan.export import write_csv, write_defects_gds, write_json, write_overlay, write_rdb
from chipscan.geometry import BBox, WarpField
from chipscan.io import open_source
from chipscan.io.positions import StagePosition, write_position_list
from chipscan.io.readers import ImageSource
from chipscan.registration import coarse_register, edge_map, fine_register
from chipscan.segmentation import Segmenter, build_segmenter
from chipscan.tiling import make_tiles

console = Console()


@dataclass
class InspectionResult:
    config: PipelineConfig
    source: ImageSource
    design: Design
    warp: WarpField
    pixel_size_um: float
    defects: list[Defect]
    cuts: list[CutResult] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    registration_info: dict[str, Any] = field(default_factory=dict)
    pred_mask_path: Path | None = None


# ---------------------------------------------------------------------------
def load_inputs(cfg: PipelineConfig) -> tuple[ImageSource, Design]:
    src = open_source(
        cfg.image.path,
        cfg.image.pixel_size_um,
        cfg.image.pyramid_path,
        channel=cfg.image.channel,
        chunk_px=cfg.image.chunk_px,
        invert=cfg.image.invert,
    )
    des = Design.load(cfg.design.path, cfg.design.cell, cfg.design.layers, cfg.design.polarity, cfg.design.region_um)
    console.print(f"image {src.shape[1]}x{src.shape[0]} px, {src.num_levels} levels; design {des.num_polygons} polygons, bbox {des.bbox.width:.0f}x{des.bbox.height:.0f} um")
    return src, des


def register(cfg: PipelineConfig, src: ImageSource, des: Design) -> tuple[WarpField, float, dict[str, Any]]:
    tp = cfg.registration.transform_path
    if tp is not None and Path(tp).exists():
        warp = WarpField.load(tp)
        px = 1.0 / warp.affine.scale_factor
        console.print(f"loaded registration from {tp}: {warp}")
        return warp, px, {"loaded_from": str(tp)}
    t0 = time.time()
    coarse = coarse_register(src, des, cfg.registration.coarse, src.pixel_size_um)
    console.print(f"coarse: {coarse.affine} score={coarse.score:.3f} ({time.time() - t0:.1f}s)")
    info: dict[str, Any] = {"coarse": coarse.to_dict()}
    warp = WarpField(coarse.affine)
    px = coarse.pixel_size_um
    if cfg.registration.fine.enabled:
        t1 = time.time()
        fine = fine_register(src, des, coarse.affine, cfg.registration.fine, px)
        console.print(f"fine: {fine.warp} rms={fine.rms_residual_px:.2f}px used {fine.n_used}/{fine.n_total} tiles ({time.time() - t1:.1f}s)")
        info["fine"] = fine.to_dict()
        warp = fine.warp
        px = 1.0 / warp.affine.scale_factor
    return warp, px, info


def segment_and_compare(
    cfg: PipelineConfig,
    src: ImageSource,
    des: Design,
    warp: WarpField,
    pixel_size_um: float,
    segmenter: Segmenter | None = None,
    pred_store: zarr.Array | None = None,
    classifier=None,
) -> list[Defect]:
    seg = segmenter or build_segmenter(cfg.segmentation, cfg.design.metal_bright)
    region = warp.affine.apply_bbox(des.bbox).intersection(src.bbox_px)
    if region is None:
        raise RuntimeError("design does not overlap image")
    tiles = make_tiles(src.shape, cfg.segmentation.tile_px, cfg.segmentation.margin_px, region)
    classifier = classifier or RuleClassifier(tolerance_px=cfg.detection.tolerance_px)
    defects: list[Defect] = []
    with Progress() as prog:
        task = prog.add_task(f"segment+compare ({seg.name})", total=len(tiles))
        for t in tiles:
            x0, y0, x1, y1 = t.read.to_int()
            img = src.read_region(x0, y0, x1, y1, level=0).astype(np.float32) / 255.0
            a = warp.local_affine(t.read)
            prior = des.render(a, BBox(x0, y0, x1, y1))
            prob = seg.predict(img[None], prior[None])[0]
            sl = t.core_slice_in_read
            prob_c, design_c = np.ascontiguousarray(prob[sl]), np.ascontiguousarray(prior[sl])
            if pred_store is not None:
                cx0, cy0, cx1, cy1 = t.core.to_int()
                pred_store[cy0:cy1, cx0:cx1] = np.rint(prob_c * 255).astype(np.uint8)
            defects.extend(
                compare_tile(prob_c, design_c, (t.core.x0, t.core.y0), warp, pixel_size_um, cfg.detection, t.index, classifier, len(defects), cfg.segmentation.threshold)
            )
            prog.advance(task)
    merged = merge_defects(defects, cfg.detection.merge_margin_um)
    console.print(f"{len(defects)} raw blobs -> {len(merged)} defects")
    return merged


def run_metrology(cfg: PipelineConfig, pred: zarr.Array | np.ndarray, warp: WarpField, pixel_size_um: float) -> list[CutResult]:
    if not cfg.metrology.cuts:
        return []

    def read_prob(b: BBox) -> np.ndarray:
        x0, y0, x1, y1 = b.to_int()
        h, w = pred.shape
        out = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
        if cx1 > cx0 and cy1 > cy0:
            out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = np.asarray(pred[cy0:cy1, cx0:cx1]).astype(np.float32) / 255.0
        return out

    return [measure_cut(c, read_prob, warp, pixel_size_um, cfg.metrology.sample_step_um) for c in cfg.metrology.cuts]


def export(res: InspectionResult) -> dict[str, Path]:
    cfg, out = res.config, res.config.export.out_dir
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    name = cfg.name
    res.warp.save(out / f"{name}.registration.json")
    written["registration"] = out / f"{name}.registration.json"
    (out / f"{name}.registration_info.json").write_text(json.dumps(res.registration_info, indent=2, default=str))
    extra = {"pixel_size_um": res.pixel_size_um, "timings_s": res.timings, "image": str(cfg.image.path), "design": str(cfg.design.path)}
    if res.cuts:
        extra["metrology"] = [c.to_dict() for c in res.cuts]
    write_json(out / f"{name}.defects.json", res.defects, extra)
    write_csv(out / f"{name}.defects.csv", res.defects)
    written["json"], written["csv"] = out / f"{name}.defects.json", out / f"{name}.defects.csv"
    if cfg.export.klayout_rdb:
        p = out / f"{name}.lyrdb"
        write_rdb(p, res.defects, res.design.cell_name, original_file=str(cfg.design.path))
        written["lyrdb"] = p
    if cfg.export.gds_layer:
        layer, dt = (int(v) for v in cfg.export.gds_layer.split("/"))
        p = out / f"{name}.defects.gds"
        write_defects_gds(p, res.defects, res.design.cell_name, layer, dt)
        written["gds"] = p
    if cfg.export.positions_xml and cfg.stage.enabled:
        ox, oy = cfg.stage.origin_stage_um
        pos = [
            StagePosition(i, f"{d.cls}#{d.id}", ox + d.centroid_um[0], oy + (-d.centroid_um[1] if cfg.stage.flip_y else d.centroid_um[1]), cfg.stage.z_um)
            for i, d in enumerate(res.defects)
        ]
        p = out / f"{name}.revisit_positions.xml"
        write_position_list(p, pos)
        written["positions"] = p
    if cfg.export.overlay_png:
        p = out / f"{name}.overlay.png"
        lvl = 0
        while lvl + 1 < res.source.num_levels and max(res.source.level_shape(lvl)) > cfg.export.overlay_max_px:
            lvl += 1
        s = 2**lvl
        from chipscan.geometry import Affine2D

        a = Affine2D.scaling(1.0 / s).compose(res.warp.affine)
        h, w = res.source.level_shape(lvl)
        de = edge_map(res.design.render(a, BBox(0, 0, w, h)).astype(np.float32), 0.0)
        write_overlay(p, res.source, res.defects, res.warp, cfg.export.overlay_max_px, de)
        written["overlay"] = p
    return written


def run_inspection(cfg: PipelineConfig, segmenter: Segmenter | None = None, classifier=None) -> InspectionResult:
    timings: dict[str, float] = {}
    t = time.time()
    src, des = load_inputs(cfg)
    timings["load"] = time.time() - t
    t = time.time()
    warp, px, info = register(cfg, src, des)
    timings["register"] = time.time() - t
    pred_path = None
    pred_store = None
    if cfg.export.save_pred_mask:
        cfg.export.out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = cfg.export.out_dir / f"{cfg.name}.pred.zarr"
        g = zarr.open_group(str(pred_path), mode="w")
        pred_store = g.create_array("0", shape=src.shape, chunks=(cfg.image.chunk_px, cfg.image.chunk_px), dtype="uint8")
        g.attrs["pixel_size_um"] = px
    t = time.time()
    defects = segment_and_compare(cfg, src, des, warp, px, segmenter, pred_store, classifier)
    timings["segment_compare"] = time.time() - t
    cuts: list[CutResult] = []
    if cfg.metrology.cuts and pred_store is not None:
        t = time.time()
        cuts = run_metrology(cfg, pred_store, warp, px)
        timings["metrology"] = time.time() - t
    res = InspectionResult(cfg, src, des, warp, px, defects, cuts, timings, info, pred_path)
    t = time.time()
    written = export(res)
    timings["export"] = time.time() - t
    console.print("wrote: " + ", ".join(f"{k}={v.name}" for k, v in written.items()))
    return res

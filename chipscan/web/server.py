"""Web viewer: deep-zoom inspection results with defect review.

Run with ``chipscan serve run1.yaml run2.yaml --port 8000`` and open http://localhost:8000.
Each pipeline YAML whose ``inspect`` outputs exist becomes a "run". Tiles are cut from the
zarr pyramid on demand and cached on disk; design edges and model heatmaps are rendered per
tile through the saved registration. Review decisions go to a SQLite file next to the outputs.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import zarr
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chipscan.config import PipelineConfig
from chipscan.design import Design
from chipscan.detection.metrology import runs_from_profile, sample_along
from chipscan.geometry import Affine2D, BBox, WarpField
from chipscan.io import open_source
from chipscan.io.positions import StagePosition, write_position_list
from chipscan.io.readers import ImageSource

TILE = 256
STATIC = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
@dataclass
class Run:
    name: str
    cfg: PipelineConfig | None
    source: ImageSource
    design: Design | None
    warp: WarpField
    pixel_size_um: float
    defects: list[dict]
    pred: zarr.Array | None
    out_dir: Path
    cache_dir: Path
    reg_info: dict = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @staticmethod
    def load_anomaly(anomalies_json: Path) -> Run:
        """A design-free run: <name>.anomalies.json next to <name>.anomaly.zarr (from `chipscan detect`)."""
        anomalies_json = Path(anomalies_json)
        out = anomalies_json.parent
        name = anomalies_json.name[: -len(".anomalies.json")]
        rec = json.loads(anomalies_json.read_text())
        defects = rec["defects"]
        px = rec.get("summary", {}).get("pixel_size_um") or 1.0
        src_path = rec.get("summary", {}).get("image")
        if not src_path:
            raise FileNotFoundError(f"{anomalies_json}: no image path recorded; re-run `chipscan detect`")
        src = open_source(src_path, px)
        amap = None
        az = out / f"{name}.anomaly.zarr"
        if az.exists():
            amap = zarr.open_group(str(az), mode="r")["0"]
        for d in defects:  # anomalies already carry px polygons; keep the viewer's field names
            d.setdefault("polygons_px", [])
            d.setdefault("bbox_um", {k: v * px for k, v in d["bbox_px"].items()})
        cfg = None
        run = Run(name, cfg, src, None, WarpField(Affine2D.scaling(1.0 / px)), px, defects, amap, out, out / f"{name}.tiles")
        run.cache_dir.mkdir(parents=True, exist_ok=True)
        run.anomaly_run = True
        run.specks = []
        for extra_kind in ("specks", "die2die"):
            fp = out / f"{name}.{extra_kind}.json"
            if fp.exists():
                for d in json.loads(fp.read_text())["defects"]:
                    d.setdefault("polygons_px", [])
                    d.setdefault("bbox_um", {k: v * px for k, v in d["bbox_px"].items()})
                    run.specks.append(d)
        run.extract_params = {"threshold": rec.get("summary", {}).get("threshold", 0.5), "min_area_px": 6, "merge_px": 8, "include_specks": True, "include_die2die": True}
        run.defects = run._with_specks(defects)
        return run

    anomaly_run: bool = False
    specks: list[dict] = field(default_factory=list)
    extract_params: dict = field(default_factory=dict)
    scan_run: bool = False
    meta: dict = field(default_factory=dict)
    layout: zarr.Group | None = None
    heat: zarr.Group | None = None

    @staticmethod
    def load_scan(run_dir: Path) -> Run:
        """A run of the full flow (`chipscan add` / the Add-scan dialog): runs/<name>/."""
        from chipscan.io.readers import PyramidSource

        run_dir = Path(run_dir)
        meta = json.loads((run_dir / "scan.json").read_text())
        if meta.get("status") != "done":
            raise FileNotFoundError(f"{run_dir}: scan not finished ({meta.get('status')})")
        rec = json.loads((run_dir / "findings.json").read_text())
        src = PyramidSource(run_dir / "corrected.zarr")
        px = meta["spec"].get("pixel_size_um") or src.pixel_size_um or 1.0
        src.pixel_size_um = px
        run = Run(meta["spec"]["name"], None, src, None, WarpField(Affine2D.scaling(1.0 / px)), px, rec["findings"], None, run_dir, run_dir / "tiles")
        run.cache_dir.mkdir(parents=True, exist_ok=True)
        run.scan_run, run.meta = True, meta
        run.layout = zarr.open_group(str(run_dir / "layout.zarr"), mode="r")
        run.heat = zarr.open_group(str(run_dir / "heat.zarr"), mode="r")
        run.pred = run.heat["0"]
        for d in run.defects:
            d.setdefault("polygons_px", [])
        return run

    def rethreshold(self, speck: float, trace: float, min_area_px: int) -> int:
        from chipscan.stages.flow import ScanSpec, refresh_findings

        with self.lock:
            sp = {k: v for k, v in self.meta["spec"].items() if k in ScanSpec.__dataclass_fields__}
            sp.update({"speck_threshold": speck, "trace_threshold": trace, "min_area_px": min_area_px})
            spec = ScanSpec(**sp)
            self.defects = refresh_findings(self.out_dir, spec)
            self.meta = json.loads((self.out_dir / "scan.json").read_text())
            for d in self.defects:
                d.setdefault("polygons_px", [])
        return len(self.defects)

    def layout_tile(self, level: int, x: int, y: int) -> bytes:
        p = self.cache_dir / f"lay_{level}_{x}_{y}.png"
        if p.exists():
            return p.read_bytes()
        rgba = np.zeros((TILE, TILE, 4), np.uint8)
        if self.layout is not None and str(level) in self.layout:
            a = self.layout[str(level)]
            h, w = a.shape
            x0, y0 = x * TILE, y * TILE
            blk = np.asarray(a[y0 : min(y0 + TILE, h), x0 : min(x0 + TILE, w)])
            hh, ww = blk.shape
            # BGRA: trace blue, component red, text green
            for code, col in ((1, (255, 150, 40, 120)), (2, (60, 60, 255, 190)), (3, (80, 220, 80, 160))):
                rgba[:hh, :ww][blk == code] = col
        data = _png_bytes(rgba)
        p.write_bytes(data)
        return data

    def _with_specks(self, recs: list[dict]) -> list[dict]:
        """Append the classical finds (faint specks, die-to-die) that the learned records do not already cover."""
        inc_sp = self.extract_params.get("include_specks", True)
        inc_dd = self.extract_params.get("include_die2die", True)
        min_px = self.extract_params.get("min_area_px", 0)
        extra = [d for d in self.specks if ((d["kind"] == "speck" and inc_sp) or (d["kind"] == "die2die" and inc_dd)) and d.get("area_px", 0) >= min_px]
        if not extra:
            return recs
        out = list(recs)
        if recs:
            cxy = np.array([r["centroid_px"] for r in recs])
            sp = [d for d in extra if np.hypot(*(cxy - d["centroid_px"]).T).min() > 12]
        else:
            sp = list(extra)
        nid = (max(r["id"] for r in recs) + 1) if recs else 0
        for d in sp:
            d = dict(d)
            d["id"] = nid
            nid += 1
            out.append(d)
        return out

    def reextract(self, threshold: float, min_area_px: int, merge_px: int, include_specks: bool, include_die2die: bool = True) -> int:
        """Recompute anomaly records from the stored probability map (no model run needed)."""
        from chipscan.segmentation.anomaly import anomalies_from_map

        if self.pred is None:
            raise HTTPException(400, "no anomaly map stored for this run")
        with self.lock:
            recs = anomalies_from_map(self.pred, threshold, min_area_px, self.pixel_size_um, merge_px=merge_px)
            for d in recs:
                d.setdefault("polygons_px", [])
            self.extract_params = {"threshold": threshold, "min_area_px": min_area_px, "merge_px": merge_px, "include_specks": include_specks, "include_die2die": include_die2die}
            self.defects = self._with_specks(recs)
        return len(self.defects)

    @staticmethod
    def load(config_path: Path) -> Run:
        cfg = PipelineConfig.load(config_path)
        out = cfg.export.out_dir
        reg = out / f"{cfg.name}.registration.json"
        dj = out / f"{cfg.name}.defects.json"
        if not reg.exists() or not dj.exists():
            raise FileNotFoundError(f"{config_path}: run `chipscan inspect` first (missing {reg.name} or {dj.name})")
        src = open_source(cfg.image.path, cfg.image.pixel_size_um, cfg.image.pyramid_path, cfg.image.channel, cfg.image.chunk_px, cfg.image.invert)
        des = Design.load(cfg.design.path, cfg.design.cell, cfg.design.layers, cfg.design.polarity, cfg.design.region_um)
        warp = WarpField.load(reg)
        px = 1.0 / warp.affine.scale_factor
        rec = json.loads(dj.read_text())
        defects = rec["defects"]
        pred = None
        pz = out / f"{cfg.name}.pred.zarr"
        if pz.exists():
            pred = zarr.open_group(str(pz), mode="r")["0"]
        info_p = out / f"{cfg.name}.registration_info.json"
        info = json.loads(info_p.read_text()) if info_p.exists() else {}
        cache = out / f"{cfg.name}.tiles"
        cache.mkdir(parents=True, exist_ok=True)
        run = Run(cfg.name, cfg, src, des, warp, px, defects, pred, out, cache, info)
        run._attach_polygons()
        return run

    def _attach_polygons(self) -> None:
        """Defect outlines live in the defects GDS (um); pull them in as px polygons."""
        gds_path = self.out_dir / f"{self.name}.defects.gds"
        by_id: dict[int, list] = {}
        if gds_path.exists():
            import gdstk

            cell = gdstk.read_gds(str(gds_path)).top_level()[0]
            labels = [(np.array(lab.origin), int(lab.text.split("#")[-1])) for lab in cell.labels]
            if labels:
                lab_xy = np.array([o for o, _ in labels])
                lab_id = [i for _, i in labels]
                for p in cell.polygons:
                    c = np.asarray(p.points).mean(axis=0)
                    k = int(np.argmin(((lab_xy - c) ** 2).sum(axis=1)))
                    by_id.setdefault(lab_id[k], []).append(np.asarray(p.points, dtype=float))
        for d in self.defects:
            polys = by_id.get(d["id"], [])
            d["polygons_px"] = [self.warp.world_to_px(p).round(2).tolist() for p in polys]

    # ---- tiles --------------------------------------------------------
    @property
    def num_levels(self) -> int:
        return self.source.num_levels

    def image_tile(self, level: int, x: int, y: int) -> bytes:
        p = self.cache_dir / f"img_{level}_{x}_{y}.jpg"
        if p.exists():
            return p.read_bytes()
        s = 2**level
        x0, y0 = x * TILE * s, y * TILE * s
        arr = self.source.read_region(x0, y0, x0 + TILE * s, y0 + TILE * s, level=level)
        ok, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        data = buf.tobytes()
        p.write_bytes(data)
        return data

    def design_tile(self, level: int, x: int, y: int) -> bytes:
        p = self.cache_dir / f"des_{level}_{x}_{y}.png"
        if p.exists():
            return p.read_bytes()
        if self.scan_run:
            return self.layout_tile(level, x, y)
        if self.design is None:
            return _png_bytes(np.zeros((TILE, TILE, 4), np.uint8))
        s = 2**level
        bb0 = BBox(x * TILE * s, y * TILE * s, (x + 1) * TILE * s, (y + 1) * TILE * s)
        a = Affine2D.scaling(1.0 / s).compose(self.warp.local_affine(bb0))
        mask = self.design.render(a, BBox(x * TILE, y * TILE, (x + 1) * TILE, (y + 1) * TILE), supersample=2 if level > 0 else 4)
        edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
        rgba = np.zeros((TILE, TILE, 4), np.uint8)
        rgba[edge] = (90, 230, 90, 230)  # BGRA
        ok, buf = cv2.imencode(".png", rgba)
        data = buf.tobytes()
        p.write_bytes(data)
        return data

    def pred_tile(self, level: int, x: int, y: int) -> bytes:
        p = self.cache_dir / f"pred_{level}_{x}_{y}.png"
        if p.exists():
            return p.read_bytes()
        rgba = np.zeros((TILE, TILE, 4), np.uint8)
        if self.heat is not None and str(level) in self.heat:
            a = self.heat[str(level)]
            h, w = a.shape
            x0, y0 = x * TILE, y * TILE
            blk = np.asarray(a[y0 : min(y0 + TILE, h), x0 : min(x0 + TILE, w)])
            hh, ww = blk.shape
            rgba[:hh, :ww, :3] = cv2.applyColorMap(blk, cv2.COLORMAP_INFERNO)
            rgba[:hh, :ww, 3] = np.clip(blk.astype(np.int32) * 2, 0, 200).astype(np.uint8)
        elif self.pred is not None and level <= 4:
            s = 2**level
            x0, y0 = x * TILE * s, y * TILE * s
            h, w = self.pred.shape
            x1, y1 = min(x0 + TILE * s, w), min(y0 + TILE * s, h)
            if x1 > x0 and y1 > y0:
                blk = np.asarray(self.pred[y0:y1, x0:x1])
                small = cv2.resize(blk, (max(1, (x1 - x0) // s), max(1, (y1 - y0) // s)), interpolation=cv2.INTER_AREA)
                hh, ww = small.shape
                heat = cv2.applyColorMap(small, cv2.COLORMAP_VIRIDIS)
                rgba[:hh, :ww, :3] = heat
                rgba[:hh, :ww, 3] = 150
        ok, buf = cv2.imencode(".png", rgba)
        data = buf.tobytes()
        p.write_bytes(data)
        return data

    def thumb(self, defect_id: int, size: int = 96) -> bytes:
        d = next((d for d in self.defects if d["id"] == defect_id), None)
        if d is None or not d.get("bbox_px"):
            raise HTTPException(404)
        b = d["bbox_px"]
        cx, cy = 0.5 * (b["x0"] + b["x1"]), 0.5 * (b["y0"] + b["y1"])
        half = max(size // 2, int(max(b["x1"] - b["x0"], b["y1"] - b["y0"]) * 0.8))
        x0, y0 = int(cx - half), int(cy - half)
        arr = self.source.read_region(x0, y0, x0 + 2 * half, y0 + 2 * half)
        rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        for poly in d.get("polygons_px", []):
            pts = np.rint(np.array(poly) - [x0, y0]).astype(np.int32)
            cv2.polylines(rgb, [pts], True, (60, 60, 255), 1)
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    def measure(self, x0: float, y0: float, x1: float, y1: float) -> dict:
        if self.pred is None or self.design is None:
            raise HTTPException(400, "measurement needs a design-referenced run with a prediction mask")
        p0 = self.warp.px_to_world(np.array([[x0, y0]]))[0]
        p1 = self.warp.px_to_world(np.array([[x1, y1]]))[0]

        def read_prob(b: BBox) -> np.ndarray:
            bx0, by0, bx1, by1 = b.to_int()
            h, w = self.pred.shape
            out = np.zeros((by1 - by0, bx1 - bx0), np.float32)
            cx0, cy0, cx1, cy1 = max(bx0, 0), max(by0, 0), min(bx1, w), min(by1, h)
            if cx1 > cx0 and cy1 > cy0:
                out[cy0 - by0 : cy1 - by0, cx0 - bx0 : cx1 - bx0] = np.asarray(self.pred[cy0:cy1, cx0:cx1]).astype(np.float32) / 255
            return out

        t, v = sample_along(read_prob, self.warp, p0, p1, self.pixel_size_um / 4)
        runs, edges = runs_from_profile(t, v)
        return {"length_um": float(np.linalg.norm(p1 - p0)), "runs": [{"phase": a, "length_um": round(b, 3)} for a, b in runs], "edges_um": [round(e, 3) for e in edges]}


def _centroid_px(d: dict) -> tuple[float, float] | None:
    """Centroid in level-0 px for any record kind (design-based records carry only bbox_px)."""
    c = d.get("centroid_px")
    if c is not None:
        return float(c[0]), float(c[1])
    b = d.get("bbox_px")
    if b:
        return 0.5 * (b["x0"] + b["x1"]), 0.5 * (b["y0"] + b["y1"])
    return None


# ---------------------------------------------------------------------------
class ReviewDB:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS review (run TEXT, defect_id INTEGER, status TEXT, cls TEXT, note TEXT, reviewer TEXT, ts REAL, PRIMARY KEY (run, defect_id))"
            )
            cols = [r[1] for r in c.execute("PRAGMA table_info(review)").fetchall()]
            if "cx" not in cols:
                c.execute("ALTER TABLE review ADD COLUMN cx REAL")
                c.execute("ALTER TABLE review ADD COLUMN cy REAL")
            c.execute("CREATE TABLE IF NOT EXISTS review_pos (run TEXT, cx REAL, cy REAL, status TEXT, cls TEXT, note TEXT, reviewer TEXT, ts REAL, PRIMARY KEY (run, cx, cy))")

    def _conn(self):
        return sqlite3.connect(str(self.path))

    def positions(self, run: str) -> list[dict]:
        """All position-keyed reviews for a run."""
        with self.lock, self._conn() as c:
            rows = c.execute("SELECT cx, cy, status, cls, note, reviewer, ts FROM review_pos WHERE run=?", (run,)).fetchall()
        return [{"cx": r[0], "cy": r[1], "status": r[2], "cls": r[3], "note": r[4], "reviewer": r[5], "ts": r[6]} for r in rows]

    def get(self, run: str, defects: list[dict] | None = None, radius_px: float = 12.0) -> dict[int, dict]:
        """Reviews keyed by current defect id: matched by position so they survive re-extraction."""
        pos = self.positions(run)
        if defects is None or not pos:
            return {}
        pxy = np.array([[p["cx"], p["cy"]] for p in pos])
        out: dict[int, dict] = {}
        for d in defects:
            c = _centroid_px(d)
            if c is None:
                continue
            dist = np.hypot(pxy[:, 0] - c[0], pxy[:, 1] - c[1])
            k = int(np.argmin(dist))
            if dist[k] <= radius_px:
                p = pos[k]
                out[d["id"]] = {"status": p["status"], "cls": p["cls"], "note": p["note"], "reviewer": p["reviewer"], "ts": p["ts"]}
        return out

    def set(self, run: str, cx: float, cy: float, status: str, cls: str | None, note: str | None, reviewer: str | None, radius_px: float = 12.0) -> None:
        with self.lock, self._conn() as c:
            # replace an existing review at (about) the same position
            rows = c.execute("SELECT cx, cy FROM review_pos WHERE run=? AND cx BETWEEN ? AND ? AND cy BETWEEN ? AND ?", (run, cx - radius_px, cx + radius_px, cy - radius_px, cy + radius_px)).fetchall()
            for ox, oy in rows:
                c.execute("DELETE FROM review_pos WHERE run=? AND cx=? AND cy=?", (run, ox, oy))
            c.execute("INSERT INTO review_pos (run, cx, cy, status, cls, note, reviewer, ts) VALUES (?,?,?,?,?,?,?,?)", (run, float(cx), float(cy), status, cls, note, reviewer, time.time()))


class ReviewIn(BaseModel):
    defect_id: int
    status: str  # real | false_alarm | unsure | unreviewed
    cls: str | None = None
    note: str | None = None
    reviewer: str | None = None


# ---------------------------------------------------------------------------
class Jobs:
    """One worker thread runs scans in order; the UI polls progress."""

    def __init__(self, runs_root: Path, on_done):
        import queue

        self.runs_root = runs_root
        self.on_done = on_done
        self.jobs: dict[str, dict] = {}
        self.q: queue.Queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, spec, resume: bool = False) -> dict:
        from chipscan.stages.flow import STAGES

        jid = f"{spec.name}-{int(time.time())}"
        job = {
            "id": jid,
            "name": spec.name,
            "status": "queued",
            "stage": None,
            "stages": [{"key": k, "label": lab, "status": "pending", "progress": 0.0} for k, lab in STAGES],
            "created": time.time(),
            "error": None,
        }
        with self.lock:
            self.jobs[jid] = job
        job["resumed"] = resume
        self.q.put((jid, spec, resume))
        return job

    def _progress(self, jid: str):
        def cb(stage: str, frac: float, msg: str):
            with self.lock:
                job = self.jobs[jid]
                job["stage"] = stage
                seen = False
                for st in job["stages"]:
                    if st["key"] == stage:
                        st["status"] = "done" if frac >= 1.0 else "running"
                        st["progress"] = round(float(frac), 3)
                        seen = True
                    elif not seen and st["status"] != "done":
                        st["status"], st["progress"] = "done", 1.0

        return cb

    def _worker(self):
        from chipscan.stages.flow import run_scan

        while True:
            jid, spec, resume = self.q.get()
            with self.lock:
                self.jobs[jid]["status"] = "running"
                self.jobs[jid]["started"] = time.time()
            try:
                run_dir = run_scan(spec, self.runs_root, self._progress(jid), resume=resume)
                self.on_done(run_dir)
                with self.lock:
                    self.jobs[jid]["status"] = "done"
            except Exception as e:  # surface the failure in the UI
                with self.lock:
                    self.jobs[jid]["status"] = "failed"
                    self.jobs[jid]["error"] = f"{type(e).__name__}: {e}"
            with self.lock:
                self.jobs[jid]["finished"] = time.time()

    def resume_interrupted(self) -> list[str]:
        """Scans whose server stopped mid-run (status still "running") continue where they were,
        with today's models for the stages they have not reached."""
        from dataclasses import fields

        from chipscan.stages.flow import ScanSpec

        names = []
        for meta_p in sorted(self.runs_root.glob("*/scan.json"), key=lambda q: q.stat().st_mtime):
            try:
                meta = json.loads(meta_p.read_text())
            except Exception:
                continue
            if meta.get("status") != "running":
                continue
            fresh = ScanSpec(name="", image="")
            known = {f.name for f in fields(ScanSpec)}
            spec = ScanSpec(**{k: v for k, v in meta.get("spec", {}).items() if k in known})
            done = {k for k, v in meta.get("stages", {}).items() if v.get("seconds") is not None}
            for key, attr in (("specks", "speck_model"), ("specks", "anomaly_model"), ("traces", "trace_model"), ("layout", "layout_model")):
                if key not in done:
                    setattr(spec, attr, getattr(fresh, attr))
            self.submit(spec, resume=True)
            names.append(spec.name)
        return names

    def list(self) -> list[dict]:
        with self.lock:
            return sorted((json.loads(json.dumps(j)) for j in self.jobs.values()), key=lambda j: -j["created"])


IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".jp2", ".bmp"}


class Watcher:
    """Live scans: watch microscope export folders and inspect every new image automatically.

    PRECiV (or any acquisition software) writes the stitched image into a folder; once a new
    image file has stopped growing for `settle_s`, it is queued like an uploaded scan. Watched
    folders survive restarts (runs/_watch.json). Files present when watching starts are skipped
    unless `include_existing`."""

    def __init__(self, runs_root: Path, jobs: "Jobs", poll_s: float = 5.0, settle_s: float = 10.0):
        self.runs_root, self.jobs, self.poll_s, self.settle_s = runs_root, jobs, poll_s, settle_s
        self.cfg = runs_root / "_watch.json"
        self.folders: dict[str, dict] = {}
        self.lock = threading.Lock()
        if self.cfg.exists():
            try:
                self.folders = json.loads(self.cfg.read_text())
            except Exception:
                self.folders = {}
        threading.Thread(target=self._loop, daemon=True).start()

    def _save(self) -> None:
        self.cfg.parent.mkdir(parents=True, exist_ok=True)
        self.cfg.write_text(json.dumps(self.folders, indent=2))

    def _images(self, folder: Path) -> dict[str, tuple[int, float]]:
        out = {}
        for f in folder.rglob("*"):
            if f.suffix.lower() in IMAGE_EXT and f.is_file() and not f.name.startswith("."):
                st = f.stat()
                out[str(f)] = (st.st_size, st.st_mtime)
        return out

    def add(self, path: str, pixel_size_um: float | None = None, straighten: bool = True, include_existing: bool = False) -> dict:
        folder = Path(path).expanduser()
        if not folder.is_dir():
            raise ValueError(f"not a folder on this computer: {path}")
        seen = [] if include_existing else list(self._images(folder))
        with self.lock:
            self.folders[str(folder)] = {"path": str(folder), "pixel_size_um": pixel_size_um, "straighten": straighten, "seen": seen, "pending": {}, "started": time.time(), "queued": []}
            self._save()
            return self._public(self.folders[str(folder)])

    def remove(self, path: str) -> None:
        with self.lock:
            self.folders.pop(str(Path(path).expanduser()), None)
            self._save()

    @staticmethod
    def _public(w: dict) -> dict:
        return {k: w[k] for k in ("path", "pixel_size_um", "straighten", "started", "queued")}

    def list(self) -> list[dict]:
        with self.lock:
            return [self._public(w) for w in self.folders.values()]

    def _loop(self) -> None:
        from chipscan.stages.flow import ScanSpec

        while True:
            time.sleep(self.poll_s)
            with self.lock:
                items = list(self.folders.values())
            for w in items:
                folder = Path(w["path"])
                if not folder.is_dir():
                    continue
                try:
                    now = self._images(folder)
                except OSError:
                    continue
                changed = False
                for f, (size, mtime) in now.items():
                    if f in w["seen"]:
                        continue
                    prev = w["pending"].get(f)
                    if prev is None or prev[0] != size:  # still being written
                        w["pending"][f] = [size, time.time()]
                        continue
                    if time.time() - prev[1] < self.settle_s:
                        continue
                    name = _slug(Path(f).stem)
                    if (self.runs_root / name / "scan.json").exists():
                        name = f"{name}_{time.strftime('%m%d_%H%M%S')}"
                    spec = ScanSpec(name=name, image=f, pixel_size_um=w.get("pixel_size_um"), straighten=w.get("straighten", True))
                    self.jobs.submit(spec)
                    w["seen"].append(f)
                    w["pending"].pop(f, None)
                    w["queued"].append({"file": f, "name": name, "at": time.time()})
                    changed = True
                if changed:
                    with self.lock:
                        self._save()


def _slug(name: str) -> str:
    import re

    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()).strip("._")
    return s[:80] or f"scan_{int(time.time())}"


def create_app(config_paths: list[Path], db_path: Path | None = None, runs_root: Path | None = None) -> FastAPI:
    runs: dict[str, Run] = {}
    runs_root = Path(runs_root) if runs_root else Path.cwd() / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    # finished scans first (newest first): they are what the lab works from; older runs follow
    scans = [d for d in runs_root.iterdir() if (d / "scan.json").exists()]
    for d in sorted(scans, key=lambda d: -(d / "scan.json").stat().st_mtime):
        try:
            r = Run.load_scan(d)
            runs[r.name] = r
        except Exception:
            pass
    for p in config_paths:
        p = Path(p)
        if p.is_dir() and (p / "scan.json").exists():
            r = Run.load_scan(p)
        else:
            r = Run.load_anomaly(p) if p.name.endswith(".anomalies.json") else Run.load(p)
        runs.setdefault(r.name, r)
    db = ReviewDB(db_path or ((next(iter(runs.values())).out_dir if runs else runs_root) / "reviews.sqlite"))

    def add_run(run_dir: Path) -> None:
        r = Run.load_scan(run_dir)
        runs[r.name] = r

    jobs = Jobs(runs_root, add_run)
    resumed = jobs.resume_interrupted()
    if resumed:
        print(f"resuming interrupted scans: {', '.join(resumed)}", flush=True)
    app = FastAPI(title="chipscan viewer")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.middleware("http")
    async def _revalidate_static(request, call_next):
        # the UI changes with the code: browsers must revalidate (ETag) instead of using a stale copy
        resp = await call_next(request)
        if request.url.path == "/" or request.url.path.startswith("/static/"):
            resp.headers["Cache-Control"] = "no-cache"
        return resp

    def get_run(name: str) -> Run:
        if name not in runs:
            raise HTTPException(404, f"unknown run {name}")
        return runs[name]

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/api/runs")
    def list_runs():
        out = []
        for r in runs.values():
            h, w = r.source.shape
            rv = db.get(r.name, r.defects)
            out.append(
                {
                    "name": r.name,
                    "width": w,
                    "height": h,
                    "levels": r.num_levels,
                    "pixel_size_um": r.pixel_size_um,
                    "n_defects": len(r.defects),
                    "n_reviewed": sum(1 for v in rv.values() if v["status"] != "unreviewed"),
                    "image": str(r.cfg.image.path.name) if r.cfg else getattr(r.source, "path", Path("?")).name,
                    "design": str(r.cfg.design.path.name) if r.cfg else "none (design-free)",
                    "segmenter": r.cfg.segmentation.method if r.cfg else ("speck + trace + component" if r.scan_run else "anomaly"),
                    "registration": r.warp.affine.to_dict(),
                    "anomaly_run": r.anomaly_run,
                    "extract_params": r.extract_params,
                    "n_specks": len(r.specks),
                    "scan_run": r.scan_run,
                    "summary": (r.meta or {}).get("summary"),
                    "scan": {
                        "straighten": ((r.meta or {}).get("stages") or {}).get("straighten"),
                        "layout": ((r.meta or {}).get("stages") or {}).get("layout"),
                        "components": ((r.meta or {}).get("stages") or {}).get("components"),
                        "spec": {k: (r.meta or {}).get("spec", {}).get(k) for k in ("speck_threshold", "trace_threshold", "min_area_px", "gds", "report_traces")},
                    }
                    if r.scan_run
                    else None,
                    "fine_rms_px": (r.reg_info.get("fine") or {}).get("rms_residual_px"),
                    "coarse_score": (r.reg_info.get("coarse") or {}).get("score"),
                }
            )
        return out

    @app.get("/api/runs/{name}/defects")
    def defects(name: str):
        r = get_run(name)
        rv = db.get(name, r.defects)
        out = []
        for d in r.defects:
            rec = {k: d.get(k) for k in ("id", "kind", "class", "bbox_px", "bbox_um", "centroid_um", "centroid_px", "area_um2", "area_px", "score", "polygons_px")}
            for k in ("group", "risk", "risk_level", "d_jj_um", "d_trace_um", "size_um", "on_jj", "on_trace", "z", "ncc", "n_ref", "reference", "risk_factors"):
                if k in d:
                    rec[k] = d[k]
            rec["repeat_ncc"] = d.get("repeat_ncc")
            rec["p_real"] = d.get("p_real")
            rec["review"] = rv.get(d["id"], {"status": "unreviewed", "cls": None, "note": None})
            out.append(rec)
        return out

    @app.post("/api/runs/{name}/review")
    def review(name: str, body: ReviewIn):
        r = get_run(name)
        if body.status not in ("real", "false_alarm", "unsure", "unreviewed"):
            raise HTTPException(400, "bad status")
        d = next((d for d in r.defects if d["id"] == body.defect_id), None)
        if d is None:
            raise HTTPException(404, "unknown defect")
        cx, cy = _centroid_px(d)
        db.set(name, cx, cy, body.status, body.cls, body.note, body.reviewer)
        return {"ok": True}

    @app.post("/api/runs/{name}/learn")
    def learn(name: str, epochs: int = 40):
        """Train the reviewer classifier from this run's reviews and score every candidate (p_real)."""
        from chipscan.segmentation.reviewer import save_reviewer, score_candidates, train_reviewer

        r = get_run(name)
        pos = db.positions(name)
        lab = [(p["cx"], p["cy"], 1) for p in pos if p["status"] == "real"] + [(p["cx"], p["cy"], 0) for p in pos if p["status"] == "false_alarm"]
        n_pos, n_neg = sum(1 for x in lab if x[2] == 1), sum(1 for x in lab if x[2] == 0)
        if n_pos < 5 or n_neg < 5:
            raise HTTPException(400, f"need at least 5 real and 5 false-alarm reviews (have {n_pos} / {n_neg})")
        model, metrics = train_reviewer(r.source, lab, epochs=epochs)
        p = score_candidates(model, r.source, r.defects)
        for d, v in zip(r.defects, p, strict=True):
            d["p_real"] = float(v)
        save_reviewer(model, r.out_dir / f"{r.name}.reviewer.pt", metrics)
        return {"ok": True, "metrics": metrics, "n_scored": len(r.defects)}

    @app.get("/api/runs/{name}/tiles/{level}/{x}/{y}.jpg")
    def tile(name: str, level: int, x: int, y: int):
        r = get_run(name)
        if not (0 <= level < r.num_levels):
            raise HTTPException(404)
        return Response(r.image_tile(level, x, y), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/runs/{name}/design/{level}/{x}/{y}.png")
    def design_tile(name: str, level: int, x: int, y: int):
        r = get_run(name)
        return Response(r.design_tile(level, x, y), media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/runs/{name}/pred/{level}/{x}/{y}.png")
    def pred_tile(name: str, level: int, x: int, y: int):
        r = get_run(name)
        return Response(r.pred_tile(level, x, y), media_type="image/png", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/runs/{name}/thumb/{defect_id}.jpg")
    def thumb(name: str, defect_id: int, size: int = Query(96, ge=32, le=512)):
        return Response(get_run(name).thumb(defect_id, size), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

    @app.get("/api/runs/{name}/measure")
    def measure(name: str, x0: float, y0: float, x1: float, y1: float):
        return get_run(name).measure(x0, y0, x1, y1)

    @app.get("/api/runs/{name}/positions.xml")
    def positions(name: str, ids: str = Query("")):
        r = get_run(name)
        want = {int(i) for i in ids.split(",") if i.strip()} if ids else None
        if r.cfg is None:
            raise HTTPException(400, "stage frame unknown for a design-free run")
        st = r.cfg.stage
        ox, oy = st.origin_stage_um
        pos = [
            StagePosition(i, f"{d['class']}#{d['id']}", ox + d["centroid_um"][0], oy + (-d["centroid_um"][1] if st.flip_y else d["centroid_um"][1]), st.z_um)
            for i, d in enumerate(r.defects)
            if want is None or d["id"] in want
        ]
        p = r.out_dir / f"{r.name}.revisit_selected.xml"
        write_position_list(p, pos)
        return FileResponse(str(p), media_type="application/xml", filename=p.name)

    @app.get("/api/runs/{name}/export.csv")
    def export_csv(name: str):
        r = get_run(name)
        rv = db.get(name, r.defects)
        lines = ["id,kind,class,area_um2,score,x_um,y_um,review_status,review_class,note"]
        for d in r.defects:
            v = rv.get(d["id"], {})
            lines.append(f"{d['id']},{d['kind']},{d['class']},{d['area_um2']:.3f},{d['score']:.3f},{d['centroid_um'][0]:.3f},{d['centroid_um'][1]:.3f},{v.get('status', 'unreviewed')},{v.get('cls') or ''},{(v.get('note') or '').replace(',', ';')}")
        return Response("\n".join(lines) + "\n", media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={name}.reviewed.csv"})

    @app.post("/api/runs/{name}/reextract")
    def reextract(name: str, threshold: float = 0.5, min_area_px: int = 6, merge_px: int = 8, include_specks: bool = True, include_die2die: bool = True):
        r = get_run(name)
        if not r.anomaly_run:
            raise HTTPException(400, "re-extraction applies to design-free anomaly runs")
        n = r.reextract(threshold, min_area_px, merge_px, include_specks, include_die2die)
        return {"ok": True, "n": n, "params": r.extract_params}

    @app.get("/api/runs/{name}/stats")
    def stats(name: str):
        r = get_run(name)
        rv = db.get(name, r.defects)
        by_cls: dict[str, dict[str, int]] = {}
        for d in r.defects:
            c = by_cls.setdefault(d["class"], {"n": 0, "real": 0, "false_alarm": 0, "unsure": 0})
            c["n"] += 1
            s = rv.get(d["id"], {}).get("status", "unreviewed")
            if s in c:
                c[s] += 1
        by_group: dict[str, dict[str, int]] = {}
        for d in r.defects:
            g = by_group.setdefault(d.get("group") or d.get("kind", "other"), {"n": 0, "critical": 0, "high": 0, "medium": 0, "low": 0})
            g["n"] += 1
            if d.get("risk_level") in g:
                g[d["risk_level"]] += 1
        return {"by_class": by_cls, "by_group": by_group, "n": len(r.defects), "n_reviewed": sum(1 for v in rv.values() if v["status"] != "unreviewed")}

    @app.post("/api/runs/{name}/rethreshold")
    def rethreshold(name: str, speck: float = 0.5, trace: float = 0.5, min_area_px: int = 4):
        r = get_run(name)
        if not r.scan_run:
            raise HTTPException(400, "re-thresholding applies to scans")
        n = r.rethreshold(speck, trace, min_area_px)
        return {"ok": True, "n": n, "summary": r.meta.get("summary")}

    @app.post("/api/scans")
    async def add_scan(request: Request):
        """Start the flow on an uploaded image (multipart) or an image path on this machine."""
        from chipscan.stages.flow import ScanSpec

        form = await request.form()
        name = _slug(str(form.get("name") or ""))
        image_path = str(form.get("image_path") or "").strip()
        upload_dir = runs_root / "_uploads" / name
        f = form.get("image")
        if f is not None and getattr(f, "filename", None):
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / Path(f.filename).name
            with open(dest, "wb") as out:
                while chunk := await f.read(8 << 20):
                    out.write(chunk)
            image_path = str(dest)
            if not str(form.get("name") or "").strip():
                name = _slug(Path(f.filename).stem)
        if not image_path:
            raise HTTPException(400, "give an image file or a path")
        if not Path(image_path).exists():
            raise HTTPException(400, f"not found on this computer: {image_path}")
        if not str(form.get("name") or "").strip() and not getattr(f, "filename", None):
            name = _slug(Path(image_path).stem)
        if name in runs or (runs_root / name / "scan.json").exists():
            name = f"{name}_{time.strftime('%m%d_%H%M%S')}"
        gds_path = str(form.get("gds_path") or "").strip() or None
        g = form.get("gds")
        if g is not None and getattr(g, "filename", None):
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / Path(g.filename).name
            with open(dest, "wb") as out:
                while chunk := await g.read(8 << 20):
                    out.write(chunk)
            gds_path = str(dest)

        def lst(key: str) -> list[str]:
            return [t.strip() for t in str(form.get(key) or "").replace(";", ",").split(",") if t.strip()]

        px = str(form.get("pixel_size_um") or "").strip()
        spec = ScanSpec(
            name=name,
            image=image_path,
            pixel_size_um=float(px) if px else None,
            gds=gds_path,
            gds_cell=str(form.get("gds_cell") or "").strip() or None,
            trace_layers=lst("trace_layers"),
            component_layers=lst("component_layers"),
            straighten=str(form.get("straighten") or "true").lower() != "false",
        )
        if spec.gds and not (spec.trace_layers or spec.component_layers):
            raise HTTPException(400, "with a GDS, name its trace and/or component layers (e.g. 1/0 and 92/0)")
        return jobs.submit(spec)

    @app.get("/api/jobs")
    def list_jobs():
        return jobs.list()

    watcher = Watcher(runs_root, jobs)

    @app.get("/api/watch")
    def list_watch():
        return watcher.list()

    @app.post("/api/watch")
    async def add_watch(request: Request):
        body = await request.json()
        try:
            px = body.get("pixel_size_um")
            return watcher.add(str(body.get("path") or ""), float(px) if px else None, bool(body.get("straighten", True)), bool(body.get("include_existing", False)))
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/watch")
    def remove_watch(path: str):
        watcher.remove(path)
        return {"ok": True}

    @app.get("/api/health")
    def health():
        return JSONResponse({"ok": True, "runs": list(runs)})

    return app


def _png_bytes(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes()


def serve(config_paths: list[Path], host: str = "127.0.0.1", port: int = 8000, db_path: Path | None = None, runs_root: Path | None = None) -> None:
    import uvicorn

    app = create_app(config_paths, db_path, runs_root)
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["Run", "ReviewDB", "create_app", "serve", "io"]

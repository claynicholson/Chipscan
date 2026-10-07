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
from fastapi import FastAPI, HTTPException, Query
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
        if self.pred is not None and level <= 4:
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
def create_app(config_paths: list[Path], db_path: Path | None = None) -> FastAPI:
    runs: dict[str, Run] = {}
    for p in config_paths:
        p = Path(p)
        r = Run.load_anomaly(p) if p.name.endswith(".anomalies.json") else Run.load(p)
        runs[r.name] = r
    if not runs:
        raise ValueError("no runs loaded")
    db = ReviewDB(db_path or (next(iter(runs.values())).out_dir / "reviews.sqlite"))
    app = FastAPI(title="chipscan viewer")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

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
                    "segmenter": r.cfg.segmentation.method if r.cfg else "anomaly",
                    "registration": r.warp.affine.to_dict(),
                    "anomaly_run": r.anomaly_run,
                    "extract_params": r.extract_params,
                    "n_specks": len(r.specks),
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
    def thumb(name: str, defect_id: int):
        return Response(get_run(name).thumb(defect_id), media_type="image/jpeg", headers={"Cache-Control": "max-age=86400"})

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
        return {"by_class": by_cls, "n": len(r.defects), "n_reviewed": sum(1 for v in rv.values() if v["status"] != "unreviewed")}

    @app.get("/api/health")
    def health():
        return JSONResponse({"ok": True, "runs": list(runs)})

    return app


def _png_bytes(rgba: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", rgba)
    return buf.tobytes()


def serve(config_paths: list[Path], host: str = "127.0.0.1", port: int = 8000, db_path: Path | None = None) -> None:
    import uvicorn

    app = create_app(config_paths, db_path)
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["Run", "ReviewDB", "create_app", "serve", "io"]

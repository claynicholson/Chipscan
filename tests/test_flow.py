"""End-to-end plumbing of the scan flow: tilt + tiling correction, maps, three groups, risk, viewer API."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from chipscan.io.readers import ArraySource
from chipscan.segmentation.unet import UNet
from chipscan.stages.components import component_errors, find_instances
from chipscan.stages.flow import ScanSpec, run_scan
from chipscan.stages.straighten import estimate_tile_grid, estimate_tilt


def _chip(h=1600, w=2000, tilt=0.4, seed=0):
    """CPW-ish lines with JJ dumbbells every 60 px, tilted, with per-frame vignetting."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 150, np.float32)
    for y in range(120, h - 120, 160):
        cv2.rectangle(img, (80, y), (w - 80, y + 3), 70, -1)
        cv2.rectangle(img, (80, y + 14), (w - 80, y + 17), 70, -1)
        for x in range(140, w - 140, 60):
            cv2.rectangle(img, (x - 5, y - 16), (x + 5, y - 6), 230, -1)
            cv2.rectangle(img, (x - 5, y + 24), (x + 5, y + 34), 230, -1)
            cv2.rectangle(img, (x - 1, y - 6), (x + 1, y + 24), 230, -1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -tilt, 1)
    img = cv2.warpAffine(img, m, (w, h), borderValue=150)
    yy, xx = np.mgrid[0:h, 0:w]
    u, v = (xx % 500) / 500 - 0.5, (yy % 375) / 375 - 0.5
    img = img * (1 - 0.1 * (u * u + v * v) * 2) + rng.normal(0, 2, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_tilt_and_tiling():
    src = ArraySource(_chip(2400, 3200), 1.0)
    t = estimate_tilt(src)
    assert abs(t["angle_deg"] - 0.4) < 0.02
    g = estimate_tile_grid(src)
    assert g.found
    assert abs(g.period_px[0] - 500) < 10 and abs(g.period_px[1] - 375) < 10


def test_component_lines_find_a_damaged_junction():
    img = _chip(tilt=0.0)
    cls = np.zeros(img.shape, np.uint8)
    cls[img > 200] = 2
    insts = find_instances(cls, min_area_px=10)
    assert len(insts) > 50
    target = sorted(insts, key=lambda d: (d["cy"], d["cx"]))[len(insts) // 2]
    bad = img.copy()
    cv2.circle(bad, (int(target["cx"]), int(target["cy"]) - 8), 3, 60, -1)
    recs, summ = component_errors(ArraySource(bad, 1.0), insts, 1.0)
    assert summ["n_lines"] >= 1
    hits = [r for r in recs if abs(r["centroid_px"][0] - target["cx"]) < 12 and abs(r["centroid_px"][1] - target["cy"]) < 25]
    assert hits, recs[:5]
    assert len(recs) <= 3


@pytest.fixture()
def tiny_models(tmp_path):
    torch.manual_seed(0)
    paths = {}
    for name, out in (("layout", 4), ("speck", 1), ("trace", 1)):
        p = tmp_path / f"{name}.pt"
        UNet(1, 4, 2, out_channels=out).save(p, {"task": name})
        paths[name] = str(p)
    return paths


def test_run_scan_end_to_end(tmp_path, tiny_models):
    img = _chip()
    ip = tmp_path / "scan.png"
    cv2.imwrite(str(ip), img)
    spec = ScanSpec(
        name="t1", image=str(ip), pixel_size_um=0.8,
        layout_model=tiny_models["layout"], speck_model=tiny_models["speck"], trace_model=tiny_models["trace"], device="cpu",
    )
    seen = []
    run_dir = run_scan(spec, tmp_path / "runs", lambda st, f, m: seen.append(st))
    meta = json.loads((run_dir / "scan.json").read_text())
    assert meta["status"] == "done"
    assert abs(meta["stages"]["straighten"]["angle_deg"] - 0.4) < 0.03
    assert {"ingest", "straighten", "layout", "specks", "traces", "components", "risk", "export"} <= set(seen)
    rec = json.loads((run_dir / "findings.json").read_text())
    for d in rec["findings"]:
        assert d["group"] in ("speck", "trace", "component")
        assert 0 <= d["risk"] <= 100 and d["risk_level"] in ("critical", "high", "medium", "low")

    # the viewer loads the scan folder and serves its groups
    from fastapi.testclient import TestClient

    from chipscan.web.server import create_app

    app = create_app([], tmp_path / "reviews.sqlite", tmp_path / "runs")
    c = TestClient(app)
    runs = c.get("/api/runs").json()
    assert runs[0]["scan_run"] and runs[0]["name"] == "t1"
    assert c.get("/api/runs/t1/tiles/0/0/0.jpg").status_code == 200
    assert c.get("/api/runs/t1/design/2/0/0.png").status_code == 200  # layout overlay
    st = c.get("/api/runs/t1/stats").json()
    assert "by_group" in st
    r = c.post("/api/runs/t1/rethreshold?speck=0.3&trace=0.3&min_area_px=4").json()
    assert r["ok"]

"""Viewer API on a small synthetic run."""

from fastapi.testclient import TestClient

from chipscan.config import PipelineConfig
from chipscan.pipeline import run_inspection
from chipscan.synth import make_synthetic
from chipscan.web.server import create_app


def test_viewer_api(tmp_path):
    paths = make_synthetic(tmp_path, 2.0, 1.0, False, 1.0, 1200.0, 10, seed=5)
    cfg = PipelineConfig.load(paths["config"])
    cfg.registration.fine.grid = 4
    cfg.dump(paths["config"])
    run_inspection(cfg)
    app = create_app([paths["config"]], tmp_path / "reviews.sqlite", tmp_path / "runs")
    c = TestClient(app)
    assert c.get("/").status_code == 200
    runs = c.get("/api/runs").json()
    assert len(runs) == 1 and runs[0]["name"] == "synthetic" and runs[0]["levels"] >= 2
    n = runs[0]["levels"]
    for lvl in (0, n - 1):
        r = c.get(f"/api/runs/synthetic/tiles/{lvl}/0/0.jpg")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg" and len(r.content) > 500
    assert c.get("/api/runs/synthetic/design/0/2/2.png").headers["content-type"] == "image/png"
    assert c.get("/api/runs/synthetic/pred/1/0/0.png").status_code == 200
    defs = c.get("/api/runs/synthetic/defects").json()
    assert defs and all(d["review"]["status"] == "unreviewed" for d in defs)
    d0 = defs[0]
    assert c.get(f"/api/runs/synthetic/thumb/{d0['id']}.jpg").status_code == 200
    r = c.post("/api/runs/synthetic/review", json={"defect_id": d0["id"], "status": "false_alarm", "cls": "particle", "reviewer": "t"})
    assert r.json()["ok"]
    defs2 = c.get("/api/runs/synthetic/defects").json()
    assert next(d for d in defs2 if d["id"] == d0["id"])["review"]["status"] == "false_alarm"
    st = c.get("/api/runs/synthetic/stats").json()
    assert st["n_reviewed"] == 1
    csv = c.get("/api/runs/synthetic/export.csv").text
    assert "false_alarm" in csv and csv.startswith("id,kind,class")
    b = d0["bbox_px"]
    m = c.get("/api/runs/synthetic/measure", params={"x0": b["x0"] - 30, "y0": b["y0"], "x1": b["x1"] + 30, "y1": b["y0"]}).json()
    assert m["runs"] and m["length_um"] > 0
    xml = c.get("/api/runs/synthetic/positions.xml", params={"ids": str(d0["id"])})
    assert xml.status_code == 200 and "positionList" in xml.content.decode("utf-16")

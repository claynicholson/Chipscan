"""Design-free anomaly detector: dataset, short training, inference, outline extraction."""

import numpy as np
import torch

from chipscan.io.readers import ArraySource
from chipscan.segmentation.anomaly import (
    AnomalyDetector,
    AnomalyInjection,
    AnomalyTiles,
    anomalies_from_map,
)
from chipscan.segmentation.anomaly_train import evaluate, train_anomaly
from chipscan.segmentation.unet import UNet


def _scene(seed=0, size=700):
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 0.6, np.float32)
    for y in range(40, size, 60):  # stripes with periodic boxes = "structure"
        img[y : y + 12] = 0.3
        for x in range(30, size, 90):
            img[y - 8 : y + 20, x : x + 10] = 0.75
    img += rng.normal(0, 0.02, img.shape)
    return ArraySource((np.clip(img, 0, 1) * 255).astype(np.uint8), 1.0)


def test_dataset_injects_and_labels():
    ds = AnomalyTiles([_scene(), _scene(1)], 96, 20, AnomalyInjection(p_anomaly_tile=1.0, max_per_tile=2), augment=True, seed=0)
    x, y = ds[0]
    assert x.shape == (1, 96, 96) and y.shape == (1, 96, 96)
    n_pos = sum(int(ds[i][1].sum() > 0) for i in range(12))
    assert n_pos >= 8
    # unlabelled pixels are (almost) untouched: check on a non-augmented sample
    ds2 = AnomalyTiles([_scene()], 96, 4, AnomalyInjection(p_anomaly_tile=1.0, max_per_tile=1, p_types={"bar": 1.0}), augment=False, seed=2)
    img, k = ds2._draw_tile()
    out, t = ds2.inject(img, k)
    assert t.sum() > 0 and np.abs(out - img)[t > 0].mean() > 0.05


def test_train_and_detect(tmp_path):
    ds = AnomalyTiles([_scene()], 64, 32, seed=0)
    vs = AnomalyTiles([_scene(5)], 64, 8, augment=False, seed=1)
    model, hist = train_anomaly(ds, tmp_path / "a.pt", epochs=2, batch_size=8, base=8, depth=2, device="cpu", val_dataset=vs)
    assert isinstance(model, UNet) and model.out_channels == 1 and len(hist) == 2
    m = evaluate(model, vs, torch.device("cpu"), 8)
    assert 0 <= m["iou"] <= 1 and 0 <= m["clean_tile_fa_rate"] <= 1
    det = AnomalyDetector(tmp_path / "a.pt", device="cpu")
    src = _scene(7, 300)
    amap = det.run(src, tile_px=128, margin_px=16, progress=False)
    assert amap.shape == src.shape and amap.dtype == np.uint8


def test_anomalies_from_map_outlines_and_chunking():
    amap = np.zeros((300, 300), np.uint8)
    amap[20:30, 20:40] = 255
    amap[250:262, 100:110] = 200
    amap[2:4, 2:4] = 255  # too small
    recs = anomalies_from_map(amap, 0.5, min_area_px=6, pixel_size_um=0.5, chunk=128)
    assert len(recs) == 2
    r = sorted(recs, key=lambda r: r["bbox_px"]["y0"])[0]
    assert r["area_px"] == 200 and abs(r["area_um2"] - 50) < 1e-6 and r["polygons_px"]

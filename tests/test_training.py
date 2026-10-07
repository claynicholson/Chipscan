"""Dataset construction, defect injection, and a short training run on a synthetic scene."""

import numpy as np
import torch

from chipscan.config import PipelineConfig
from chipscan.pipeline import load_scene
from chipscan.segmentation.dataset import InjectionConfig, InspectionTiles, split_scene
from chipscan.segmentation.train import evaluate, inspection_loss, train
from chipscan.segmentation.unet import UNet
from chipscan.synth import make_synthetic


def _scene(tmp_path):
    paths = make_synthetic(tmp_path, 1.5, 1.0, False, 1.0, 1200.0, 0, seed=3)
    cfg = PipelineConfig.load(paths["config"])
    cfg.registration.fine.grid = 4
    return load_scene(cfg)


def test_dataset_shapes_and_injection(tmp_path):
    sc = _scene(tmp_path)
    ds = InspectionTiles([sc], tile_px=128, n_samples=16, use_prior=True, injection=InjectionConfig(p_defect_tile=1.0, max_per_tile=3), seed=0)
    x, y, w = ds[0]
    assert x.shape == (2, 128, 128) and y.shape == (3, 128, 128) and w.shape == (1, 128, 128)
    # over a few samples some injected defects must appear, and only where the design allows them
    n_extra = n_missing = 0
    for i in range(12):
        x, y, w = ds[i]
        metal, extra, missing = y[0].numpy(), y[1].numpy(), y[2].numpy()
        n_extra += extra.sum() > 0
        n_missing += missing.sum() > 0
        # injected extra metal is now metal; injected missing metal is now gap
        assert (metal[extra > 0] == 1).all() and (metal[missing > 0] == 0).all()
    assert n_extra > 0 and n_missing > 0


def test_injection_changes_image_where_marked(tmp_path):
    sc = _scene(tmp_path)
    ds = InspectionTiles([sc], tile_px=128, n_samples=4, augment=False, prior_jitter_px=0.0, injection=InjectionConfig(p_defect_tile=1.0, max_per_tile=2, min_radius_px=4, max_radius_px=8), seed=1)
    img, lab, _ = ds._draw_tile()
    img2, metal, extra, missing = ds.inject(img, lab)
    changed = (extra > 0) | (missing > 0)
    assert changed.any()
    assert np.abs(img2 - img)[changed].mean() > 0.05
    # blobs have soft edges, so allow a few pixels of halo; beyond that the image is untouched
    from scipy import ndimage as ndi

    halo = ndi.binary_dilation(changed, iterations=3)
    assert np.abs(img2 - img)[~halo].max() < 1e-2  # only the Gaussian tail remains


def test_loss_and_short_training(tmp_path):
    sc = _scene(tmp_path)
    tr, va = split_scene(sc, 0.2)
    ds = InspectionTiles([tr], tile_px=64, n_samples=24, seed=0)
    vs = InspectionTiles([va], tile_px=64, n_samples=8, augment=False, seed=1)
    logits = torch.randn(2, 3, 64, 64)
    y = torch.zeros(2, 3, 64, 64)
    loss, parts = inspection_loss(logits, y, torch.ones(2, 1, 64, 64))
    assert loss.item() > 0 and set(parts) == {"metal", "defect"}
    model, hist = train(ds, tmp_path / "m.pt", epochs=2, batch_size=4, base=8, depth=2, device="cpu", val_dataset=vs)
    assert isinstance(model, UNet) and len(hist.epochs) == 2
    assert (tmp_path / "m.pt").exists() and (tmp_path / "m.history.json").exists()
    m = evaluate(model, vs, torch.device("cpu"), 4)
    assert 0.0 <= m["defect_iou"] <= 1.0

import numpy as np
import torch

from chipscan.segmentation.unet import UNet, UNetSegmenter, _pad_to_multiple


def test_unet_forward_shapes():
    m = UNet(1, base=8, depth=3, out_channels=1)
    y = m(torch.rand(2, 1, 96, 80))
    assert y.shape == (2, 1, 96, 80)
    m2 = UNet(2, base=8, depth=2, out_channels=3)
    assert m2(torch.rand(1, 2, 64, 64)).shape == (1, 3, 64, 64)


def test_pad_to_multiple():
    x = torch.rand(1, 1, 70, 65)
    assert _pad_to_multiple(x, 16).shape[-2:] == (80, 80)


def test_segmenter_roundtrip_plain(tmp_path):
    m = UNet(1, base=8, depth=2, out_channels=1)
    p = tmp_path / "m.pt"
    m.save(p)
    seg = UNetSegmenter(p, device="cpu", batch_size=2)
    assert not seg.predicts_defects and not seg.uses_design_prior
    out = seg.predict(np.random.rand(3, 50, 70).astype(np.float32))
    assert out.shape == (3, 50, 70) and out.min() >= 0 and out.max() <= 1


def test_segmenter_defect_head(tmp_path):
    m = UNet(2, base=8, depth=2, out_channels=3)
    p = tmp_path / "m.pt"
    m.save(p)
    seg = UNetSegmenter(p, device="cpu", batch_size=2)
    assert seg.predicts_defects and seg.uses_design_prior
    tiles = np.random.rand(2, 40, 40).astype(np.float32)
    priors = (np.random.rand(2, 40, 40) > 0.5).astype(np.uint8)
    metal = seg.predict(tiles, priors)
    extra, missing = seg.predict_defects(tiles, priors)
    assert metal.shape == extra.shape == missing.shape == (2, 40, 40)

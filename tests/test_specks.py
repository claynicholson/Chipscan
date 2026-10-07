import numpy as np

from chipscan.io.readers import ArraySource
from chipscan.segmentation.specks import find_specks, find_specks_tile


def test_faint_speck_found_noise_ignored():
    rng = np.random.default_rng(0)
    img = np.full((400, 400), 0.55, np.float32) + rng.normal(0, 0.006, (400, 400)).astype(np.float32)
    yy, xx = np.mgrid[0:400, 0:400]
    img -= 0.03 * np.exp(-((xx - 120) ** 2 + (yy - 300) ** 2) / (2 * 6.0**2))  # faint 12 px smudge, 8 grey levels
    img[50:60, 200:300] = 0.2  # a designed dark bar: strong edge, must be excluded
    mask, z = find_specks_tile(np.clip(img, 0, 1))
    assert mask[300, 120] and z[300, 120] > 7
    assert not mask[45:65, 195:305].any()
    src = ArraySource((np.clip(img, 0, 1) * 255).astype(np.uint8), 1.0)
    recs = find_specks(src, tile_px=256, margin_px=32, progress=False)
    assert any(abs(r["centroid_px"][0] - 120) < 4 and abs(r["centroid_px"][1] - 300) < 4 for r in recs)
    assert len(recs) <= 3

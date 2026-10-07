import numpy as np

from chipscan.io.readers import ArraySource
from chipscan.segmentation.repeats import estimate_pitches, suppress_repeats


def test_pitch_and_suppression():
    rng = np.random.default_rng(0)
    img = np.full((900, 900), 0.6, np.float32)
    for y in range(60, 900, 150):  # rows of identical "cells" every 150 px, cells every 120 px in x
        for x in range(40, 900, 120):
            img[y : y + 30, x : x + 50] = 0.3
            img[y + 10 : y + 20, x + 20 : x + 30] = 0.8
    img[400:412, 500:512] = 0.95  # a unique blob (defect)
    img += rng.normal(0, 0.01, img.shape)
    src = ArraySource((np.clip(img, 0, 1) * 255).astype(np.uint8), 1.0)
    p = estimate_pitches(src, max_level_px=900)
    assert any(abs(v - 120) < 3 for v in p["x"]) and any(abs(v - 150) < 3 for v in p["y"])
    cands = [
        {"centroid_px": [65, 75], "area_px": 300, "score": 0.9},  # a designed cell
        {"centroid_px": [506, 406], "area_px": 144, "score": 0.9},  # the unique blob
    ]
    kept, dropped = suppress_repeats(src, cands, p, half_px=24)
    assert len(dropped) == 1 and dropped[0]["centroid_px"] == [65, 75]
    assert len(kept) == 1 and kept[0]["centroid_px"] == [506, 406]

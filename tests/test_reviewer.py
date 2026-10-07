import numpy as np

from chipscan.io.readers import ArraySource
from chipscan.segmentation.reviewer import crop_pair, score_candidates, train_reviewer


def test_reviewer_learns_a_simple_rule():
    rng = np.random.default_rng(0)
    img = np.full((1200, 1200), 0.6, np.float32) + rng.normal(0, 0.01, (1200, 1200)).astype(np.float32)
    labelled = []
    for i in range(60):
        cx, cy = int(rng.uniform(100, 1100)), int(rng.uniform(100, 1100))
        if i % 2 == 0:  # "real": dark blob
            img[cy - 4 : cy + 4, cx - 4 : cx + 4] = 0.2
            labelled.append((cx, cy, 1))
        else:  # "false alarm": bright blob
            img[cy - 4 : cy + 4, cx - 4 : cx + 4] = 0.95
            labelled.append((cx, cy, 0))
    src = ArraySource((np.clip(img, 0, 1) * 255).astype(np.uint8), 1.0)
    assert crop_pair(src, 500, 500).shape == (2, 64, 64)
    model, m = train_reviewer(src, labelled, epochs=25, device="cpu", seed=1)
    assert m["val_acc"] >= 0.8
    p = score_candidates(model, src, [{"centroid_px": [cx, cy]} for cx, cy, _ in labelled], device="cpu")
    y = np.array([lab for _, _, lab in labelled])
    assert ((p > 0.5) == (y == 1)).mean() >= 0.85

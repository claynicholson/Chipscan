"""End-to-end on a small synthetic chip: tilt + scale + flip must be recovered and
injected defects found with the right classes."""

import json

import numpy as np
import pytest

from chipscan.config import PipelineConfig
from chipscan.pipeline import run_inspection
from chipscan.synth import make_synthetic


@pytest.mark.parametrize("rotation,scale,flip", [(3.7, 1.02, False), (-91.0, 0.985, True)])
def test_end_to_end(tmp_path, rotation, scale, flip):
    paths = make_synthetic(tmp_path, rotation, scale, flip, 1.0, 1800.0, 25, seed=1)
    cfg = PipelineConfig.load(paths["config"])
    res = run_inspection(cfg)
    truth = json.loads(paths["truth"].read_text())
    T = np.array(truth["world_to_px"]["matrix"])
    pts = np.array([[0, 0], [1800, 0], [0, 1800], [1800, 1800], [900, 900]], dtype=float)
    est = res.warp.world_to_px(pts)
    tru = (np.hstack([pts, np.ones((5, 1))]) @ T.T)[:, :2]
    assert np.hypot(*(est - tru).T).max() < 2.0, "registration error > 2 px"
    assert abs(res.warp.affine.rotation_deg - rotation) < 0.1
    assert abs(res.warp.affine.scale_factor - scale) < 0.005
    assert res.warp.affine.is_flipped == flip

    found = 0
    right_class = 0
    for td in truth["defects"]:
        near = [d for d in res.defects if np.hypot(d.centroid_um[0] - td["x_um"], d.centroid_um[1] - td["y_um"]) < max(6.0, td["size_um"])]
        if near:
            found += 1
            if any(d.cls == td["class"] for d in near):
                right_class += 1
    n = len(truth["defects"])
    assert found >= 0.8 * n, f"recall {found}/{n}"
    assert right_class >= 0.7 * n, f"class accuracy {right_class}/{n}"
    # false alarms: everything not near a truth defect
    fa = [d for d in res.defects if not any(np.hypot(d.centroid_um[0] - td["x_um"], d.centroid_um[1] - td["y_um"]) < max(6.0, td["size_um"]) for td in truth["defects"])]
    assert len(fa) <= 0.3 * n + 3, f"{len(fa)} false alarms"
    assert (cfg.export.out_dir / "synthetic.lyrdb").exists()
    assert len(res.cuts) == 2 and all(abs(e) < 2.0 for c in res.cuts for e in c.error_um)

import numpy as np

from chipscan.config import DetectionConfig, MetrologyCut
from chipscan.detection import (
    EXTRA,
    MISSING,
    RuleClassifier,
    compare_tile,
    merge_defects,
    runs_from_profile,
)
from chipscan.detection.metrology import measure_cut
from chipscan.geometry import Affine2D, BBox, WarpField


def _cpw(h=100, w=200):
    """Design: ground plane with a horizontal CPW (gap 10, centre 10) through the middle."""
    d = np.ones((h, w), dtype=np.uint8)
    d[35:45] = 0
    d[55:65] = 0
    return d


def test_short_open_particle_void_classes():
    d = _cpw()
    prob = d.astype(np.float32)
    prob[35:45, 50:53] = 1.0  # short: bridges gap between centre and ground
    prob[30:70, 120:124] = 0.0  # open: cuts the centre conductor (and the gaps, harmless)
    prob[38:42, 160:164] = 1.0  # particle in the gap, touches nothing
    prob[10:14, 20:24] = 0.0  # void in ground plane
    cfg = DetectionConfig(tolerance_px=1, min_area_um2=0.0, min_area_px=2)
    defs = compare_tile(prob, d, (0, 0), WarpField(Affine2D.identity()), 1.0, cfg, 0, RuleClassifier(tolerance_px=1))
    classes = sorted((x.cls, x.kind) for x in defs)
    assert ("short", EXTRA) in classes
    assert ("open", MISSING) in classes
    assert ("particle", EXTRA) in classes
    assert ("void", MISSING) in classes


def test_tolerance_band_suppresses_edge_slop():
    d = _cpw()
    prob = np.roll(d, 1, axis=0).astype(np.float32)  # 1 px misregistration everywhere
    cfg = DetectionConfig(tolerance_px=2, min_area_um2=0.0, min_area_px=2)
    defs = compare_tile(prob, d, (0, 0), WarpField(Affine2D.identity()), 1.0, cfg)
    assert defs == []


def test_merge_across_tiles():
    from chipscan.detection.defects import Defect

    a = Defect(0, EXTRA, "particle", BBox(0, 0, 5, 5), (2.5, 2.5), 25, 25, 1.0, [], BBox(0, 0, 5, 5), 0)
    b = Defect(1, EXTRA, "short", BBox(5.5, 0, 10, 5), (7.5, 2.5), 20, 20, 1.0, [], BBox(5.5, 0, 10, 5), 1)
    c = Defect(2, MISSING, "void", BBox(5.5, 0, 10, 5), (7.5, 2.5), 20, 20, 1.0, [], None, 1)
    m = merge_defects([a, b, c], margin_um=1.0)
    assert len(m) == 2
    big = next(x for x in m if x.kind == EXTRA)
    assert big.cls == "short" and big.area_px == 45 and big.bbox_um == BBox(0, 0, 10, 5)


def test_runs_from_profile_subpixel():
    t = np.arange(0, 40, 1.0)
    v = np.where((t >= 10.5) & (t < 25.5), 1.0, 0.0)
    v[10], v[25] = 0.5, 0.5
    runs, edges = runs_from_profile(t, v)
    assert len(edges) == 2 and abs(edges[0] - 10) < 1e-6
    assert runs[1][0] == "metal"


def test_measure_cut_cpw():
    d = _cpw()
    prob = d.astype(np.float32)
    warp = WarpField(Affine2D.identity())
    cut = MetrologyCut(name="c", p0_um=[100, 20], p1_um=[100, 80], expect_um=[10, 10, 10])
    res = measure_cut(cut, lambda b: prob[int(b.y0) : int(b.y1), int(b.x0) : int(b.x1)] if b.x0 >= 0 and b.y0 >= 0 else _pad(prob, b), warp, 1.0)
    phases = [p for p, _ in res.runs]
    assert phases == ["metal", "gap", "metal", "gap", "metal"]
    assert all(abs(e) < 1.0 for e in res.error_um)


def _pad(prob, b):
    x0, y0, x1, y1 = b.to_int()
    out = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x1, prob.shape[1]), min(y1, prob.shape[0])
    out[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = prob[cy0:cy1, cx0:cx1]
    return out

import numpy as np

from chipscan.geometry import Affine2D, BBox, WarpField


def test_similarity_roundtrip():
    a = Affine2D.from_similarity(1.07, 12.5, 100.0, -40.0)
    assert abs(a.scale_factor - 1.07) < 1e-9
    assert abs(a.rotation_deg - 12.5) < 1e-9
    assert not a.is_flipped
    p = np.array([[3.0, 4.0], [-1.0, 2.0]])
    q = a.inverse().apply(a.apply(p))
    assert np.allclose(p, q)


def test_flip_decomposition():
    a = Affine2D.from_similarity(2.0, -30.0, 0, 0, flip_x=True)
    assert a.is_flipped
    assert abs(a.scale_factor - 2.0) < 1e-9
    assert abs(a.rotation_deg + 30.0) < 1e-9


def test_fit_recovers_affine():
    truth = Affine2D.from_similarity(0.95, 3.3, 20, 30)
    src = np.random.default_rng(0).uniform(-100, 100, (20, 2))
    dst = truth.apply(src)
    est = Affine2D.fit(src, dst)
    assert np.allclose(est.m, truth.m, atol=1e-8)


def test_bbox_ops():
    a, b = BBox(0, 0, 10, 10), BBox(5, 5, 20, 20)
    assert a.intersection(b) == BBox(5, 5, 10, 10)
    assert a.union(b) == BBox(0, 0, 20, 20)
    assert a.overlaps(b) and not a.overlaps(BBox(11, 0, 12, 1))
    assert a.overlaps(BBox(11, 0, 12, 1), margin=2)


def test_warpfield_inverse_with_residual():
    aff = Affine2D.from_similarity(1.0, 2.0, 50, 50)
    gx, gy = np.array([0.0, 500.0, 1000.0]), np.array([0.0, 500.0, 1000.0])
    dx = np.array([[0, 1, 2], [0, 1.5, 3], [0, 2, 4]], dtype=float)
    dy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=float)
    w = WarpField(aff, gx, gy, dx, dy)
    pts = np.array([[100.0, 100.0], [400.0, 700.0], [900.0, 20.0]])
    px = w.world_to_px(pts)
    back = w.px_to_world(px)
    assert np.allclose(back, pts, atol=1e-3)
    # local affine approximates the warp inside a box
    box = BBox(300, 300, 600, 600)
    la = w.local_affine(box)
    c = np.array([[450.0, 450.0]])
    wc = w.px_to_world(c)
    assert np.allclose(la.apply(wc), c, atol=0.05)
    d = WarpField.from_dict(w.to_dict())
    assert np.allclose(d.world_to_px(pts), px)

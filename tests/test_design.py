import gdstk
import numpy as np

from chipscan.design import Design, list_layers
from chipscan.geometry import Affine2D, BBox


def _square_gds(path, polarity_layers=((1, 0),)):
    lib = gdstk.Library()
    c = lib.new_cell("TOP")
    for layer, dt in polarity_layers:
        c.add(gdstk.rectangle((10, 10), (30, 20), layer=layer, datatype=dt))
    c.add(gdstk.rectangle((0, 0), (5, 5), layer=7, datatype=0))
    lib.write_gds(str(path))


def test_load_and_layers(tmp_path):
    p = tmp_path / "a.gds"
    _square_gds(p)
    assert list_layers(p) == {(1, 0): 1, (7, 0): 1}
    d = Design.load(p, layers=["1/0"])
    assert d.num_polygons == 1
    assert d.bbox == BBox(10, 10, 30, 20)


def test_render_positive_identity_scale(tmp_path):
    p = tmp_path / "a.gds"
    _square_gds(p)
    d = Design.load(p, layers=["1/0"])
    a = Affine2D.scaling(2.0)  # 2 px per um
    m = d.render(a, BBox(0, 0, 80, 60))
    assert m.shape == (60, 80)
    assert m[25:40, 20:60].all()
    assert m[:20].sum() == 0 and m[40:].sum() == 0
    assert abs(m.sum() - 20 * 10 * 4) <= 2 * (40 + 20)


def test_render_negative_polarity(tmp_path):
    p = tmp_path / "a.gds"
    _square_gds(p)
    d = Design.load(p, layers=["1/0"], polarity="drawn_is_etch")
    m = d.render(Affine2D.identity(), BBox(0, 0, 40, 30))
    # region is the design bbox [10,30]x[10,20]; drawn rectangle is etched -> no metal anywhere
    assert m.sum() == 0
    d2 = Design.from_polygons([np.array([[0, 0], [40, 0], [40, 30], [0, 30.0]]), np.array([[10, 10], [30, 10], [30, 20], [10, 20.0]])], polarity="drawn_is_etch")
    d2.polygons = d2.polygons[1:]
    d2._index()
    m2 = d2.render(Affine2D.identity(), BBox(0, 0, 40, 30))
    assert m2[0:10].all() and m2[12:18, 12:28].sum() == 0


def test_render_rotated_matches_point_test(tmp_path):
    p = tmp_path / "a.gds"
    _square_gds(p)
    d = Design.load(p, layers=["1/0"])
    a = Affine2D.from_similarity(3.0, 30.0, 40, 10)
    m = d.render(a, BBox(0, 0, 200, 150))
    c = a.apply(np.array([[20.0, 15.0]]))[0]  # rectangle centre
    assert m[int(c[1]), int(c[0])] == 1
    o = a.apply(np.array([[5.0, 5.0]]))[0]
    assert m[int(o[1]), int(o[0])] == 0

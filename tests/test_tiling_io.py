import numpy as np

from chipscan.geometry import BBox
from chipscan.io.positions import StagePosition, read_position_list, write_position_list
from chipscan.io.readers import ArraySource, downsample2
from chipscan.tiling import make_tiles


def test_tiles_partition_exactly():
    tiles = make_tiles((1000, 1500), 400, 32)
    cover = np.zeros((1000, 1500), dtype=int)
    for t in tiles:
        x0, y0, x1, y1 = t.core.to_int()
        cover[y0:y1, x0:x1] += 1
        assert t.read == t.core.pad(32)
    assert (cover == 1).all()


def test_tiles_in_region():
    tiles = make_tiles((1000, 1000), 300, 0, BBox(100.5, 200.2, 700, 650))
    assert all(t.core.x0 >= 100 and t.core.y1 <= 650 for t in tiles)


def test_array_source_levels_and_padding():
    a = np.arange(600 * 800, dtype=np.float64).reshape(600, 800) % 251
    s = ArraySource(a.astype(np.uint8), pixel_size_um=0.5)
    assert s.num_levels == 3 and s.level_shape(1) == (300, 400)
    r = s.read_region(-10, -10, 20, 20)
    assert r.shape == (30, 30) and r[:10].sum() == 0 and (r[10:, 10:] == s.levels[0][:20, :20]).all()
    r1 = s.read_region(0, 0, 100, 100, level=1)
    assert r1.shape == (50, 50)
    assert s.best_level_for(1.0) == 1 and s.best_level_for(0.6) == 0


def test_downsample2():
    a = np.array([[0, 2, 4, 6], [2, 4, 6, 8]], dtype=np.uint8)
    assert (downsample2(a) == np.array([[2, 6]])).all()


def test_position_list_roundtrip(tmp_path):
    pos = [StagePosition(0, "Position 0", 98506.83, 112995.4, 34861.08, 7167.5, 6844.0, True), StagePosition(1, "short#3", 1.5, 2.5, None)]
    p = tmp_path / "pl.xml"
    write_position_list(p, pos)
    back = read_position_list(p)
    assert len(back) == 2
    rect = next(b for b in back if b.is_rect)
    assert abs(rect.width - 7167.5) < 1e-6 and rect.single
    pt = next(b for b in back if not b.is_rect)
    assert pt.name == "short#3" and pt.x == 1.5

import numpy as np
import tifffile

from chipscan import export_image as E


class Pyr:
    """A small in-memory pyramid with the ImageSource calls the exporter uses."""

    def __init__(self, a):
        self.levels = [a]
        while min(self.levels[-1].shape) > 64:
            self.levels.append(self.levels[-1][::2, ::2].copy())

    shape = property(lambda self: self.levels[0].shape)
    num_levels = property(lambda self: len(self.levels))

    def level_shape(self, L):
        return self.levels[L].shape

    def read_level(self, L):
        return self.levels[L]

    def read_region(self, x0, y0, x1, y1, level=0):
        s = 2**level
        a = self.levels[level]
        out = np.zeros((-(-y1 // s) - y0 // s, -(-x1 // s) - x0 // s), np.uint8)
        sub = a[max(0, y0 // s) : -(-y1 // s), max(0, x0 // s) : -(-x1 // s)]
        out[: sub.shape[0], : sub.shape[1]] = sub
        return out


def _finding(i, x, y, size, level):
    poly = [[x, y], [x + size, y], [x + size, y + size], [x, y + size]]
    return {"id": i, "risk_level": level, "bbox_px": {"x0": x, "y0": y, "x1": x + size, "y1": y + size}, "polygons_px": [poly]}


def test_exports(tmp_path):
    src = Pyr(np.full((2500, 3000), 200, np.uint8))
    fs = [_finding(0, 100, 100, 4, "critical"), _finding(1, 2000, 1500, 300, "high"), _finding(2, 50, 2400, 6, "low")]
    sel = E.select(fs)
    assert [d["id"] for d in sel] == [0, 1]
    assert len(E.select(fs, include_low=True, hidden_ids={1})) == 2

    E.overview(src, sel, tmp_path / "o.png", width=1000, title="t")
    E.crop(src, sel, tmp_path / "c.png", (0, 0, 600, 600), min_width=0)
    import cv2

    c = cv2.imread(str(tmp_path / "c.png"))
    c = c[c.shape[0] - 600 :]  # under the legend strip
    assert c.shape == (600, 600, 3)
    assert (np.abs(c[102 - 10 : 102 + 10, 102 - 10 : 102 + 10].astype(int) - 200).max(-1) > 60).any()  # the ring

    E.tiff(src, sel, tmp_path / "f.tif")
    with tifffile.TiffFile(tmp_path / "f.tif") as t:
        p = t.pages[0]
        assert p.shape == (2500, 3000, 3) and p.is_tiled
        assert len(p.pages) >= 2  # pyramid in SubIFDs
        a = p.asarray()
    assert (np.abs(a[1490:1510, 2000:2300].astype(int) - 200).max(-1) > 60).any()  # the polygon outline
    edge = a[1490:1510, 2000:2300].reshape(-1, 3).astype(int)
    r, g, b = edge[np.abs(edge - 200).max(-1).argmax()]
    assert r > b + 60  # high = orange: red-heavy in RGB order


def test_small_crop_is_enlarged(tmp_path):
    import cv2

    src = Pyr(np.full((500, 500), 200, np.uint8))
    E.crop(src, [_finding(0, 30, 30, 4, "critical")], tmp_path / "c.png", (0, 0, 64, 40))
    c = cv2.imread(str(tmp_path / "c.png"))
    assert c.shape[1] >= 1200


def test_heatmap(tmp_path):
    import cv2

    src = Pyr(np.full((2000, 2000), 200, np.uint8))
    fs = [_finding(i, 300 + (i % 10) * 5, 300 + (i // 10) * 5, 3, "high") for i in range(100)] + [_finding(999, 1700, 1700, 3, "high")]
    E.heatmap(src, fs, tmp_path / "h.png", width=1000, cell_um=100, px_um=1.0)
    h = cv2.imread(str(tmp_path / "h.png"))
    h = h[h.shape[0] - 1000 :].astype(int)
    hot, lone, empty = h[160], h[850], h[500]
    assert hot[160].sum() > lone[850].sum()  # the cluster is brighter than one stray finding
    assert np.ptp(empty[500]) < 10  # grey where there are none


def test_heatmap_without_scan(tmp_path):
    import cv2

    a = np.full((2000, 2000), 200, np.uint8)
    a[900:1100, :] = 30  # a "structure" that must not show
    src = Pyr(a)
    mask = np.zeros((200, 200), bool)
    mask[20:180, 20:180] = True
    fs = [_finding(i, 400 + i, 400, 3, "high") for i in range(30)]
    E.heatmap(src, fs, tmp_path / "h.png", width=1000, cell_um=100, px_um=1.0, show_scan=False, chip_mask=mask)
    h = cv2.imread(str(tmp_path / "h.png")).astype(int)
    h = h[h.shape[0] - 1000 :]
    assert np.ptp(h[500, 300:700]) < 6  # the dark band is gone: flat silhouette
    assert h[50, 50].min() >= 250 and h[1500 // 2 + 100, 750].max() < 240  # white off the chip, grey on it

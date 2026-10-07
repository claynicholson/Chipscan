"""Image sources: anything that can hand back a region of a (possibly huge) grayscale image.

All sources expose a pyramid: level 0 is full resolution, level k is downsampled by 2**k.
Regions are requested in *level-0 pixel coordinates* and returned as uint8 arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import zarr

from chipscan.geometry import BBox


@runtime_checkable
class ImageSource(Protocol):
    pixel_size_um: float | None

    @property
    def shape(self) -> tuple[int, int]:
        """(height, width) at level 0."""
        ...

    @property
    def num_levels(self) -> int: ...

    def level_shape(self, level: int) -> tuple[int, int]: ...

    def read_region(self, x0: int, y0: int, x1: int, y1: int, level: int = 0) -> np.ndarray:
        """Read [y0:y1, x0:x1] given in level-0 coordinates, from the given level.

        Out-of-bounds parts are zero-padded so callers always get the shape they asked for.
        """
        ...

    def read_level(self, level: int) -> np.ndarray: ...


def _pad_read(arr_getter, lvl_shape: tuple[int, int], ys: int, ye: int, xs: int, xe: int) -> np.ndarray:
    """Read with zero padding outside the array bounds."""
    h, w = lvl_shape
    out = np.zeros((ye - ys, xe - xs), dtype=np.uint8)
    cy0, cy1 = max(ys, 0), min(ye, h)
    cx0, cx1 = max(xs, 0), min(xe, w)
    if cy1 > cy0 and cx1 > cx0:
        out[cy0 - ys : cy1 - ys, cx0 - xs : cx1 - xs] = arr_getter(cy0, cy1, cx0, cx1)
    return out


class _LevelMixin:
    def level_scale(self, level: int) -> int:
        return 2**level

    def best_level_for(self, target_pixel_size_um: float) -> int:
        """Largest level whose pixel size does not exceed the target."""
        if self.pixel_size_um is None:
            return 0
        lvl = 0
        while lvl + 1 < self.num_levels and self.pixel_size_um * (2 ** (lvl + 1)) <= target_pixel_size_um:
            lvl += 1
        return lvl

    def level_pixel_size(self, level: int) -> float | None:
        return None if self.pixel_size_um is None else self.pixel_size_um * (2**level)

    @property
    def bbox_px(self) -> BBox:
        h, w = self.shape
        return BBox(0, 0, w, h)


class ArraySource(_LevelMixin):
    """In-memory source; builds its own pyramid by 2x2 block averaging. For tests and small images."""

    def __init__(self, arr: np.ndarray, pixel_size_um: float | None = None, min_level_px: int = 256):
        a = np.asarray(arr)
        if a.ndim == 3:
            a = a.mean(axis=2)
        a = np.clip(a, 0, 255).astype(np.uint8)
        self.levels: list[np.ndarray] = [a]
        while min(self.levels[-1].shape) > min_level_px:
            self.levels.append(downsample2(self.levels[-1]))
        self.pixel_size_um = pixel_size_um

    @property
    def shape(self) -> tuple[int, int]:
        return self.levels[0].shape  # type: ignore[return-value]

    @property
    def num_levels(self) -> int:
        return len(self.levels)

    def level_shape(self, level: int) -> tuple[int, int]:
        return self.levels[level].shape  # type: ignore[return-value]

    def read_region(self, x0: int, y0: int, x1: int, y1: int, level: int = 0) -> np.ndarray:
        s = 2**level
        a = self.levels[level]
        ys, ye, xs, xe = y0 // s, -(-y1 // s), x0 // s, -(-x1 // s)
        return _pad_read(lambda a0, a1, b0, b1: a[a0:a1, b0:b1], a.shape, ys, ye, xs, xe)

    def read_level(self, level: int) -> np.ndarray:
        return self.levels[level]


class PyramidSource(_LevelMixin):
    """Zarr-backed pyramid written by :func:`chipscan.io.pyramid.build_pyramid`."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.group = zarr.open_group(str(self.path), mode="r")
        self._levels: list[zarr.Array] = []
        i = 0
        while str(i) in self.group:
            self._levels.append(self.group[str(i)])  # type: ignore[arg-type]
            i += 1
        if not self._levels:
            raise ValueError(f"no pyramid levels found in {self.path}")
        attrs = dict(self.group.attrs)
        self.pixel_size_um: float | None = attrs.get("pixel_size_um")
        self.attrs = attrs

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self._levels[0].shape)  # type: ignore[return-value]

    @property
    def num_levels(self) -> int:
        return len(self._levels)

    def level_shape(self, level: int) -> tuple[int, int]:
        return tuple(self._levels[level].shape)  # type: ignore[return-value]

    def read_region(self, x0: int, y0: int, x1: int, y1: int, level: int = 0) -> np.ndarray:
        s = 2**level
        a = self._levels[level]
        ys, ye, xs, xe = y0 // s, -(-y1 // s), x0 // s, -(-x1 // s)
        return _pad_read(lambda a0, a1, b0, b1: np.asarray(a[a0:a1, b0:b1]), a.shape, ys, ye, xs, xe)

    def read_level(self, level: int) -> np.ndarray:
        return np.asarray(self._levels[level][:])


def downsample2(a: np.ndarray) -> np.ndarray:
    """2x2 block mean; odd edges are dropped (consistent with level-0 coords // 2)."""
    h, w = a.shape[:2]
    h2, w2 = h // 2, w // 2
    if h2 == 0 or w2 == 0:
        return a[:max(h2, 1), :max(w2, 1)]
    b = a[: h2 * 2, : w2 * 2].astype(np.float32)
    out = 0.25 * (b[0::2, 0::2] + b[1::2, 0::2] + b[0::2, 1::2] + b[1::2, 1::2])
    return np.rint(out).astype(np.uint8)

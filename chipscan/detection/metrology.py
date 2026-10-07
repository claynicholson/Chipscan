"""Line-cut metrology: measure feature widths along user-defined cuts in design coordinates.

A cut is a segment p0 -> p1 in microns.  We sample the predicted metal
probability along it (through the warp, so the cut lands on the right pixels
even on a tilted mosaic), locate the 0.5 crossings with sub-pixel linear
interpolation, and report the run lengths of metal / gap in microns.  For a
coplanar waveguide this yields [gap, centre, gap] directly, which is exactly what
the lab now measures by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

from chipscan.config import MetrologyCut
from chipscan.geometry import BBox, WarpField

ReadProb = Callable[[BBox], np.ndarray]
"""Function returning the metal probability (float32 H x W) for a level-0 px box."""


@dataclass
class CutResult:
    name: str
    runs: list[tuple[str, float]]  # ("metal"|"gap", length_um), in order along the cut
    edges_um: list[float]  # distance along the cut of each metal/gap transition
    expect_um: list[float] | None = None
    error_um: list[float] | None = None
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "runs": [{"phase": p, "length_um": length} for p, length in self.runs],
            "edges_um": self.edges_um,
            "expect_um": self.expect_um,
            "error_um": self.error_um,
            **self.meta,
        }


def sample_along(
    read_prob: ReadProb, warp: WarpField, p0: np.ndarray, p1: np.ndarray, step_um: float, pad_px: int = 4
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinearly sample the probability map along p0->p1. Returns (t_um, values)."""
    length = float(np.linalg.norm(p1 - p0))
    n = max(int(np.ceil(length / step_um)) + 1, 2)
    t = np.linspace(0.0, length, n)
    pts = p0[None, :] + (t / length)[:, None] * (p1 - p0)[None, :]
    px = warp.world_to_px(pts)
    bb = BBox.of_points(px).pad(pad_px)
    x0, y0, x1, y1 = bb.to_int()
    prob = read_prob(BBox(x0, y0, x1, y1))
    coords = np.vstack([px[:, 1] - y0 - 0.5, px[:, 0] - x0 - 0.5])  # pixel-centre convention
    vals = ndi.map_coordinates(prob.astype(np.float32), coords, order=1, mode="nearest")
    return t, vals


def runs_from_profile(t: np.ndarray, v: np.ndarray, thr: float = 0.5) -> tuple[list[tuple[str, float]], list[float]]:
    above = v > thr
    edges: list[float] = []
    for i in range(1, len(v)):
        if above[i] != above[i - 1]:
            # linear interpolation of the crossing
            a, b = v[i - 1] - thr, v[i] - thr
            frac = a / (a - b) if (a - b) != 0 else 0.5
            edges.append(float(t[i - 1] + frac * (t[i] - t[i - 1])))
    bounds = [float(t[0]), *edges, float(t[-1])]
    runs = []
    phase = above[0]
    for a, b in zip(bounds[:-1], bounds[1:], strict=True):
        runs.append(("metal" if phase else "gap", b - a))
        phase = not phase
    return runs, edges


def measure_cut(
    cut: MetrologyCut, read_prob: ReadProb, warp: WarpField, pixel_size_um: float, step_um: float | None = None
) -> CutResult:
    p0, p1 = np.array(cut.p0_um, dtype=float), np.array(cut.p1_um, dtype=float)
    step = step_um or pixel_size_um / 4.0
    t, v = sample_along(read_prob, warp, p0, p1, step)
    runs, edges = runs_from_profile(t, v)
    res = CutResult(cut.name, runs, edges, cut.expect_um)
    if cut.expect_um:
        # compare the interior runs (those bounded by two edges) against expectations
        interior = [length for _, length in runs[1:-1]] if len(runs) > 2 else [length for _, length in runs]
        k = min(len(interior), len(cut.expect_um))
        res.error_um = [interior[i] - cut.expect_um[i] for i in range(k)]
        res.meta["n_runs_interior"] = len(interior)
    return res

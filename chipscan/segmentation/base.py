"""Segmenter interface: image tile -> probability of metal, per pixel.

Any model that maps a float32 [0,1] grayscale tile (optionally with a design
prior tile) to a float32 [0,1] probability map can plug into the pipeline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Segmenter(Protocol):
    name: str
    uses_design_prior: bool

    def predict(self, tiles: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
        """tiles: (B,H,W) float32 in [0,1]; priors: (B,H,W) uint8/float design masks or None.

        Returns (B,H,W) float32 probabilities of metal.
        """
        ...


def to_float01(tile: np.ndarray) -> np.ndarray:
    t = np.asarray(tile)
    if t.dtype == np.uint8:
        return t.astype(np.float32) / 255.0
    if t.dtype == np.uint16:
        return t.astype(np.float32) / 65535.0
    return np.clip(t.astype(np.float32), 0.0, 1.0)

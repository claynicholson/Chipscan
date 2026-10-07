"""Global (coarse) registration of a microscope image to the design.

Solves rotation ("tilt"), isotropic scale, mirror, and translation with no initial
guess beyond an approximate pixel size:

1. Work at ~`target_pixel_size_um` on the pyramid (few thousand px on a side).
2. Compare *edge maps* so the result is independent of polarity/contrast.
3. For each gross hypothesis (flip x {0, 90, 180, 270} deg): Fourier-Mellin on the FFT
   magnitudes gives the residual rotation and scale; then phase correlation gives
   translation.  Every sign/direction ambiguity is resolved by *scoring* the
   candidates (normalized cross-correlation of edge maps), never by convention.
4. The best-scoring hypothesis is returned as a world->level-0-px Affine2D.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.fft import fft2, fftshift
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar

from chipscan.config import CoarseRegistrationConfig
from chipscan.design import Design
from chipscan.geometry import Affine2D, BBox
from chipscan.io.readers import ImageSource


@dataclass
class CoarseResult:
    affine: Affine2D  # world (um) -> level-0 px
    score: float
    pixel_size_um: float
    level: int
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "affine": self.affine.to_dict(),
            "score": self.score,
            "pixel_size_um": self.pixel_size_um,
            "level": self.level,
            "candidates": self.candidates,
        }


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def edge_map(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Gradient magnitude, robustly normalized to [0,1]. Polarity-invariant."""
    f = img.astype(np.float32)
    if f.max() > 1.5:
        f /= 255.0
    if sigma > 0:
        f = cv2.GaussianBlur(f, (0, 0), sigma)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    m = np.hypot(gx, gy)
    hi = np.percentile(m, 99.5)
    if hi <= 0:
        return np.zeros_like(m)
    return np.clip(m / hi, 0, 1).astype(np.float32)


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    if a.size < 100:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return 0.0 if d == 0 else float((a * b).sum() / d)


def _hann2d(shape: tuple[int, int]) -> np.ndarray:
    wy = np.hanning(shape[0])[:, None]
    wx = np.hanning(shape[1])[None, :]
    return (wy * wx).astype(np.float32)


def _log_polar_spectrum(img: np.ndarray, n_angles: int = 720) -> tuple[np.ndarray, float, int]:
    h, w = img.shape
    f = np.abs(fftshift(fft2(img * _hann2d(img.shape))))
    # high-pass: kill the DC/low-frequency blob that otherwise dominates the log-polar image
    yy = np.linspace(-1.0, 1.0, h)[:, None]
    xx = np.linspace(-1.0, 1.0, w)[None, :]
    r = np.sqrt(yy**2 + xx**2)
    f = f * (1.0 - np.cos(np.clip(r, 0, 1) * np.pi)) * 0.5
    radius = min(h, w) // 2
    lp = warp_polar(np.log1p(f), radius=radius, output_shape=(n_angles, radius), scaling="log", order=1)
    k_radius = radius / math.log(radius)  # skimage: r = exp(col / k_radius)
    return lp.astype(np.float32), k_radius, n_angles


def fourier_mellin(ref: np.ndarray, mov: np.ndarray) -> list[tuple[float, float]]:
    """Return candidate (rotation_deg, scale) pairs that map `mov` onto `ref`.

    Fourier-Mellin is ambiguous in sign conventions and by 180 degrees; we return all
    combinations and let the caller score them.
    """
    lp_r, k_r, n_ang = _log_polar_spectrum(ref)
    lp_m, _, _ = _log_polar_spectrum(mov)
    shift, _, _ = phase_cross_correlation(lp_r, lp_m, upsample_factor=20, normalization=None)
    d_ang = float(shift[0]) * 360.0 / n_ang
    d_scale = math.exp(float(shift[1]) / k_r)
    out = []
    for ang in (d_ang, -d_ang):
        for sc in (d_scale, 1.0 / d_scale):
            out.append((ang, sc))
            out.append((ang + 180.0, sc))
    return out


# ---------------------------------------------------------------------------
# main solve
# ---------------------------------------------------------------------------
def _square_pad(a: np.ndarray) -> np.ndarray:
    h, w = a.shape
    n = max(h, w)
    out = np.zeros((n, n), dtype=a.dtype)
    out[(n - h) // 2 : (n - h) // 2 + h, (n - w) // 2 : (n - w) // 2 + w] = a
    return out


def coarse_register(
    source: ImageSource,
    design: Design,
    cfg: CoarseRegistrationConfig,
    pixel_size_um: float | None = None,
    max_dim_px: int | None = None,
) -> CoarseResult:
    """Find the global world->px transform.

    Strategy: at a coarse resolution, for every gross orientation (and mirror) do a
    brute-force sweep of the residual angle, phase-correlate the rendered design's edge
    map with the image's edge map for translation, and score the hypothesis by normalized
    cross-correlation. Fourier-Mellin supplies extra scale/angle candidates. A second
    sweep refines the angle at a tenth of the step around the winner. Every ambiguity is
    settled by score, never by convention.
    """
    max_dim_px = max_dim_px or cfg.max_dim_px
    if pixel_size_um is None:
        h0, w0 = source.shape
        pixel_size_um = max(design.bbox.width / w0, design.bbox.height / h0)
    level = 0
    while level + 1 < source.num_levels and pixel_size_um * 2 ** (level + 1) <= cfg.target_pixel_size_um:
        level += 1
    img = source.read_level(level).astype(np.float32) / 255.0
    extra = 1.0
    if max(img.shape) > max_dim_px:
        extra = max_dim_px / max(img.shape)
        img = cv2.resize(img, None, fx=extra, fy=extra, interpolation=cv2.INTER_AREA)
    lvl_scale = (2**level) / extra  # level-0 px per working px
    k_nominal = 1.0 / (pixel_size_um * lvl_scale)  # working px per um
    H, W = img.shape
    img_e = edge_map(img, cfg.edge_sigma_px)
    img_e_sq = _square_pad(img_e)

    def hypothesis(scale_k: float, rot: float, flip: bool) -> Affine2D:
        a0 = Affine2D.from_similarity(scale_k, rot, 0.0, 0.0, flip_x=flip)
        cx, cy = a0.apply(np.array([design.bbox.center]))[0]
        return Affine2D.translation(W / 2 - cx, H / 2 - cy).compose(a0)

    def render_edges(a: Affine2D) -> tuple[np.ndarray, np.ndarray]:
        m = design.render(a, BBox(0, 0, W, H), supersample=1).astype(np.float32)
        ext = a.apply_bbox(design.bbox).intersection(BBox(0, 0, W, H))
        valid = np.zeros((H, W), dtype=bool)
        if ext is not None:
            x0, y0, x1, y1 = ext.to_int()
            valid[max(y0, 0) : y1, max(x0, 0) : x1] = True
        return edge_map(m, cfg.edge_sigma_px), valid

    cands: list[dict] = []
    best: tuple[float, Affine2D] | None = None

    def evaluate(rot: float, scale: float, flip: bool) -> float:
        nonlocal best
        a1 = hypothesis(k_nominal * scale, rot, flip)
        de, _ = render_edges(a1)
        if de.max() == 0:
            return -1.0
        shift, _, _ = phase_cross_correlation(img_e, de, upsample_factor=4, normalization="phase")
        top = -1.0
        for sgn in (1.0, -1.0):
            dy, dx = sgn * float(shift[0]), sgn * float(shift[1])
            a2 = Affine2D.translation(dx, dy).compose(a1)
            de2, valid2 = render_edges(a2)
            sc = ncc(img_e, de2, valid2)
            cands.append({"flip": flip, "rot": rot, "scale": scale, "dx": dx, "dy": dy, "score": sc})
            if best is None or sc > best[0]:
                best = (sc, a2)
            top = max(top, sc)
        return top

    flips = [False, True] if cfg.allow_flip else [False]
    for flip in flips:
        for rot0 in cfg.rotation_candidates_deg:
            # scale candidates: nominal plus whatever Fourier-Mellin suggests
            scales = {1.0}
            if cfg.allow_scale:
                de_seed, _ = render_edges(hypothesis(k_nominal, rot0, flip))
                if de_seed.max() > 0:
                    for d_rot, d_scale in fourier_mellin(img_e_sq, _square_pad(de_seed)):
                        if abs(d_scale - 1.0) <= cfg.max_scale_error:
                            scales.add(round(d_scale, 4))
                        if abs(d_rot) <= cfg.rotation_search_deg + 1e-6:
                            pass  # the sweep below covers it
            sweep = np.arange(-cfg.rotation_search_deg, cfg.rotation_search_deg + 1e-9, cfg.rotation_step_deg)
            for scale in sorted(scales):
                for d in sweep:
                    evaluate(rot0 + float(d), scale, flip)

    if best is None:
        raise RuntimeError("coarse registration found no viable hypothesis")
    # refine angle (and a little scale) around the winner
    top = max(cands, key=lambda c: c["score"])
    fine_step = cfg.rotation_step_deg / 10.0
    for scale in sorted({top["scale"], top["scale"] * 0.995, top["scale"] * 1.005} if cfg.allow_scale else {top["scale"]}):
        for d in np.arange(-cfg.rotation_step_deg, cfg.rotation_step_deg + 1e-9, fine_step):
            evaluate(top["rot"] + float(d), scale, top["flip"])

    score, a_work = best
    cands.sort(key=lambda c: -c["score"])
    affine0 = Affine2D.scaling(lvl_scale).compose(a_work)  # world -> level-0 px
    px_um = 1.0 / affine0.scale_factor
    if score < cfg.min_score:
        raise RuntimeError(f"coarse registration score {score:.3f} below min_score {cfg.min_score}")
    return CoarseResult(affine0, score, px_um, level, cands[:12])

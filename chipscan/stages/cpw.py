"""CPW and gap metrology against the design: measured line widths and gap widths in um.

Where: the design says where narrow structures are.  Along each chip, horizontal cuts are taken
every `pitch_um`; on each cut the rendered design gives the nominal runs (gap / metal / gap /
...) and their widths.
What: the scan, averaged over `avg_px` rows along the line, shows every metal/substrate edge as
a thin dark outline (brightfield edge contrast).  Each outline is located to sub-pixel precision
(parabola through the minimum).  The design's edge pattern is slid along the image profile to
the best match (this absorbs registration and stitching offsets of up to `search_px`), then each
nominal run is measured between its two matched edges.

Output per run: kind (gap / metal), nominal um, measured um, error um, position; and per cut a
match score.  Aggregates per structure give mean, spread and the worst sites.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi

from chipscan.design.wafer import WaferDesign
from chipscan.design.wafer_register import WaferRegistration
from chipscan.io.readers import ImageSource


@dataclass
class Run:
    kind: str  # "gap" or "metal"
    nominal_um: float
    measured_um: float
    x_um: float  # design x of the run's left edge
    y_um: float
    col_px: float  # image column of the run's left edge
    row_px: float

    @property
    def error_um(self) -> float:
        return self.measured_um - self.nominal_um


def design_runs(wd: WaferDesign, y_um: float, x0_um: float, x1_um: float, res_um: float = 0.25) -> list[tuple[int, float, float]]:
    """(metal?, start_um, width_um) along a horizontal design line."""
    r = wd.render(res_um, (x0_um, y_um, x1_um, y_um + res_um))[0].astype(np.int8)
    ch = np.flatnonzero(np.diff(r)) + 1
    b = np.r_[0, ch, len(r)]
    return [(int(r[b[i]]), x0_um + b[i] * res_um, (b[i + 1] - b[i]) * res_um) for i in range(len(b) - 1)]


def edge_dips(profile: np.ndarray, min_depth: float = 6.0) -> np.ndarray:
    """Sub-pixel positions of dark outline minima (the edges) in a 1-D brightfield profile."""
    p = profile.astype(np.float64)
    base = ndi.median_filter(p, size=9, mode="nearest")
    d = base - p
    i = np.flatnonzero((d[1:-1] >= d[:-2]) & (d[1:-1] > d[2:]) & (d[1:-1] > min_depth)) + 1
    out = []
    for k in i:
        a, b, c = p[k - 1], p[k], p[k + 1]
        den = a - 2 * b + c
        out.append(k + (0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0))
    return np.asarray(out)


def measure_cut(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, y_um: float, x0_um: float, x1_um: float, avg_px: int = 31, search_px: int = 160, max_run_um: float = 150.0, tol_px: float = 1.6):
    """Measure every narrow run on one cut. Returns (runs, score) where score = matched edge fraction."""
    runs = design_runs(wd, y_um, x0_um, x1_um)
    # keep interior runs that are narrow enough to be a line or gap (not the open ground plane)
    interior = [(m, s, w) for m, s, w in runs[1:-1] if 1.0 <= w <= max_run_um]
    if len(interior) < 2:
        return [], 0.0
    px = reg.pixel_size_um
    c0, r0 = reg.to_px(np.array([x0_um]), np.array([y_um]))
    c1, _ = reg.to_px(np.array([x1_um]), np.array([y_um]))
    col_a, col_b = int(min(c0[0], c1[0])) - search_px, int(max(c0[0], c1[0])) + search_px
    row = int(round(float(r0[0])))
    img = src.read_region(col_a, row - avg_px // 2, col_b, row + avg_px // 2 + 1).astype(np.float64)
    if img.ndim == 3:
        img = img.mean(-1)
    prof = img.mean(0)
    dips = edge_dips(prof)
    if len(dips) < 2:
        return [], 0.0
    # design edges in image px (affine + local field), relative to col_a
    edges_um = np.array(sorted({s for _, s, _ in interior} | {s + w for _, s, w in interior}))
    ec, _ = reg.to_px(edges_um, np.full_like(edges_um, y_um))
    ec = np.asarray(ec) - col_a
    flip = ec[-1] < ec[0]
    # slide the design edge pattern along the profile: most edges within tol wins
    best = (-1, 0.0)
    for sh in np.arange(-search_px, search_px + 0.01, 0.5):
        e = ec + sh
        dist = np.abs(dips[None, :] - e[:, None]).min(1)
        n = int((dist <= tol_px).sum())
        if n > best[0] or (n == best[0] and abs(sh) < abs(best[1])):
            best = (n, sh)
    n_ok, sh = best
    e = ec + sh
    idx = np.abs(dips[None, :] - e[:, None]).argmin(1)
    ok = np.abs(dips[idx] - e) <= tol_px
    pos = {float(u): (float(dips[j]) if g else None) for u, j, g in zip(edges_um, idx, ok)}
    out = []
    for m, s, w in interior:
        a, b = pos.get(float(s)), pos.get(float(s + w))
        if a is None or b is None:
            continue
        out.append(Run("metal" if m else "gap", round(w, 2), round(abs(b - a) * px, 3), round(s, 2), round(y_um, 2), round(min(a, b) + col_a, 2), float(row)))
    return out, n_ok / max(1, len(edges_um)) if not flip or True else 0.0


def survey(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, windows: list[tuple[float, float, float, float]], pitch_um: float = 500.0, min_score: float = 0.7, progress=None) -> list[dict]:
    """Cuts every `pitch_um` along y in each (x0, y0, x1, y1) design window."""
    rows: list[dict] = []
    cuts = [(x0, y, x1) for x0, y0, x1, y1 in windows for y in np.arange(y0 + pitch_um / 2, y1, pitch_um)]
    for i, (x0, y, x1) in enumerate(cuts):
        runs, score = measure_cut(src, wd, reg, float(y), x0, x1)
        if score >= min_score:
            for r in runs:
                d = asdict(r)
                d["error_um"] = round(r.error_um, 3)
                d["cut_score"] = round(score, 3)
                rows.append(d)
        if progress and i % 20 == 0:
            progress(i + 1, len(cuts))
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    """Per (kind, nominal width): count, mean measured, mean error, spread, worst."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["kind"], round(r["nominal_um"])), []).append(r)
    out = []
    for (kind, nom), g in sorted(groups.items()):
        m = np.array([r["measured_um"] for r in g])
        e = m - nom
        worst = max(g, key=lambda r: abs(r["error_um"]))
        out.append({"kind": kind, "nominal_um": nom, "n": len(g), "resolved": bool(nom >= 3.0 * float(np.median([r.get("px_um", 1.63) for r in g]))), "mean_um": round(float(m.mean()), 2), "bias_um": round(float(e.mean()), 2), "std_um": round(float(m.std()), 2), "worst_error_um": round(worst["error_um"], 2), "worst_at_px": [worst["col_px"], worst["row_px"]]})
    return out


def structure_groups(runs: list[tuple[int, float, float]], max_run_um: float = 150.0, min_runs: int = 3) -> list[tuple[float, float]]:
    """Clusters of narrow runs (a line structure) along a cut, as (x0, x1) design windows with
    a little open ground on each side."""
    groups, cur = [], []
    for m, s, w in runs:
        if w <= max_run_um:
            cur.append((s, w))
        else:
            if len(cur) >= min_runs:
                groups.append((cur[0][0] - 20.0, cur[-1][0] + cur[-1][1] + 20.0))
            cur = []
    if len(cur) >= min_runs:
        groups.append((cur[0][0] - 20.0, cur[-1][0] + cur[-1][1] + 20.0))
    return groups


def survey_wafer(src: ImageSource, wd: WaferDesign, reg: WaferRegistration, pitch_um: float = 2000.0, min_score: float = 0.8, progress=None) -> list[dict]:
    """Cuts every `pitch_um` down each reticle; every line structure on a cut is measured on its own."""
    rows: list[dict] = []
    reticles = [p for p in wd.placements if p.cell.startswith("RETICLE")]
    half = 20600.0
    cuts = [(p, y) for p in reticles for y in np.arange(p.origin[1] - half + pitch_um / 2, p.origin[1] + half, pitch_um)]
    for i, (p, y) in enumerate(cuts):
        x0, x1 = p.origin[0] - half, p.origin[0] + half
        for gx0, gx1 in structure_groups(design_runs(wd, float(y), x0, x1, res_um=0.5)):
            runs, score = measure_cut(src, wd, reg, float(y), gx0, gx1)
            if score < min_score:
                continue
            for r in runs:
                d = asdict(r)
                d["error_um"] = round(r.error_um, 3)
                d["cut_score"] = round(score, 3)
                d["reticle"] = p.cell
                rows.append(d)
        if progress and i % 10 == 0:
            progress(i + 1, len(cuts))
    return rows

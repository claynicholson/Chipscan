"""Figures for the Chip Scanner journal paper, in IEEE style (matplotlib).

    uv run python scripts/paper/make_figures.py [--out paper/figures]

Every figure is drawn from data the pipeline wrote: model training histories, the planted-speck
benchmark reports, the CPW metrology files and the wafer heatmap exports. No scan image is shown
except as a findings heatmap with the scan hidden.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
COL = 3.5  # IEEE single column, inches
DCOL = 7.16  # double column

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "legend.frameon": True,
    "legend.edgecolor": "black",
    "legend.fancybox": False,
    "legend.framealpha": 1.0,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linewidth": 0.4,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

GREYS = ["0.0", "0.45", "0.7"]
MARKERS = ["o", "s", "^", "D", "v"]
REPORTS = {  # planted-speck benchmark on held-out chip C1 (scripts/eval_specks.py)
    "speck_v2": "runs/_eval/report_reference.json",
    "speck_v3": "runs/_eval_v3/report_reference.json",
    "speck_v4": "runs/_eval_v4/report_reference.json",
}


def save(fig, out: Path, name: str) -> None:
    """PNG for previews; PDF (vector) for LaTeX, except the raster heatmap, which goes in as JPEG."""
    fig.savefig(out / f"{name}.png")
    if name.startswith("fig5"):
        import cv2

        cv2.imwrite(str(out / f"{name}.jpg"), cv2.imread(str(out / f"{name}.png")), [cv2.IMWRITE_JPEG_QUALITY, 90])
    else:
        fig.savefig(out / f"{name}.pdf")


def panel_label(ax, s):
    ax.text(0.5, -0.30, s, transform=ax.transAxes, ha="center", va="top")


def fig_pipeline(out: Path):
    fig, ax = plt.subplots(figsize=(DCOL, 1.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 24)
    ax.axis("off")

    def box(x, y, w, h, text, dashed=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0", lw=0.7, ec="black", fc="white", ls="--" if dashed else "-")
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7, linespacing=1.15)
        return (x, y, w, h)

    def arrow(a, b, dashed=False):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=7, lw=0.7, color="black", ls="--" if dashed else "-", shrinkA=0, shrinkB=0))

    h, y = 6.5, 14.5
    xs = [0.5, 13.5, 27.5, 46.5, 63.5, 79.5]
    ws = [11, 12, 17, 15, 14, 20]
    labels = [
        "Scan\n(.vsi)",
        "Pyramid\n(zarr)",
        "Flat field, tilt\ncorrection, chip mask",
        "Fused networks:\nlayout, speck,\nanomaly, trace",
        "Candidates\n+ reference\nresidual, pad rule",
        "Verification chain\n→ risk model\n→ ranked findings",
    ]
    bxs = [box(x, y, w, h, t) for x, w, t in zip(xs, ws, labels)]
    for a, b in zip(bxs, bxs[1:]):
        arrow((a[0] + a[2], y + h / 2), (b[0], y + h / 2))
    # design branch
    yb = 2.5
    g = box(27.5, yb, 17, h, "GDS layout\n(per-reticle index)", dashed=True)
    r = box(46.5, yb, 15, h, "Registration:\naffine, drift field,\nper-frame offset", dashed=True)
    m = box(63.5, yb, 14, h, "CPW line and\ngap metrology", dashed=True)
    arrow((g[0] + g[2], yb + h / 2), (r[0], yb + h / 2), dashed=True)
    arrow((r[0] + r[2], yb + h / 2), (m[0], yb + h / 2), dashed=True)
    arrow((35.5, y), (52.0, yb + h), dashed=True)  # corrected image feeds registration
    arrow((m[0] + m[2], yb + h / 2), (89.5, y), dashed=True)  # results join the review output
    ax.text(99.5, 22.8, "solid: every scan;  dashed: when the design is available", ha="right", va="top", fontsize=6.5, style="italic")
    save(fig, out, "fig1_pipeline")
    plt.close(fig)


def fig_training(out: Path):
    sp = json.loads((ROOT / "models/speck_v4.history.json").read_text())
    la = json.loads((ROOT / "models/layout_v1.history.json").read_text())
    fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.15))
    e = [r["epoch"] for r in sp]
    a.plot(e, [r["recall"] for r in sp], "-o", color="0.0", label="Recall", markevery=2)
    a.plot(e, [r["precision"] for r in sp], "-s", color="0.45", label="Precision", markevery=2, mfc="white")
    a.plot(e, [r["clean_tile_fa"] for r in sp], "--^", color="0.0", label="Clean tiles with false detection", markevery=2, mfc="white")
    a.set_xlabel("Epoch")
    a.set_ylabel("Fraction")
    a.set_xlim(1, max(e))
    a.set_ylim(0, 1.02)
    a.legend(loc="center right")
    panel_label(a, "(a)")
    e2 = [r["epoch"] for r in la]
    for (k, name), c, mk in zip((("iou_background", "Background"), ("iou_component", "Device"), ("iou_trace", "Trace"), ("iou_text", "Lettering")), ["0.6", "0.0", "0.35", "0.0"], ["o", "s", "^", "D"]):
        b.plot(e2, [r[k] for r in la], "-" if name != "Lettering" else "--", marker=mk, color=c, label=name, mfc="white" if name in ("Device", "Lettering") else c)
    b.set_xlabel("Epoch")
    b.set_ylabel("IoU")
    b.set_xlim(1, max(e2))
    b.set_ylim(0, 1.02)
    b.legend(loc="lower right")
    panel_label(b, "(b)")
    fig.subplots_adjust(wspace=0.28, bottom=0.2)
    save(fig, out, "fig2_training")
    plt.close(fig)


def _report(path):
    return json.loads((ROOT / path).read_text())["reference"]


def fig_benchmark(out: Path):
    reps = {m: _report(p) for m, p in REPORTS.items()}
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.2))
    for ax, key, title in ((axes[0], "by_size", "Speck area"), (axes[1], "by_contrast", "Contrast")):
        cats = list(next(iter(reps.values()))[key].keys())
        n = [next(iter(reps.values()))[key][c]["n"] for c in cats]
        x = np.arange(len(cats))
        w = 0.26
        hatches = ["", "////", ""]
        fills = ["white", "0.75", "0.2"]
        for i, (m, r) in enumerate(reps.items()):
            v = [100 * r[key][c]["recall"] for c in cats]
            bars = ax.bar(x + (i - 1) * w, v, w, color=fills[i], edgecolor="black", linewidth=0.6, hatch=hatches[i], label=m)
            for bx, val in zip(bars, v):
                ax.text(bx.get_x() + bx.get_width() / 2, val + 1.2, f"{val:.0f}", ha="center", va="bottom", fontsize=5.5)
        lab = [c.split(" ", 1)[1].replace("px", " px").replace("<", "< ").replace(">", "> ") + f"\n(n = {k})" for c, k in zip(cats, n)]
        ax.set_xticks(x, lab)
        ax.set_ylim(50, 104)
        ax.set_ylabel("Recall (%)")
        ax.set_xlabel(title)
        ax.grid(axis="x", visible=False)
        ax.tick_params(axis="x", top=False)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04))
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.25, bottom=0.27, top=0.88)
    save(fig, out, "fig3_recall")
    plt.close(fig)


def fig_tradeoff(out: Path):
    steps = [("speck_v2", REPORTS["speck_v2"]), ("speck_v3", REPORTS["speck_v3"]), ("speck_v4", "runs/_eval_v4/report_reference.before_isolated.json"), ("speck_v4 + context-\nconditioned checks", REPORTS["speck_v4"])]
    pts = [(_report(p)["alarms_clean_per_window"], 100 * _report(p)["recall"], n) for n, p in steps]
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="0.5", lw=0.8, zorder=1)
    for i, (xv, yv, n) in enumerate(pts):
        ax.plot(xv, yv, MARKERS[i], color="black", mfc="black" if i == len(pts) - 1 else "white", ms=4.5, zorder=2)
        off, ha = {0: ((6, -8), "left"), 1: ((6, -9), "left"), 2: ((0, 6), "center"), 3: ((-4, -18), "right")}[i]
        ax.annotate(n, (xv, yv), xytext=off, textcoords="offset points", fontsize=6.5, ha=ha)
    ax.set_xlabel("Detections per clean 768 × 768 px window")
    ax.set_ylabel("Recall (%)")
    ax.set_xlim(0, 1.4)
    ax.set_ylim(88, 94.5)
    save(fig, out, "fig4_tradeoff")
    plt.close(fig)


def fig_heatmaps(out: Path, all_png: Path, serious_png: Path, tops: tuple[float, float]):
    """Findings heatmaps exported with the scan hidden; the export's text strip is cropped and a
    colour bar added in the paper's type."""
    import cv2

    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 3.45))
    for ax, p, top, lab in zip(axes, (all_png, serious_png), tops, ("(a)", "(b)")):
        im = cv2.imread(str(p))[..., ::-1]
        strip = int(70 * max(0.8, im.shape[1] / 3500))  # export_image._heat_legend's band height
        ax.imshow(im[strip:], interpolation="lanczos")
        ax.set_axis_off()
        sm = plt.cm.ScalarMappable(cmap="inferno", norm=matplotlib.colors.Normalize(0, top))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label("Findings per mm² (saturates)", fontsize=7)
        cb.ax.tick_params(labelsize=6, direction="out", right=True, left=False)
        cb.outline.set_linewidth(0.5)
        ax.text(0.5, -0.04, lab, transform=ax.transAxes, ha="center", va="top")
    fig.subplots_adjust(wspace=0.32)
    save(fig, out, "fig5_heatmaps")
    plt.close(fig)


def fig_metrology(out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.2), sharey=True)
    runs = (("NFT1", "runs/NFT1_wafer_speck_v4/cpw.json", "o", "black"), ("009-006 #2", "runs/2026-07-08_009-006-2_speck_v4/cpw.json", "s", "white"))
    for ax, kind, title in ((axes[0], "metal", "Lines"), (axes[1], "gap", "Gaps")):
        for j, (name, p, mk, mfc) in enumerate(runs):
            s = [r for r in json.loads((ROOT / p).read_text())["summary"] if r["kind"] == kind and r["n"] >= 30 and r["nominal_um"] > 4]
            x = np.array([r["nominal_um"] for r in s], float) * (1.03 if j else 0.97)
            ax.errorbar(x, [r["bias_um"] for r in s], yerr=[r["std_um"] for r in s], fmt=mk, color="black", mfc=mfc, ecolor="0.4", elinewidth=0.6, capsize=1.5, label=name)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 50, 100, 150], ["5", "10", "20", "50", "100", "150"])
        ax.minorticks_off()
        ax.set_xlabel(f"Nominal {title.lower()[:-1]} width (µm)")
        ax.set_title(title, fontsize=8)
    axes[0].set_ylabel("Measured − nominal (µm)")
    axes[0].legend(loc="lower right")
    panel_label(axes[0], "(a)")
    panel_label(axes[1], "(b)")
    fig.subplots_adjust(wspace=0.08, bottom=0.24)
    save(fig, out, "fig6_metrology")
    plt.close(fig)


def fig_components(out: Path):
    # planted-device test on AHQ1 A2: 36 defects on 12 device types; flags counted on the clean scan
    # (67,448 devices). From scripts/dev/plant_components.py and scripts/dev/lookalike_sweep.py.
    sweep = [(4, 27, 89), (6, 27, 54), (9, 24, 36), (12, 21, 18), (20, 16, 3)]
    row_median = (6, 719)
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.plot([s[2] for s in sweep], [s[1] for s in sweep], "-o", color="black", mfc="white", label="Look-alike reference (12 nearest)")
    for z, f, fl in sweep:
        ax.annotate(f"z = {z}", (fl, f), xytext=(4, -9) if z != 4 else (4, 3), textcoords="offset points", fontsize=6)
    ax.plot([row_median[1]], [row_median[0]], "s", color="black", label="Row-median reference")
    ax.set_xscale("log")
    ax.set_xlim(2, 1500)
    ax.set_ylim(0, 36)
    ax.set_yticks([0, 6, 12, 18, 24, 30, 36])
    ax.set_xlabel("Flags on the clean scan (of 67,448 devices)")
    ax.set_ylabel("Planted defects found (of 36)")
    ax.legend(loc="lower left")
    save(fig, out, "fig7_components")
    plt.close(fig)


ZOOM_FIELDS_UM = (1500.0, 300.0, 75.0)


def fig_zoom(out: Path, run: Path, ids: list[int]):
    """Real findings on bare ground plane at three zoom levels. Every window used here was checked
    to hold no patterned structure (scripts/paper/scout_open_ground.py), so no device is shown."""
    import cv2
    from matplotlib.patches import Circle, Polygon, Rectangle

    from chipscan.io.readers import PyramidSource

    src = PyramidSource(run / "corrected.zarr")
    px = src.pixel_size_um or 1.6285
    fs = json.loads((run / "findings.json").read_text())
    fs = fs if isinstance(fs, list) else fs["findings"]
    byid = {d["id"]: d for d in fs}
    cent = np.array([d["centroid_px"] for d in fs])
    fig, axes = plt.subplots(len(ids), 3, figsize=(COL, 1.2 * len(ids) + 0.25))
    for row, fid in zip(np.atleast_2d(axes), ids):
        cx0, cy0 = byid[fid]["centroid_px"]
        h0 = ZOOM_FIELDS_UM[0] / px / 2
        # zoom on the largest finding well inside the wide window
        inner = [fs[j] for j in np.where((abs(cent[:, 0] - cx0) < 0.6 * h0) & (abs(cent[:, 1] - cy0) < 0.6 * h0))[0]]
        tgt = max(inner, key=lambda d: d["size_um"])
        centres = [(cx0, cy0), tuple(tgt["centroid_px"]), tuple(tgt["centroid_px"])]
        wide = src.read_region(int(cx0 - h0), int(cy0 - h0), int(cx0 + h0), int(cy0 + h0), level=1).astype(np.float32)
        # raw grey levels, the same mapping in every panel of a row; no contrast stretch
        lo, hi = 0.0, 255.0
        for k, (ax, field, (cx, cy)) in enumerate(zip(row, ZOOM_FIELDS_UM, centres)):
            h = field / px / 2
            x0, y0 = cx - h, cy - h
            L = 1 if k == 0 else 0
            img = src.read_region(int(x0), int(y0), int(cx + h), int(cy + h), level=L).astype(np.float32)
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ax.imshow(np.clip((img - lo) / (hi - lo), 0, 1), cmap="gray", vmin=0, vmax=1, interpolation="nearest" if k else "lanczos", extent=(0, field, field, 0))
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.6)
            near = np.where((abs(cent[:, 0] - cx) < h) & (abs(cent[:, 1] - cy) < h))[0]
            for j in near:
                d = fs[j]
                u, v = (d["centroid_px"][0] - x0) * px, (d["centroid_px"][1] - y0) * px
                if k < 2:
                    rr = max(field / 70, d["size_um"] * 0.7)
                    ax.add_patch(Circle((u, v), rr, fill=False, lw=0.45, ec="#d62728"))
                elif d.get("polygons_px"):
                    for poly in d["polygons_px"]:
                        q = (np.asarray(poly, float) - [x0, y0]) * px
                        ax.add_patch(Polygon(q, closed=True, fill=False, lw=0.9, ec="#d62728"))
            if k < 2:  # where the next panel looks
                nf = ZOOM_FIELDS_UM[k + 1]
                ncx, ncy = centres[k + 1]
                ax.add_patch(Rectangle(((ncx - x0) * px - nf / 2, (ncy - y0) * px - nf / 2), nf, nf, fill=False, lw=0.8, ec="black", ls="--"))
            bar = {1500.0: 250, 300.0: 50, 75.0: 10}[field]
            ax.plot([field * 0.06, field * 0.06 + bar], [field * 0.93] * 2, color="black", lw=1.6, solid_capstyle="butt")
            ax.text(field * 0.06 + bar / 2, field * 0.88, f"{bar} µm", ha="center", va="bottom", fontsize=5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
            ax.set_xlim(0, field)
            ax.set_ylim(field, 0)
        row[0].set_ylabel(f"{tgt['size_um']:.0f} µm, risk {tgt['risk']:.0f}", fontsize=6)
    for k, lab in enumerate(("(a) 1.5 mm", "(b) 300 µm", "(c) 75 µm")):
        np.atleast_2d(axes)[-1][k].set_xlabel(lab, fontsize=7)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    save(fig, out, "fig8_zoom")
    plt.close(fig)


def fig_many(out: Path, run: Path, centre_ids: list[int], field_um: float = 800.0):
    """Bare-ground windows holding ~50 findings each, every finding ringed, plus the 16 largest of
    the first window at full resolution. Windows checked structure-free (scout_open_ground.py)."""
    import cv2
    from matplotlib.patches import Circle, Polygon

    from chipscan.io.readers import PyramidSource

    src = PyramidSource(run / "corrected.zarr")
    px = src.pixel_size_um or 1.6285
    fs = json.loads((run / "findings.json").read_text())
    fs = fs if isinstance(fs, list) else fs["findings"]
    byid = {d["id"]: d for d in fs}
    cent = np.array([d["centroid_px"] for d in fs])
    fig = plt.figure(figsize=(DCOL, 2.55))
    gs = fig.add_gridspec(1, 3, wspace=0.06)
    h = field_um / px / 2
    first_members = []
    for k, fid in enumerate(centre_ids):
        ax = fig.add_subplot(gs[0, k])
        cx, cy = byid[fid]["centroid_px"]
        img = src.read_region(int(cx - h), int(cy - h), int(cx + h), int(cy + h), level=0).astype(np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="lanczos", extent=(0, field_um, field_um, 0))
        members = [fs[j] for j in np.where((abs(cent[:, 0] - cx) < h) & (abs(cent[:, 1] - cy) < h))[0]]
        if k == 0:
            first_members = members
        for d in members:
            u, v = (d["centroid_px"][0] - cx + h) * px, (d["centroid_px"][1] - cy + h) * px
            ax.add_patch(Circle((u, v), max(field_um / 75, d["size_um"] * 0.7), fill=False, lw=0.45, ec="#d62728"))
        ax.plot([field_um * 0.06, field_um * 0.06 + 100], [field_um * 0.93] * 2, color="black", lw=1.6, solid_capstyle="butt")
        ax.text(field_um * 0.06 + 50, field_um * 0.88, "100 µm", ha="center", va="bottom", fontsize=5.5, bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.85))
        ax.text(field_um * 0.97, field_um * 0.05, f"{len(members)} findings", ha="right", va="top", fontsize=6, bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(0, field_um)
        ax.set_ylim(field_um, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(f"({'ab'[k]})", fontsize=7)
    # gallery of the largest findings of window (a)
    sub = gs[0, 2].subgridspec(4, 4, wspace=0.04, hspace=0.04)
    big = sorted(first_members, key=lambda d: -d["size_um"])[:16]
    for n, d in enumerate(big):
        ax = fig.add_subplot(sub[n // 4, n % 4])
        b = d["bbox_px"]
        cx, cy = (b["x0"] + b["x1"]) / 2, (b["y0"] + b["y1"]) / 2
        gallery_um = max(30.0, 1.6 * max(b["x1"] - b["x0"], b["y1"] - b["y0"]) * px)  # each tile framed to its finding
        g = gallery_um / px / 2
        img = src.read_region(int(cx - g), int(cy - g), int(cx + g), int(cy + g), level=0).astype(np.float32)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(0, gallery_um, gallery_um, 0))
        for poly in d.get("polygons_px") or []:
            q = (np.asarray(poly, float) - [cx - g, cy - g]) * px
            ax.add_patch(Polygon(q, closed=True, fill=False, lw=0.5, ec="#d62728"))
        ax.text(gallery_um * 0.05, gallery_um * 0.95, f"{d['size_um']:.0f} \u00b5m", fontsize=4.5, va="bottom", bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.8))
        ax.set_xlim(0, gallery_um)
        ax.set_ylim(gallery_um, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_linewidth(0.4)
        if n == 13:
            ax.set_xlabel("(c)", fontsize=7, labelpad=3)
    save(fig, out, "fig9_many")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "figures")
    ap.add_argument("--heat-all", type=Path, default=ROOT / "runs/NFT1_wafer_speck_v4/exports/NFT1_wafer_speck_v4_heatmap_1106_015343.png")
    ap.add_argument("--heat-serious", type=Path, default=ROOT / "runs/NFT1_wafer_speck_v4/exports/NFT1_wafer_speck_v4_heatmap_1106_015343_2.png")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    fig_pipeline(a.out)
    fig_training(a.out)
    fig_benchmark(a.out)
    fig_tradeoff(a.out)
    fig_heatmaps(a.out, a.heat_all, a.heat_serious, (21.6, 1.28))
    fig_metrology(a.out)
    fig_components(a.out)
    # findings on bare ground plane, each window checked structure-free with scout_open_ground.py
    fig_zoom(a.out, ROOT / "runs/NFT1_wafer_speck_v4", [92088, 87008, 64241])
    fig_many(a.out, ROOT / "runs/NFT1_wafer_speck_v4", [6475, 5210])
    print("figures in", a.out)


if __name__ == "__main__":
    main()

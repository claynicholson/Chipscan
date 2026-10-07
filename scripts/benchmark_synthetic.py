"""Benchmark: classical tolerance-band inspection vs. the trained UNet detector on
held-out synthetic chips (different seeds, tilts, scales, flips, defect sizes).

    uv run python scripts/benchmark_synthetic.py --model models/unet_synthetic.pt --n 3

Prints a table of recall / class accuracy / false alarms per method, and writes
benchmark.json next to the model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from chipscan.config import PipelineConfig
from chipscan.evaluate import Score, score_files
from chipscan.pipeline import run_inspection
from chipscan.synth import make_synthetic

console = Console()


def run_case(out_dir: Path, seed: int, rotation: float, scale: float, flip: bool, model: Path | None, tolerance_px: int, threshold: float) -> dict[str, Score]:
    paths = make_synthetic(out_dir, rotation, scale, flip, 1.0, 2200.0, 30, seed=seed)
    results: dict[str, Score] = {}
    cfg = PipelineConfig.load(paths["config"])
    cfg.detection.tolerance_px = tolerance_px
    cfg.name = "classical"
    run_inspection(cfg)
    cfg.registration.transform_path = cfg.export.out_dir / "classical.registration.json"
    results["classical"] = score_files(cfg.export.out_dir / "classical.defects.json", paths["truth"])
    if model is not None:
        cfg2 = PipelineConfig.load(paths["config"])
        cfg2.name = "unet"
        cfg2.registration.transform_path = cfg.registration.transform_path
        cfg2.segmentation.method = "unet"
        cfg2.segmentation.model_path = model
        cfg2.segmentation.threshold = threshold
        cfg2.detection.tolerance_px = tolerance_px
        run_inspection(cfg2)
        results["unet"] = score_files(cfg2.export.out_dir / "unet.defects.json", paths["truth"])
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("data/benchmark"))
    ap.add_argument("--tolerance-px", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    cases = [(101, 2.0, 1.0, False), (102, -47.0, 1.03, True), (103, 91.5, 0.97, False), (104, 0.4, 1.0, True), (105, 12.0, 1.015, False)][: a.n]
    agg: dict[str, dict[str, float]] = {}
    per_case = []
    for seed, rot, sc, fl in cases:
        res = run_case(a.out / f"case_{seed}", seed, rot, sc, fl, a.model, a.tolerance_px, a.threshold)
        per_case.append({"seed": seed, "rotation": rot, "scale": sc, "flip": fl, **{k: v.to_dict() for k, v in res.items()}})
        for k, v in res.items():
            d = agg.setdefault(k, {"n": 0, "found": 0, "classified": 0, "fa": 0})
            d["n"] += v.n_truth
            d["found"] += v.found
            d["classified"] += v.classified
            d["fa"] += v.false_alarms
    t = Table("method", "recall", "class accuracy", "false alarms", title=f"synthetic benchmark, {len(cases)} chips, tolerance {a.tolerance_px}px")
    for k, d in agg.items():
        t.add_row(k, f"{d['found']}/{d['n']} ({d['found'] / d['n']:.0%})", f"{d['classified']}/{d['n']} ({d['classified'] / d['n']:.0%})", str(d["fa"]))
    console.print(t)
    out = (a.model.with_suffix(".benchmark.json") if a.model else a.out / "benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"aggregate": agg, "cases": per_case}, indent=2))
    console.print(f"wrote {out}")


if __name__ == "__main__":
    main()

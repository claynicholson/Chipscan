"""Score an inspection against ground truth (synthetic truth.json or a hand-labelled list).

Ground truth format: ``{"defects": [{"class": str, "x_um": float, "y_um": float, "size_um": float}, ...]}``.
A truth defect is *found* if any detection centroid lies within max(`radius_um`, size) of it;
it is *classified* if one of those detections has the same class.  Detections matched
to no truth defect are false alarms.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Score:
    n_truth: int
    found: int
    classified: int
    false_alarms: int
    per_class: dict[str, dict[str, int]] = field(default_factory=dict)
    misses: list[dict[str, Any]] = field(default_factory=list)
    false_alarm_list: list[dict[str, Any]] = field(default_factory=list)

    @property
    def recall(self) -> float:
        return self.found / self.n_truth if self.n_truth else 1.0

    @property
    def class_accuracy(self) -> float:
        return self.classified / self.n_truth if self.n_truth else 1.0

    @property
    def precision(self) -> float:
        n_det = self.found + self.false_alarms
        return self.found / n_det if n_det else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_truth": self.n_truth,
            "found": self.found,
            "classified": self.classified,
            "false_alarms": self.false_alarms,
            "recall": self.recall,
            "class_accuracy": self.class_accuracy,
            "precision": self.precision,
            "per_class": self.per_class,
            "misses": self.misses,
            "false_alarm_list": self.false_alarm_list,
        }

    def __str__(self) -> str:
        lines = [
            f"recall {self.found}/{self.n_truth} ({self.recall:.0%}), class-correct {self.classified}/{self.n_truth} ({self.class_accuracy:.0%}), false alarms {self.false_alarms} (precision {self.precision:.0%})"
        ]
        for c, d in sorted(self.per_class.items()):
            lines.append(f"  {c:16s} found {d['found']}/{d['n']}  class-correct {d['classified']}/{d['n']}")
        return "\n".join(lines)


def score(detections: list[dict[str, Any]], truth: list[dict[str, Any]], radius_um: float = 6.0) -> Score:
    det_xy = np.array([d["centroid_um"] for d in detections], dtype=float).reshape(-1, 2)
    matched = np.zeros(len(detections), dtype=bool)
    per: dict[str, Counter] = {}
    found = classified = 0
    misses = []
    for td in truth:
        c = per.setdefault(td["class"], Counter())
        c["n"] += 1
        if len(detections) == 0:
            misses.append({**td, "near": []})
            continue
        dist = np.hypot(det_xy[:, 0] - td["x_um"], det_xy[:, 1] - td["y_um"])
        near = np.nonzero(dist < max(radius_um, td.get("size_um", 0.0)))[0]
        if len(near) == 0:
            misses.append({**td, "near": []})
            continue
        matched[near] = True
        found += 1
        c["found"] += 1
        # truth may be kind-level ('extra_metal' / 'missing_metal') when topology classes are unknown
        if any(detections[i]["class"] == td["class"] or detections[i].get("kind") == td["class"] for i in near):
            classified += 1
            c["classified"] += 1
        else:
            misses.append({**td, "near": [detections[i]["class"] for i in near]})
    fa = [detections[i] for i in np.nonzero(~matched)[0]]
    return Score(
        len(truth),
        found,
        classified,
        len(fa),
        {k: {"n": v["n"], "found": v["found"], "classified": v["classified"]} for k, v in per.items()},
        misses,
        [{"class": d["class"], "area_um2": d["area_um2"], "centroid_um": d["centroid_um"]} for d in fa],
    )


def score_files(defects_json: str | Path, truth_json: str | Path, radius_um: float = 6.0) -> Score:
    dets = json.loads(Path(defects_json).read_text())["defects"]
    truth = json.loads(Path(truth_json).read_text())["defects"]
    return score(dets, truth, radius_um)

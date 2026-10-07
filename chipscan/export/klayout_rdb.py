"""Write defects as a KLayout report database (.lyrdb).

Open the GDS in KLayout, then Tools > Marker Browser > File > Load, and every
defect shows up as a marker on top of the design, grouped by class.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from chipscan.detection.defects import Defect


def _poly_str(p) -> str:
    return "(" + ";".join(f"{x:.4f},{y:.4f}" for x, y in p) + ")"


def write_rdb(
    path: str | Path,
    defects: list[Defect],
    cell_name: str,
    description: str = "chipscan defects",
    original_file: str = "",
    flip_y: bool = False,
) -> None:
    cats = sorted({d.cls for d in defects})
    lines = ['<?xml version="1.0" encoding="utf-8"?>', "<report-database>"]
    lines.append(f"  <description>{escape(description)}</description>")
    lines.append(f"  <original-file>{escape(original_file)}</original-file>")
    lines.append("  <top-cell>" + escape(cell_name) + "</top-cell>")
    lines.append("  <categories>")
    for c in cats:
        n = sum(1 for d in defects if d.cls == c)
        lines.append(f"    <category><name>{escape(c)}</name><description>{n} defects</description></category>")
    lines.append("  </categories>")
    lines.append("  <cells>")
    lines.append(f"    <cell><name>{escape(cell_name)}</name></cell>")
    lines.append("  </cells>")
    lines.append("  <items>")
    for d in defects:
        lines.append("    <item>")
        lines.append(f"      <category>'{escape(d.cls)}'</category>")
        lines.append(f"      <cell>{escape(cell_name)}</cell>")
        lines.append("      <values>")
        polys = d.polygons_um or [d.bbox_um.corners()]
        for p in polys:
            q = p.copy()
            if flip_y:
                q[:, 1] = -q[:, 1]
            lines.append(f"        <value>polygon: {_poly_str(q)}</value>")
        lines.append(
            f"        <value>text: '{escape(d.kind)} area={d.area_um2:.2f}um2 score={d.score:.2f} id={d.id}'</value>"
        )
        lines.append("      </values>")
        lines.append("    </item>")
    lines.append("  </items>")
    lines.append("</report-database>")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

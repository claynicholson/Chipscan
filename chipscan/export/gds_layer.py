"""Write defect outlines to a GDS layer so they overlay the design in any layout tool."""

from __future__ import annotations

from pathlib import Path

import gdstk

from chipscan.detection.defects import Defect


def write_defects_gds(path: str | Path, defects: list[Defect], cell_name: str, layer: int = 99, datatype: int = 0) -> None:
    lib = gdstk.Library()
    cell = lib.new_cell(cell_name)
    for d in defects:
        polys = d.polygons_um or [d.bbox_um.corners()]
        for p in polys:
            if len(p) >= 3:
                cell.add(gdstk.Polygon(p, layer=layer, datatype=datatype))
        cx, cy = d.centroid_um
        cell.add(gdstk.Label(f"{d.cls}#{d.id}", (cx, cy), layer=layer, texttype=datatype))
    lib.write_gds(str(path))

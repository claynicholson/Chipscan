"""Olympus PRECiV PositionList XML: read stage positions, write revisit lists.

The file on the T9 drive is UTF-16 with a <positionList> root containing
<SinglePositions> (points) and <Rects> (scan areas), all in stage microns.
We read both and can write a new list of single positions for the microscope
to revisit at high magnification.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StagePosition:
    id: int
    name: str
    x: float
    y: float
    z: float | None = None
    width: float | None = None
    height: float | None = None
    single: bool = True
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_rect(self) -> bool:
        return self.width is not None and self.height is not None


def _text(el: ET.Element, tag: str, default: str | None = None) -> str | None:
    c = el.find(tag)
    return c.text if c is not None and c.text is not None else default


def _f(el: ET.Element, tag: str) -> float | None:
    t = _text(el, tag)
    return None if t is None else float(t)


def read_position_list(path: str | Path) -> list[StagePosition]:
    raw = Path(path).read_bytes()
    # PRECiV writes UTF-16 with BOM; ElementTree handles it if we hand it bytes.
    root = ET.fromstring(raw)
    out: list[StagePosition] = []
    for el in root.findall("./SinglePositions/*"):
        out.append(
            StagePosition(
                id=int(_text(el, "id", "0")),
                name=_text(el, "name", "") or "",
                x=_f(el, "x") or 0.0,
                y=_f(el, "y") or 0.0,
                z=_f(el, "z"),
                single=True,
            )
        )
    for el in root.findall("./Rects/*"):
        out.append(
            StagePosition(
                id=int(_text(el, "id", "0")),
                name=_text(el, "name", "") or "",
                x=_f(el, "x") or 0.0,
                y=_f(el, "y") or 0.0,
                z=_f(el, "z"),
                width=_f(el, "width"),
                height=_f(el, "height"),
                single=(_text(el, "single") == "1"),
            )
        )
    return out


def write_position_list(path: str | Path, positions: list[StagePosition], encoding: str = "utf-16") -> None:
    root = ET.Element("positionList")
    root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    root.set("xsi:noNamespaceSchemaLocation", "PositionList.xsd")
    ET.SubElement(root, "Version").text = "1"
    singles = ET.SubElement(root, "SinglePositions")
    rects = ET.SubElement(root, "Rects")
    for p in positions:
        parent = rects if p.is_rect else singles
        el = ET.SubElement(parent, "rect" if p.is_rect else "position")
        ET.SubElement(el, "id").text = str(p.id)
        ET.SubElement(el, "x").text = repr(float(p.x))
        ET.SubElement(el, "y").text = repr(float(p.y))
        if p.z is not None:
            ET.SubElement(el, "z").text = repr(float(p.z))
        if p.is_rect:
            ET.SubElement(el, "width").text = repr(float(p.width))
            ET.SubElement(el, "height").text = repr(float(p.height))
            if p.single:
                ET.SubElement(el, "single").text = "1"
        ET.SubElement(el, "name").text = p.name
    ET.indent(root, space="\t")
    data = ET.tostring(root, encoding=encoding, xml_declaration=True)
    Path(path).write_bytes(data)

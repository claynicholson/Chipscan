"""chipscan: design-referenced optical defect inspection.

Pipeline:  image (any size)  ->  pyramid  ->  register to GDS  ->  segment tiles
           ->  compare with design  ->  defects + metrology  ->  export (KLayout, XML, JSON)

Every stage talks in *world coordinates* (microns, design frame) so that readers,
segmenters, and exporters can be swapped without touching each other.
"""

__version__ = "0.1.0"


def default_workers() -> int:
    """Processes for CPU-parallel stages: all cores but two (Windows caps a pool at 61)."""
    import os
    import sys

    n = max(1, (os.cpu_count() or 4) - 2)
    return min(n, 61) if sys.platform == "win32" else n

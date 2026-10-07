"""chipscan: design-referenced optical defect inspection.

Pipeline:  image (any size)  ->  pyramid  ->  register to GDS  ->  segment tiles
           ->  compare with design  ->  defects + metrology  ->  export (KLayout, XML, JSON)

Every stage talks in *world coordinates* (microns, design frame) so that readers,
segmenters, and exporters can be swapped without touching each other.
"""

__version__ = "0.1.0"

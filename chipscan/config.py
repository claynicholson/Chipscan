"""Pipeline configuration (pydantic models, loadable from YAML).

Every knob that makes one chip/microscope/product differ from another lives here,
so the code stays generic and a new product is a new YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class ImageConfig(BaseModel):
    path: Path
    """Giant image (PNG/JPG/TIFF/JP2) or an existing .zarr pyramid."""
    pixel_size_um: float | None = None
    """Microns per level-0 pixel. If None, estimated from the design during coarse registration."""
    channel: Literal["gray", "r", "g", "b", "max", "min"] = "gray"
    """How to collapse RGB to one channel. 'gray' is luminance."""
    pyramid_path: Path | None = None
    """Where to write/read the zarr pyramid. Defaults to <image>.zarr next to the image."""
    chunk_px: int = 1024
    invert: bool = False
    """Invert intensities on ingest (so that 'metal bright' holds if your optics make it dark)."""


class DesignConfig(BaseModel):
    path: Path
    cell: str | None = None
    """Top cell name; None picks the largest top-level cell."""
    layers: list[str] = Field(default_factory=lambda: ["*"])
    """GDS layers as 'layer/datatype', e.g. ['1/0']. '*' means all. Union of all listed layers is 'metal'."""
    polarity: Literal["drawn_is_metal", "drawn_is_etch"] = "drawn_is_metal"
    """Whether drawn polygons are metal (positive) or etched gaps in a ground plane (negative)."""
    metal_bright: bool = True
    """Whether metal appears brighter than substrate in the image."""
    region_um: list[float] | None = None
    """Optional [x0, y0, x1, y1] in um to restrict the design to a sub-region (e.g. one die)."""


class CoarseRegistrationConfig(BaseModel):
    target_pixel_size_um: float = 8.0
    """Work at this resolution for the global solve."""
    allow_flip: bool = True
    """Try mirrored hypotheses (microscope vs. layout handedness)."""
    allow_scale: bool = True
    """Solve for scale too. Set False when pixel_size_um is trusted."""
    max_scale_error: float = 0.15
    """Reject scale solutions further than this fraction from the expected value."""
    rotation_candidates_deg: list[float] = Field(default_factory=lambda: [0.0, 90.0, 180.0, 270.0])
    """Gross orientations to seed the search; fine angle is solved on top."""
    edge_sigma_px: float = 1.0
    min_score: float = 0.05


class FineRegistrationConfig(BaseModel):
    enabled: bool = True
    level_pixel_size_um: float = 2.0
    """Pyramid level resolution used for local matching."""
    tile_px: int = 512
    grid: int = 8
    """Sample an N x N grid of tiles across the registered overlap."""
    max_shift_px: int = 48
    min_confidence: float = 0.08
    """Phase-correlation peak height below which a tile is ignored."""
    residual_field: bool = True
    """Keep the per-tile residuals as a warp field (absorbs stitching seams)."""


class RegistrationConfig(BaseModel):
    coarse: CoarseRegistrationConfig = CoarseRegistrationConfig()
    fine: FineRegistrationConfig = FineRegistrationConfig()
    transform_path: Path | None = None
    """If set and exists, skip solving and load this warp field."""


class SegmentationConfig(BaseModel):
    method: Literal["classical", "unet"] = "classical"
    tile_px: int = 1024
    margin_px: int = 64
    """Context margin: each tile is read with this margin and only the centre is kept."""
    threshold: float = 0.5
    # classical
    blur_sigma: float = 1.0
    classical_method: Literal["otsu", "adaptive", "design_prior"] = "design_prior"
    adaptive_block_px: int = 101
    morph_px: int = 0
    # unet
    model_path: Path | None = None
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    base_channels: int = 24
    depth: int = 4
    batch_size: int = 4


class DetectionConfig(BaseModel):
    tolerance_px: int = 2
    """Ignore differences within this distance of a design edge (registration slop, edge roughness)."""
    min_area_um2: float = 4.0
    min_area_px: int = 6
    merge_margin_um: float = 2.0
    """Defects closer than this across tile borders are merged."""
    classify: bool = True


class MetrologyCut(BaseModel):
    name: str
    p0_um: list[float]
    p1_um: list[float]
    expect_um: list[float] | None = None
    """Optional expected run lengths, e.g. [gap, center, gap] for a CPW cut."""


class MetrologyConfig(BaseModel):
    cuts: list[MetrologyCut] = Field(default_factory=list)
    sample_step_um: float | None = None
    """Sampling step along cuts; defaults to a quarter pixel."""


class StageFrameConfig(BaseModel):
    """How to turn design um into microscope stage um (for revisit position lists)."""

    enabled: bool = False
    origin_stage_um: list[float] = Field(default_factory=lambda: [0.0, 0.0])
    """Stage coordinates of design origin (0,0)."""
    flip_y: bool = False
    z_um: float | None = None


class ExportConfig(BaseModel):
    out_dir: Path = Path("outputs")
    klayout_rdb: bool = True
    gds_layer: str | None = "99/0"
    """Write defects as polygons on this layer of a defects GDS. None disables."""
    positions_xml: bool = True
    overlay_png: bool = True
    overlay_max_px: int = 4000
    save_pred_mask: bool = True


class InjectionParams(BaseModel):
    enabled: bool = True
    p_defect_tile: float = 0.85
    max_per_tile: int = 4
    min_radius_px: float = 1.5
    max_radius_px: float = 12.0
    p_bar: float = 0.35
    edge_softness_px: float = 0.7
    texture_noise: float = 1.0


class TrainConfig(BaseModel):
    """Training run: one or more registered scenes (each a pipeline YAML) + hyper-parameters."""

    scenes: list[Path]
    """Pipeline configs; each is registered (or its saved transform loaded) and becomes a scene."""
    out: Path = Path("models/unet.pt")
    epochs: int = 10
    samples_per_epoch: int = 2000
    tile_px: int = 256
    batch_size: int = 8
    lr: float = 1e-3
    base_channels: int = 24
    depth: int = 4
    use_prior: bool = True
    prior_jitter_px: float = 1.0
    ignore_px: int = 1
    val_frac: float = 0.1
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    num_workers: int = 0
    defect_pos_weight: float = 4.0
    init_from: Path | None = None
    seed: int = 0
    injection: InjectionParams = InjectionParams()

    @staticmethod
    def load(path: str | Path) -> TrainConfig:
        p = Path(path)
        cfg = TrainConfig.model_validate(yaml.safe_load(p.read_text()) or {})
        base = p.parent
        cfg.scenes = [(base / s).resolve() if not Path(s).is_absolute() else Path(s) for s in cfg.scenes]
        if not cfg.out.is_absolute():
            cfg.out = (base / cfg.out).resolve()
        if cfg.init_from is not None and not cfg.init_from.is_absolute():
            cfg.init_from = (base / cfg.init_from).resolve()
        return cfg


class PipelineConfig(BaseModel):
    name: str = "inspection"
    image: ImageConfig
    design: DesignConfig
    registration: RegistrationConfig = RegistrationConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    detection: DetectionConfig = DetectionConfig()
    metrology: MetrologyConfig = MetrologyConfig()
    stage: StageFrameConfig = StageFrameConfig()
    export: ExportConfig = ExportConfig()

    @staticmethod
    def load(path: str | Path) -> PipelineConfig:
        p = Path(path)
        data = yaml.safe_load(p.read_text()) or {}
        cfg = PipelineConfig.model_validate(data)
        # resolve relative paths against the config file's directory
        base = p.parent
        for field, obj in (("path", cfg.image), ("pyramid_path", cfg.image), ("path", cfg.design),
                           ("transform_path", cfg.registration), ("model_path", cfg.segmentation),
                           ("out_dir", cfg.export)):
            v = getattr(obj, field)
            if v is not None and not Path(v).is_absolute():
                setattr(obj, field, (base / v).resolve())
        return cfg

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))

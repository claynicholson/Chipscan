from chipscan.segmentation.base import Segmenter, to_float01
from chipscan.segmentation.classical import ClassicalSegmenter


def build_segmenter(cfg, metal_bright: bool = True) -> Segmenter:
    """Factory from a SegmentationConfig."""
    if cfg.method == "classical":
        return ClassicalSegmenter(
            method=cfg.classical_method,
            metal_bright=metal_bright,
            blur_sigma=cfg.blur_sigma,
            adaptive_block_px=cfg.adaptive_block_px,
            morph_px=cfg.morph_px,
        )
    if cfg.method == "unet":
        from chipscan.segmentation.unet import UNetSegmenter

        if cfg.model_path is None:
            raise ValueError("segmentation.model_path is required for method=unet")
        return UNetSegmenter(cfg.model_path, device=cfg.device, batch_size=cfg.batch_size)
    raise ValueError(f"unknown segmentation method {cfg.method!r}")


__all__ = ["ClassicalSegmenter", "Segmenter", "build_segmenter", "to_float01"]

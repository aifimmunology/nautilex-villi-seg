"""SAM2-based zero-shot villi segmentation.

Uses SAM2 automatic mask generator on a downscaled DAPI morphology image
to segment villi (large tissue structures ~100-500 um diameter) without
any training data or prompts.

Algorithm:
  1. Max-project morphology Z-stack
  2. Downscale 4x for memory efficiency
  3. Percentile-based contrast stretch to uint8
  4. SAM2 automatic mask generation (tiled if image is large)
  5. Filter masks by area and aspect ratio
  6. Vectorize to Shapely polygons in micron coordinates
  7. Merge overlapping polygons
  8. Save diagnostic overlay
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from utils.coords import PIXEL_SIZE_UM
from utils.io import load_morphology
from utils.tiles import extract_tiles, stitch_masks

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DOWNSCALE_FACTOR = 4
TILE_SIZE_DS = 1024  # tile size at downscale (= 4096 px original)
TILE_OVERLAP_DS = 128
MIN_MASK_AREA_DS = 500  # min area in downscaled pixels (~5000 um^2)
MAX_ASPECT_RATIO = 5.0
MERGE_IOU_THRESH = 0.3

# SAM2 defaults (configurable via environment)
SAM2_CHECKPOINT_DEFAULT = "/models/sam2.1_hiera_large.pt"
SAM2_CONFIG_DEFAULT = "configs/sam2.1/sam2.1_hiera_l.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _downscale(image: np.ndarray, factor: int) -> np.ndarray:
    """Downscale 2D image by integer factor via block-mean."""
    h, w = image.shape[:2]
    new_h = h // factor * factor
    new_w = w // factor * factor
    cropped = image[:new_h, :new_w]
    return cropped.reshape(new_h // factor, factor, new_w // factor, factor).mean(
        axis=(1, 3)
    )


def _normalize_uint8(
    image: np.ndarray, p_low: float = 2, p_high: float = 98
) -> np.ndarray:
    """Percentile-based contrast stretch to uint8."""
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(image, lo, hi)
    scaled = (clipped - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stack single-channel grayscale into pseudo-RGB (H, W, 3)."""
    return np.stack([gray, gray, gray], axis=-1)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Get (y0, x0, y1, x1) bounding box of a binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return int(y0), int(x0), int(y1 + 1), int(x1 + 1)


def _mask_aspect_ratio(mask: np.ndarray) -> float:
    """Aspect ratio of the mask bounding box (always >= 1)."""
    y0, x0, y1, x1 = _mask_bbox(mask)
    h = max(y1 - y0, 1)
    w = max(x1 - x0, 1)
    return max(h / w, w / h)


def _mask_to_polygon(
    mask: np.ndarray,
    offset_y: float = 0.0,
    offset_x: float = 0.0,
    scale: float = 1.0,
) -> Polygon | None:
    """Convert binary mask to shapely Polygon via contour finding.

    Applies offset (for tile stitching) and scale (for upscale + micron
    conversion) to the contour coordinates.
    """
    from skimage.measure import find_contours

    contours = find_contours(mask.astype(np.uint8), level=0.5)
    if not contours:
        return None

    # Take the longest contour
    contour = max(contours, key=len)
    if len(contour) < 4:
        return None

    # contour is (N, 2) in (row, col) = (y, x) order
    coords = np.column_stack(
        [
            (contour[:, 1] + offset_x) * scale,  # x in microns
            (contour[:, 0] + offset_y) * scale,  # y in microns
        ]
    )

    try:
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            return None
        # Ensure we return a Polygon, not a MultiPolygon from buffer(0)
        if isinstance(poly, MultiPolygon):
            poly = max(poly.geoms, key=lambda g: g.area)
        return poly
    except Exception:
        return None


def _compute_iou(p1: Polygon, p2: Polygon) -> float:
    """Intersection over union of two polygons."""
    if not p1.intersects(p2):
        return 0.0
    try:
        inter = p1.intersection(p2).area
        union = p1.union(p2).area
        return inter / union if union > 0 else 0.0
    except Exception:
        return 0.0


def _merge_overlapping(polygons: list[Polygon], iou_thresh: float) -> list[Polygon]:
    """Greedy merge of overlapping polygons by IoU threshold."""
    if len(polygons) <= 1:
        return polygons

    from shapely.strtree import STRtree

    merged = list(polygons)
    changed = True

    while changed:
        changed = False
        tree = STRtree(merged)
        used = set()
        new_merged = []

        for i, poly_i in enumerate(merged):
            if i in used:
                continue
            # Find candidate overlaps via spatial index
            candidates = tree.query(poly_i)
            group = [poly_i]
            for j in candidates:
                if j <= i or j in used:
                    continue
                if _compute_iou(poly_i, merged[j]) >= iou_thresh:
                    group.append(merged[j])
                    used.add(j)
                    changed = True
            used.add(i)

            if len(group) > 1:
                union_geom = unary_union(group)
                if isinstance(union_geom, MultiPolygon):
                    new_merged.extend(union_geom.geoms)
                else:
                    new_merged.append(union_geom)
            else:
                new_merged.append(poly_i)

        merged = new_merged

    return merged


def _log_gpu_memory(label: str = "") -> None:
    """Log current GPU memory usage if CUDA is available."""
    try:
        import torch

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(
                "GPU memory %s: %.2f GB allocated, %.2f GB reserved, %.2f GB total",
                label,
                allocated,
                reserved,
                total,
            )
    except Exception:
        pass


def _save_diagnostic_png(
    image_uint8: np.ndarray,
    masks: list[dict],
    output_path: Path,
) -> None:
    """Save a diagnostic overlay of SAM2 masks on the image."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb

        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        ax.imshow(image_uint8, cmap="gray")

        n_masks = len(masks)
        for idx, mask_info in enumerate(masks):
            mask = mask_info["segmentation"]
            # Deterministic color per mask
            hue = idx / max(n_masks, 1)
            color = list(hsv_to_rgb([hue, 0.8, 0.9])) + [0.35]

            overlay = np.zeros((*mask.shape, 4))
            overlay[mask] = color
            ax.imshow(overlay)

        ax.set_title(f"SAM2 masks: {n_masks} detected", fontsize=14)
        ax.axis("off")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        logger.info("Saved diagnostic PNG: %s", output_path)
    except Exception as e:
        logger.warning("Could not save diagnostic PNG: %s", e)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi from Xenium morphology image using SAM2.

    Parameters
    ----------
    data_path : str
        Path to the Xenium dataset directory containing morphology.ome.tif.
    output_dir : str
        Directory for output artifacts (diagnostic PNGs, intermediate masks).

    Returns
    -------
    list[Polygon]
        Villi boundaries as shapely Polygons in micron coordinates.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 0. Import SAM2 (fail gracefully if not installed)
    # ------------------------------------------------------------------
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except ImportError as e:
        logger.error(
            "SAM2 is not installed. Install with: pip install sam2. Error: %s", e
        )
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ------------------------------------------------------------------
    # 1. Load and preprocess morphology image
    # ------------------------------------------------------------------
    logger.info("Loading morphology image from %s ...", data_path)
    t0 = time.time()
    raw = load_morphology(data_path, max_project=True)
    logger.info(
        "Loaded morphology: shape=%s, dtype=%s (%.1fs)",
        raw.shape,
        raw.dtype,
        time.time() - t0,
    )

    # Downscale for memory
    logger.info("Downscaling %dx ...", DOWNSCALE_FACTOR)
    ds = _downscale(raw.astype(np.float32), DOWNSCALE_FACTOR)
    ds_h, ds_w = ds.shape
    logger.info("Downscaled shape: %s", ds.shape)

    # Normalize to uint8
    img8 = _normalize_uint8(ds)
    logger.info(
        "Contrast stretch: [%.0f, %.0f] -> [0, 255]",
        np.percentile(ds, 2),
        np.percentile(ds, 98),
    )

    # ------------------------------------------------------------------
    # 2. Build SAM2 model
    # ------------------------------------------------------------------
    checkpoint = os.environ.get("SAM2_CHECKPOINT", SAM2_CHECKPOINT_DEFAULT)
    config = os.environ.get("SAM2_CONFIG", SAM2_CONFIG_DEFAULT)

    logger.info("Loading SAM2 model: checkpoint=%s, config=%s", checkpoint, config)
    _log_gpu_memory("before model load")

    try:
        sam2_model = build_sam2(
            config_file=config,
            ckpt_path=checkpoint,
            device=str(device),
        )
        _log_gpu_memory("after model load")
    except Exception as e:
        logger.error("Failed to load SAM2 model: %s", e)
        return []

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2_model,
        points_per_side=32,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.8,
        min_mask_region_area=MIN_MASK_AREA_DS,
    )

    # ------------------------------------------------------------------
    # 3. Generate masks (tiled if image is large)
    # ------------------------------------------------------------------
    # Threshold: if the downscaled image fits in a single tile, skip tiling
    single_shot_limit = 1536  # px — single-shot if both dims are under this
    use_tiling = ds_h > single_shot_limit or ds_w > single_shot_limit

    all_masks: list[dict] = []

    if use_tiling:
        logger.info(
            "Image too large for single-shot (%dx%d). Tiling at %dx%d with %dpx overlap.",
            ds_h,
            ds_w,
            TILE_SIZE_DS,
            TILE_SIZE_DS,
            TILE_OVERLAP_DS,
        )

        tiles = extract_tiles(img8, tile_size=TILE_SIZE_DS, overlap=TILE_OVERLAP_DS)
        logger.info("Extracted %d tiles", len(tiles))

        for tile_idx, (tile_gray, (y0, x0, y1, x1)) in enumerate(tiles):
            tile_rgb = _to_rgb(tile_gray)
            logger.info(
                "Processing tile %d/%d: (%d,%d)-(%d,%d)",
                tile_idx + 1,
                len(tiles),
                y0,
                x0,
                y1,
                x1,
            )

            t_tile = time.time()
            try:
                tile_masks = mask_generator.generate(tile_rgb)
            except Exception as e:
                logger.warning("SAM2 failed on tile %d: %s", tile_idx, e)
                continue
            logger.info(
                "  Tile %d: %d masks in %.1fs",
                tile_idx,
                len(tile_masks),
                time.time() - t_tile,
            )

            # Offset masks to global coordinates
            for m in tile_masks:
                # Crop mask to actual tile bounds (remove padding)
                actual_h = y1 - y0
                actual_w = x1 - x0
                m["segmentation"] = m["segmentation"][:actual_h, :actual_w]
                # Store tile offset for later polygon conversion
                m["_tile_offset_y"] = y0
                m["_tile_offset_x"] = x0
                all_masks.append(m)

            _log_gpu_memory(f"after tile {tile_idx}")
    else:
        logger.info("Running single-shot SAM2 on %dx%d image", ds_h, ds_w)
        img_rgb = _to_rgb(img8)

        t_gen = time.time()
        try:
            all_masks = mask_generator.generate(img_rgb)
        except Exception as e:
            logger.error("SAM2 mask generation failed: %s", e)
            return []
        logger.info("Generated %d masks in %.1fs", len(all_masks), time.time() - t_gen)

        # No tile offset for single-shot
        for m in all_masks:
            m["_tile_offset_y"] = 0
            m["_tile_offset_x"] = 0

    logger.info("Total raw masks: %d", len(all_masks))

    # ------------------------------------------------------------------
    # 4. Filter masks by area and aspect ratio
    # ------------------------------------------------------------------
    filtered_masks = []
    for m in all_masks:
        seg = m["segmentation"]
        area = seg.sum()

        if area < MIN_MASK_AREA_DS:
            continue

        aspect = _mask_aspect_ratio(seg)
        if aspect > MAX_ASPECT_RATIO:
            continue

        filtered_masks.append(m)

    logger.info(
        "After filtering: %d / %d masks (area > %d ds-px, aspect < %.1f)",
        len(filtered_masks),
        len(all_masks),
        MIN_MASK_AREA_DS,
        MAX_ASPECT_RATIO,
    )

    if not filtered_masks:
        logger.warning("No masks passed filtering. Returning empty list.")
        _save_diagnostic_png(img8, [], output_path / "sam2_diagnostic.png")
        return []

    # ------------------------------------------------------------------
    # 5. Convert masks to polygons in micron coordinates
    # ------------------------------------------------------------------
    # Scale factor: downscale pixels -> original pixels -> microns
    scale_to_micron = DOWNSCALE_FACTOR * PIXEL_SIZE_UM

    polygons: list[Polygon] = []
    for m in filtered_masks:
        seg = m["segmentation"]
        offset_y = m["_tile_offset_y"]
        offset_x = m["_tile_offset_x"]

        poly = _mask_to_polygon(
            seg,
            offset_y=offset_y,
            offset_x=offset_x,
            scale=scale_to_micron,
        )
        if poly is not None:
            polygons.append(poly)

    logger.info(
        "Vectorized %d polygons from %d masks", len(polygons), len(filtered_masks)
    )

    # ------------------------------------------------------------------
    # 6. Merge overlapping polygons
    # ------------------------------------------------------------------
    if len(polygons) > 1:
        logger.info("Merging overlapping polygons (IoU > %.2f) ...", MERGE_IOU_THRESH)
        t_merge = time.time()
        polygons = _merge_overlapping(polygons, MERGE_IOU_THRESH)
        logger.info(
            "After merge: %d polygons (%.1fs)", len(polygons), time.time() - t_merge
        )

    # Ensure all polygons are valid
    clean_polygons = []
    for p in polygons:
        if not p.is_valid:
            p = p.buffer(0)
        if isinstance(p, MultiPolygon):
            clean_polygons.extend(g for g in p.geoms if g.area > 0)
        elif p.area > 0:
            clean_polygons.append(p)
    polygons = clean_polygons

    logger.info("Final polygon count: %d", len(polygons))

    # Log area statistics
    if polygons:
        areas = [p.area for p in polygons]
        logger.info(
            "Area stats (um^2): min=%.0f, median=%.0f, max=%.0f, mean=%.0f",
            np.min(areas),
            np.median(areas),
            np.max(areas),
            np.mean(areas),
        )

    # ------------------------------------------------------------------
    # 7. Save diagnostic PNG
    # ------------------------------------------------------------------
    _save_diagnostic_png(img8, filtered_masks, output_path / "sam2_diagnostic.png")

    # ------------------------------------------------------------------
    # 8. Free GPU memory
    # ------------------------------------------------------------------
    try:
        import torch

        del sam2_model, mask_generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _log_gpu_memory("after cleanup")
    except Exception:
        pass

    return polygons

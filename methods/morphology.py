"""Villi segmentation via DAPI morphology image (classical CV).

Max-projects the Z-stack, downscales for speed, applies Otsu + watershed
to separate individual villi.  All output polygons are in micron
coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import find_contours, label
from skimage.morphology import (
    disk,
    binary_closing,
    binary_opening,
    remove_small_objects,
)
from skimage.segmentation import watershed
from skimage.transform import rescale

from utils.coords import PIXEL_SIZE_UM
from utils.io import load_morphology


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi from DAPI morphology using classical CV.

    Parameters
    ----------
    data_path : str
        Root directory containing ``morphology.ome.tif``.
    output_dir : str
        Directory to write diagnostic PNGs.

    Returns
    -------
    list[Polygon]
        Villi polygons in micron coordinates.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load morphology (max-projected)
    # ------------------------------------------------------------------
    print(f"[{_ts()}] morphology: loading DAPI image ...")
    img = load_morphology(data_path, max_project=True)
    print(f"[{_ts()}] morphology: image shape {img.shape}, dtype {img.dtype}")

    if img.size == 0:
        print(f"[{_ts()}] morphology: empty image — returning empty list")
        return []

    # ------------------------------------------------------------------
    # 2. Downscale 4× for speed
    # ------------------------------------------------------------------
    scale_factor = 0.25
    inv_scale = int(1 / scale_factor)  # 4
    print(f"[{_ts()}] morphology: downscaling {inv_scale}× ...")

    # Normalise to float [0, 1] for rescale
    img_f = img.astype(np.float32)
    img_max = img_f.max()
    if img_max > 0:
        img_f /= img_max

    img_ds = rescale(img_f, scale_factor, anti_aliasing=True, preserve_range=True)
    print(f"[{_ts()}] morphology: downscaled shape {img_ds.shape}")

    # ------------------------------------------------------------------
    # 3. Gaussian blur sigma=8 (on downscaled image)
    # ------------------------------------------------------------------
    img_blur = gaussian(img_ds, sigma=8, preserve_range=True)

    # ------------------------------------------------------------------
    # 4. Otsu threshold → binary mask
    # ------------------------------------------------------------------
    nonzero = img_blur[img_blur > 0]
    if nonzero.size == 0:
        print(f"[{_ts()}] morphology: blurred image all zeros — returning empty")
        return []

    thresh = threshold_otsu(nonzero)
    mask = img_blur >= thresh
    print(
        f"[{_ts()}] morphology: Otsu threshold = {thresh:.4f} "
        f"({mask.sum():,} / {mask.size:,} pixels)"
    )

    # ------------------------------------------------------------------
    # 5. Morphological closing(15) then opening(8)
    # ------------------------------------------------------------------
    mask = binary_closing(mask, footprint=disk(15))
    mask = binary_opening(mask, footprint=disk(8))

    # ------------------------------------------------------------------
    # 6. Remove small objects (< 5000 pixels at downscale)
    # ------------------------------------------------------------------
    min_obj_px = 5000
    mask = remove_small_objects(mask, min_size=min_obj_px)
    print(f"[{_ts()}] morphology: removed objects < {min_obj_px} downscaled px")

    if not mask.any():
        print(f"[{_ts()}] morphology: mask empty after cleaning — returning empty")
        return []

    # ------------------------------------------------------------------
    # 7. Distance transform → local maxima → markers
    # ------------------------------------------------------------------
    dist = ndimage.distance_transform_edt(mask)
    local_max_coords = peak_local_max(dist, min_distance=50, labels=mask)

    if len(local_max_coords) == 0:
        print(f"[{_ts()}] morphology: no local maxima — treating as one region")
        markers = label(mask)
    else:
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (r, c) in enumerate(local_max_coords, start=1):
            markers[r, c] = i
        print(f"[{_ts()}] morphology: {len(local_max_coords)} watershed seeds")

    # ------------------------------------------------------------------
    # 8. Watershed
    # ------------------------------------------------------------------
    labels_ws = watershed(-dist, markers, mask=mask)
    n_regions = labels_ws.max()
    print(f"[{_ts()}] morphology: watershed produced {n_regions} regions")

    # ------------------------------------------------------------------
    # 9. Extract contours → scale up → micron coordinates → Polygons
    # ------------------------------------------------------------------
    # downscale pixel → full-res pixel: ×inv_scale
    # full-res pixel → micron: ×PIXEL_SIZE_UM
    px_to_um = inv_scale * PIXEL_SIZE_UM

    min_polygon_area_um2 = 5000.0
    polygons: list[Polygon] = []

    for region_id in range(1, n_regions + 1):
        region_mask = (labels_ws == region_id).astype(np.uint8)
        contours = find_contours(region_mask, level=0.5)
        if not contours:
            continue

        contour = max(contours, key=len)
        if len(contour) < 4:
            continue

        # contour is (row, col) in downscaled pixels → (x_um, y_um)
        coords_um = np.column_stack(
            [
                contour[:, 1] * px_to_um,  # col → x micron
                contour[:, 0] * px_to_um,  # row → y micron
            ]
        )

        try:
            poly = Polygon(coords_um)
            if not poly.is_valid:
                poly = poly.buffer(0)
            if poly.is_empty:
                continue
            if poly.area >= min_polygon_area_um2:
                polygons.append(poly)
        except Exception:
            continue

    print(
        f"[{_ts()}] morphology: {len(polygons)} polygons after area filter "
        f"(>= {min_polygon_area_um2} µm²)"
    )

    # ------------------------------------------------------------------
    # 10-11. Save diagnostic PNG
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img_ds, cmap="gray")
    axes[0].set_title("Downscaled DAPI (max-proj)")

    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title(f"Binary mask (Otsu={thresh:.4f})")

    cmap_ws = plt.cm.nipy_spectral.copy()
    cmap_ws.set_under("black")
    axes[2].imshow(
        np.where(labels_ws > 0, labels_ws, np.nan),
        cmap="nipy_spectral",
        interpolation="nearest",
    )
    axes[2].set_title(f"Watershed ({len(polygons)} villi)")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("col (downscaled px)")
        ax.set_ylabel("row (downscaled px)")

    fig.tight_layout()
    png_path = out / "morphology_intermediate.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}] morphology: saved diagnostic → {png_path}")

    return polygons

"""Villi segmentation via transcript density + watershed.

Builds a 2D transcript density map, thresholds it, and uses watershed
to separate individual villi regions.  All output polygons are in micron
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
from skimage.filters import threshold_otsu
from skimage.measure import find_contours, label
from skimage.morphology import disk, binary_opening, remove_small_objects
from skimage.segmentation import watershed

from utils.io import load_transcripts


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi from transcript density using watershed.

    Parameters
    ----------
    data_path : str
        Root directory containing ``transcripts.parquet``.
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
    # 1. Load transcripts
    # ------------------------------------------------------------------
    print(f"[{_ts()}] density: loading transcripts (QV >= 20) ...")
    tx = load_transcripts(data_path, min_qv=20.0)
    if tx.empty:
        print(f"[{_ts()}] density: no transcripts found — returning empty list")
        return []

    x = tx["x_location"].values
    y = tx["y_location"].values
    print(f"[{_ts()}] density: {len(tx):,} transcripts loaded")

    # ------------------------------------------------------------------
    # 2. 2D histogram (5 µm bins)
    # ------------------------------------------------------------------
    bin_size_um = 5.0
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_edges = np.arange(x_min, x_max + bin_size_um, bin_size_um)
    y_edges = np.arange(y_min, y_max + bin_size_um, bin_size_um)

    density, _, _ = np.histogram2d(y, x, bins=[y_edges, x_edges])
    print(
        f"[{_ts()}] density: histogram shape {density.shape}, bin_size={bin_size_um} µm"
    )

    # ------------------------------------------------------------------
    # 3. Gaussian smooth (sigma = 5 bins = 25 µm)
    # ------------------------------------------------------------------
    sigma_bins = 5
    density_smooth = ndimage.gaussian_filter(density, sigma=sigma_bins)
    print(
        f"[{_ts()}] density: smoothed with sigma={sigma_bins} bins "
        f"({sigma_bins * bin_size_um} µm)"
    )

    # ------------------------------------------------------------------
    # 4. Otsu threshold → binary mask
    # ------------------------------------------------------------------
    if density_smooth.max() == 0:
        print(f"[{_ts()}] density: density map is all zeros — returning empty")
        return []

    thresh = threshold_otsu(density_smooth[density_smooth > 0])
    mask = density_smooth >= thresh
    print(f"[{_ts()}] density: Otsu threshold = {thresh:.2f}")

    # ------------------------------------------------------------------
    # 5. Morphological opening (disk(3))
    # ------------------------------------------------------------------
    mask = binary_opening(mask, footprint=disk(3))

    # ------------------------------------------------------------------
    # 6. Remove small objects (< 2000 µm² → bins)
    # ------------------------------------------------------------------
    min_area_um2 = 2000.0
    min_area_bins = int(min_area_um2 / (bin_size_um**2))
    mask = remove_small_objects(mask, min_size=max(min_area_bins, 1))
    print(
        f"[{_ts()}] density: removed objects < {min_area_um2} µm² "
        f"({min_area_bins} bins)"
    )

    if not mask.any():
        print(f"[{_ts()}] density: mask empty after cleaning — returning empty")
        return []

    # ------------------------------------------------------------------
    # 7. Distance transform → local maxima → watershed markers
    # ------------------------------------------------------------------
    dist = ndimage.distance_transform_edt(mask)
    min_distance_bins = 20
    local_max_coords = peak_local_max(dist, min_distance=min_distance_bins, labels=mask)

    if len(local_max_coords) == 0:
        print(f"[{_ts()}] density: no local maxima found — treating as one region")
        markers = label(mask)
    else:
        markers = np.zeros_like(mask, dtype=np.int32)
        for i, (r, c) in enumerate(local_max_coords, start=1):
            markers[r, c] = i
        print(f"[{_ts()}] density: {len(local_max_coords)} watershed seeds")

    # ------------------------------------------------------------------
    # 8. Watershed
    # ------------------------------------------------------------------
    labels_ws = watershed(-dist, markers, mask=mask)
    n_regions = labels_ws.max()
    print(f"[{_ts()}] density: watershed produced {n_regions} regions")

    # ------------------------------------------------------------------
    # 9-10. Extract contours → Polygons in microns, filter by area
    # ------------------------------------------------------------------
    min_polygon_area_um2 = 5000.0
    polygons: list[Polygon] = []

    for region_id in range(1, n_regions + 1):
        region_mask = (labels_ws == region_id).astype(np.uint8)
        contours = find_contours(region_mask, level=0.5)
        if not contours:
            continue

        # Take the longest contour
        contour = max(contours, key=len)
        if len(contour) < 4:
            continue

        # Contour is in (row, col) → convert to micron (x, y)
        # row → y, col → x; bins → microns
        coords_um = np.column_stack(
            [
                contour[:, 1] * bin_size_um + x_min,  # col → x micron
                contour[:, 0] * bin_size_um + y_min,  # row → y micron
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
        f"[{_ts()}] density: {len(polygons)} polygons after area filter "
        f"(>= {min_polygon_area_um2} µm²)"
    )

    # ------------------------------------------------------------------
    # 11. Save diagnostic PNG
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(density_smooth, origin="lower", cmap="hot")
    axes[0].set_title("Smoothed density")
    axes[0].set_xlabel("x bin")
    axes[0].set_ylabel("y bin")

    axes[1].imshow(mask, origin="lower", cmap="gray")
    axes[1].set_title(f"Binary mask (Otsu={thresh:.1f})")

    axes[2].imshow(labels_ws, origin="lower", cmap="nipy_spectral")
    axes[2].set_title(f"Watershed ({len(polygons)} villi)")

    for ax in axes:
        ax.set_aspect("equal")

    fig.tight_layout()
    png_path = out / "density_intermediate.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}] density: saved diagnostic → {png_path}")

    return polygons

"""Output helpers: GeoJSON, cell-to-villus CSV, metrics JSON."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import mapping, Polygon, Point

from utils.coords import micron_to_pixel


def save_geojson(
    polygons: list[Polygon],
    output_path: str | Path,
    save_pixels: bool = True,
) -> Path:
    """Save polygons as GeoJSON FeatureCollection.
    Polygons are expected in micron coords; optionally also stores pixel coords.
    """
    features = []
    for i, poly in enumerate(polygons):
        if not poly.is_valid:
            poly = poly.buffer(0)
        feat = {
            "type": "Feature",
            "properties": {"villus_id": i, "area_um2": poly.area},
            "geometry": mapping(poly),
        }
        if save_pixels:
            px_poly = micron_to_pixel(poly)
            feat["properties"]["geometry_pixels"] = mapping(px_poly)
        features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fc, indent=2))
    return out


def assign_cells_to_villi(
    cells_df: pd.DataFrame,
    polygons: list[Polygon],
    x_col: str = "x_centroid",
    y_col: str = "y_centroid",
) -> pd.Series:
    """Point-in-polygon assignment of cells to villi.
    Returns Series of villus IDs (-1 = unassigned).
    """
    from shapely.strtree import STRtree

    # Build spatial index
    tree = STRtree(polygons)
    labels = np.full(len(cells_df), -1, dtype=np.int32)

    xs = cells_df[x_col].values
    ys = cells_df[y_col].values

    for idx in range(len(cells_df)):
        pt = Point(xs[idx], ys[idx])
        hits = tree.query(pt)
        for hit_idx in hits:
            if polygons[hit_idx].contains(pt):
                labels[idx] = hit_idx
                break

    return pd.Series(labels, index=cells_df.index, name="villus_id")


def save_cell_villus_map(
    cells_df: pd.DataFrame,
    villus_labels: pd.Series,
    output_path: str | Path,
) -> Path:
    """Save cell-to-villus mapping as CSV."""
    out_df = (
        cells_df[["cell_id"]].copy()
        if "cell_id" in cells_df.columns
        else cells_df.iloc[:, :1].copy()
    )
    out_df["villus_id"] = villus_labels.values
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    return out


def save_metrics(metrics: dict, output_path: str | Path) -> Path:
    """Save metrics dict as JSON."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    return out

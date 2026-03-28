"""Villi segmentation via Delaunay cell graph + connected components.

Builds a spatial graph from cell centroids, prunes long edges, finds
connected components, and wraps each cluster in a concave hull.  All
output polygons are in micron coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from scipy.sparse.csgraph import connected_components
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import unary_union

from utils.io import load_cells


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _concave_hull(points: np.ndarray, buffer_um: float = 15.0) -> Polygon | None:
    """Build a concave-ish hull from a point cloud.

    Strategy: try ``shapely.concave_hull`` (Shapely >= 2.0) first.
    Fall back to buffered union → simplify → exterior.
    """
    mp = MultiPoint(points)

    # Attempt 1: shapely.concave_hull (ratio=0.3 gives a moderately tight hull)
    try:
        from shapely import concave_hull as _concave_hull_fn

        hull = _concave_hull_fn(mp, ratio=0.3)
        if hull is not None and not hull.is_empty:
            # Ensure we return a Polygon (concave_hull can give LineString for
            # degenerate inputs)
            if hull.geom_type == "Polygon":
                return hull
    except (ImportError, AttributeError):
        pass

    # Attempt 2: buffer each point → union → simplify
    buffered = unary_union(
        [mp.geoms[i].buffer(buffer_um) for i in range(len(mp.geoms))]
    )
    if buffered.is_empty:
        return None

    # unary_union of buffers can produce MultiPolygon; take the largest
    if buffered.geom_type == "MultiPolygon":
        buffered = max(buffered.geoms, key=lambda g: g.area)

    simplified = buffered.simplify(tolerance=buffer_um * 0.5, preserve_topology=True)
    if simplified.geom_type == "Polygon":
        return simplified
    return None


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi from a Delaunay cell graph.

    Parameters
    ----------
    data_path : str
        Root directory containing ``cells.parquet``.
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
    # 1. Load cells
    # ------------------------------------------------------------------
    print(f"[{_ts()}] graph: loading cells ...")
    cells = load_cells(data_path)
    if cells.empty:
        print(f"[{_ts()}] graph: no cells found — returning empty list")
        return []

    print(f"[{_ts()}] graph: {len(cells):,} cells loaded")

    # ------------------------------------------------------------------
    # 2. Extract centroids (N, 2) in microns
    # ------------------------------------------------------------------
    coords = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
    n_cells = len(coords)

    if n_cells < 4:
        print(f"[{_ts()}] graph: too few cells ({n_cells}) for triangulation")
        return []

    # ------------------------------------------------------------------
    # 3. Delaunay triangulation
    # ------------------------------------------------------------------
    print(f"[{_ts()}] graph: computing Delaunay triangulation ...")
    tri = Delaunay(coords)
    simplices = tri.simplices  # (M, 3)
    print(f"[{_ts()}] graph: {len(simplices):,} simplices")

    # ------------------------------------------------------------------
    # 4. Compute all edge lengths
    # ------------------------------------------------------------------
    # Extract unique edges from simplices
    edges_set: set[tuple[int, int]] = set()
    for s in simplices:
        for i in range(3):
            a, b = int(s[i]), int(s[(i + 1) % 3])
            edges_set.add((min(a, b), max(a, b)))

    edges = np.array(list(edges_set), dtype=np.int64)  # (E, 2)
    diffs = coords[edges[:, 0]] - coords[edges[:, 1]]
    lengths = np.linalg.norm(diffs, axis=1)

    print(
        f"[{_ts()}] graph: {len(edges):,} unique edges, "
        f"median={np.median(lengths):.1f} µm, "
        f"mean={lengths.mean():.1f} µm"
    )

    # ------------------------------------------------------------------
    # 5. Adaptive threshold: use 90th percentile of edge lengths.
    #    Delaunay produces long hull edges that inflate std, so
    #    percentile-based thresholds are more robust than mean+k*std.
    # ------------------------------------------------------------------
    threshold = float(np.percentile(lengths, 90))
    print(f"[{_ts()}] graph: edge threshold = {threshold:.1f} µm (p90)")

    # ------------------------------------------------------------------
    # 6. Build sparse adjacency, removing long edges
    # ------------------------------------------------------------------
    keep = lengths <= threshold
    kept_edges = edges[keep]
    n_kept = len(kept_edges)
    print(f"[{_ts()}] graph: {n_kept:,} / {len(edges):,} edges retained")

    if n_kept == 0:
        print(f"[{_ts()}] graph: no edges survive pruning — returning empty")
        return []

    # Symmetric adjacency
    rows = np.concatenate([kept_edges[:, 0], kept_edges[:, 1]])
    cols = np.concatenate([kept_edges[:, 1], kept_edges[:, 0]])
    data = np.ones(len(rows), dtype=np.float32)
    adj = sp.csr_matrix((data, (rows, cols)), shape=(n_cells, n_cells))

    # ------------------------------------------------------------------
    # 7. Connected components
    # ------------------------------------------------------------------
    n_components, comp_labels = connected_components(adj, directed=False)
    print(f"[{_ts()}] graph: {n_components:,} connected components")

    # ------------------------------------------------------------------
    # 8-9. Build hull for each component with >= 10 cells
    # ------------------------------------------------------------------
    min_cells = 10
    min_polygon_area_um2 = 5000.0
    polygons: list[Polygon] = []

    for comp_id in range(n_components):
        member_mask = comp_labels == comp_id
        n_members = member_mask.sum()
        if n_members < min_cells:
            continue

        pts = coords[member_mask]
        hull = _concave_hull(pts, buffer_um=15.0)
        if hull is None or hull.is_empty:
            continue
        if not hull.is_valid:
            hull = hull.buffer(0)
        if hull.area >= min_polygon_area_um2:
            polygons.append(hull)

    print(
        f"[{_ts()}] graph: {len(polygons)} polygons after area filter "
        f"(>= {min_polygon_area_um2} µm²)"
    )

    # ------------------------------------------------------------------
    # 10. Save diagnostic PNG
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Left: cell scatter coloured by component
    # Use only components that made it into polygons for colour coding,
    # show the rest in light grey
    polygon_comp_ids: set[int] = set()
    for comp_id in range(n_components):
        member_mask = comp_labels == comp_id
        if member_mask.sum() >= min_cells:
            polygon_comp_ids.add(comp_id)

    bg = ~np.isin(comp_labels, list(polygon_comp_ids))
    axes[0].scatter(
        coords[bg, 0],
        coords[bg, 1],
        s=0.2,
        c="lightgrey",
        alpha=0.3,
        rasterized=True,
    )

    fg = np.isin(comp_labels, list(polygon_comp_ids))
    if fg.any():
        axes[0].scatter(
            coords[fg, 0],
            coords[fg, 1],
            s=0.3,
            c=comp_labels[fg],
            cmap="nipy_spectral",
            alpha=0.5,
            rasterized=True,
        )
    axes[0].set_title(f"Cell components ({len(polygon_comp_ids)} large)")
    axes[0].set_xlabel("x (µm)")
    axes[0].set_ylabel("y (µm)")
    axes[0].set_aspect("equal")
    axes[0].invert_yaxis()

    # Right: polygon outlines
    axes[1].scatter(
        coords[:, 0],
        coords[:, 1],
        s=0.1,
        c="grey",
        alpha=0.2,
        rasterized=True,
    )
    for poly in polygons:
        xs, ys = poly.exterior.xy
        axes[1].plot(xs, ys, linewidth=0.8)
    axes[1].set_title(f"Villi polygons ({len(polygons)})")
    axes[1].set_xlabel("x (µm)")
    axes[1].set_ylabel("y (µm)")
    axes[1].set_aspect("equal")
    axes[1].invert_yaxis()

    fig.tight_layout()
    png_path = out / "graph_intermediate.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}] graph: saved diagnostic → {png_path}")

    return polygons

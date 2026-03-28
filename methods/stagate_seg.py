"""Villi segmentation via STAGATE / BANKSY-style spatial domain identification.

Constructs an AnnData object from the Xenium cell-feature matrix and spatial
coordinates, learns spatially-aware cell embeddings (STAGATE if installed,
otherwise a BANKSY-style spatially-smoothed-PCA fallback), clusters with
Leiden, and extracts villi as concave-hull polygons.  All output polygons are
in micron coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from shapely import MultiPoint, concave_hull

from utils.io import load_cells, load_expression_h5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _dense(X) -> np.ndarray:
    """Return a dense numpy array regardless of input type."""
    if hasattr(X, "toarray"):
        return X.toarray()
    return np.asarray(X)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi using spatial-domain clustering (STAGATE / BANKSY).

    Parameters
    ----------
    data_path : str
        Root directory containing ``cells.parquet`` and
        ``cell_feature_matrix/`` (or ``.h5``).
    output_dir : str
        Directory to write diagnostic PNGs.

    Returns
    -------
    list[Polygon]
        Villi polygons in micron coordinates.
    """
    import scanpy as sc
    import anndata as ad

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load cell centroids and gene expression
    # ------------------------------------------------------------------
    print(f"[{_ts()}] stagate_seg: loading cells …")
    cells_df = load_cells(data_path)
    print(f"[{_ts()}] stagate_seg: {len(cells_df):,} cells loaded")

    print(f"[{_ts()}] stagate_seg: loading expression matrix …")
    expr_mat, gene_names, cell_ids = load_expression_h5(data_path)
    print(
        f"[{_ts()}] stagate_seg: expression matrix {expr_mat.shape[0]:,} cells "
        f"× {expr_mat.shape[1]:,} genes"
    )

    # ------------------------------------------------------------------
    # 2. Build AnnData object
    # ------------------------------------------------------------------
    print(f"[{_ts()}] stagate_seg: building AnnData …")
    adata = ad.AnnData(X=expr_mat)
    adata.obs_names = [str(c) for c in cell_ids]
    adata.var_names = [str(g) for g in gene_names]

    # Align cell order: expression matrix cell_ids ↔ cells_df rows
    # Build a lookup from cell_id → (x, y) from the cells DataFrame.
    cells_df = (
        cells_df.set_index("cell_id") if "cell_id" in cells_df.columns else cells_df
    )
    spatial_coords = np.column_stack(
        [
            cells_df.loc[adata.obs_names, "x_centroid"].values,
            cells_df.loc[adata.obs_names, "y_centroid"].values,
        ]
    ).astype(np.float32)
    adata.obsm["spatial"] = spatial_coords
    print(
        f"[{_ts()}] stagate_seg: spatial range "
        f"x=[{spatial_coords[:, 0].min():.0f}, {spatial_coords[:, 0].max():.0f}] "
        f"y=[{spatial_coords[:, 1].min():.0f}, {spatial_coords[:, 1].max():.0f}] µm"
    )

    # ------------------------------------------------------------------
    # 3. Preprocessing (scanpy)
    # ------------------------------------------------------------------
    print(f"[{_ts()}] stagate_seg: preprocessing …")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_hvg = min(50, adata.n_vars)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg)
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"[{_ts()}] stagate_seg: {adata.n_vars} HVGs selected")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(20, adata.n_vars - 1))
    print(
        f"[{_ts()}] stagate_seg: PCA done ({adata.obsm['X_pca'].shape[1]} components)"
    )

    # ------------------------------------------------------------------
    # 4. Spatial neighbor graph (for Leiden on spatial coords)
    # ------------------------------------------------------------------
    print(f"[{_ts()}] stagate_seg: building spatial neighbor graph (k=15) …")
    sc.pp.neighbors(
        adata,
        n_neighbors=15,
        use_rep="spatial",
        key_added="spatial_neighbors",
    )

    # ------------------------------------------------------------------
    # 5. Embedding: try STAGATE, fall back to BANKSY-style
    # ------------------------------------------------------------------
    cluster_key: str | None = None

    # -- 5a. STAGATE path --------------------------------------------------
    try:
        import torch
        import STAGATE_pyG as STAGATE  # type: ignore[import-untyped]

        print(f"[{_ts()}] stagate_seg: STAGATE found — training …")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        STAGATE.Cal_Spatial_Net(adata, rad_cutoff=50)  # 50 µm radius
        adata = STAGATE.train_STAGATE(
            adata,
            alpha=0,
            random_seed=0,
            device=device,
        )
        sc.pp.neighbors(adata, use_rep="STAGATE")
        sc.tl.leiden(adata, resolution=0.5, key_added="stagate_cluster")
        cluster_key = "stagate_cluster"
        print(
            f"[{_ts()}] stagate_seg: STAGATE clustering done — "
            f"{adata.obs[cluster_key].nunique()} clusters (device={device})"
        )
    except ImportError:
        print(
            f"[{_ts()}] stagate_seg: STAGATE not available — "
            "falling back to BANKSY-style approach"
        )

    # -- 5b. BANKSY-style fallback -----------------------------------------
    if cluster_key is None:
        from sklearn.neighbors import NearestNeighbors
        from sklearn.decomposition import PCA

        print(f"[{_ts()}] stagate_seg: computing spatially-smoothed expression …")
        coords = adata.obsm["spatial"]
        nn = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(coords)
        distances, indices = nn.kneighbors(coords)

        X_dense = _dense(adata.X)
        n_cells = len(adata)

        # Spatial smoothing: inverse-distance–weighted average of neighbours
        X_smooth = np.zeros_like(X_dense)
        # Vectorised batch to avoid per-cell Python loop on 157 k cells
        weights = 1.0 / (distances + 1e-6)  # (N, k)
        weights /= weights.sum(axis=1, keepdims=True)

        # Build smoothed matrix in chunks to limit memory
        chunk = 10_000
        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            idx = indices[start:end]  # (chunk, k)
            w = weights[start:end]  # (chunk, k)
            # Gather neighbour rows: (chunk, k, G)
            neighbour_expr = X_dense[idx]
            X_smooth[start:end] = np.einsum("ij,ijk->ik", w, neighbour_expr)

        print(f"[{_ts()}] stagate_seg: running PCA on original + smoothed …")
        n_comps = min(20, X_dense.shape[1] - 1)
        pca_orig = PCA(n_components=n_comps).fit_transform(X_dense)
        pca_smooth = PCA(n_components=n_comps).fit_transform(X_smooth)
        adata.obsm["banksy_rep"] = np.hstack([pca_orig, pca_smooth]).astype(np.float32)
        print(
            f"[{_ts()}] stagate_seg: BANKSY representation shape "
            f"{adata.obsm['banksy_rep'].shape}"
        )

        del X_dense, X_smooth  # free memory

        sc.pp.neighbors(adata, use_rep="banksy_rep", n_neighbors=30)
        sc.tl.leiden(adata, resolution=0.3, key_added="banksy_cluster")
        cluster_key = "banksy_cluster"
        print(
            f"[{_ts()}] stagate_seg: BANKSY clustering done — "
            f"{adata.obs[cluster_key].nunique()} clusters"
        )

    # ------------------------------------------------------------------
    # 6. Convert clusters → concave-hull polygons
    # ------------------------------------------------------------------
    print(f"[{_ts()}] stagate_seg: extracting polygons from clusters …")
    cluster_labels = adata.obs[cluster_key].values
    xy = adata.obsm["spatial"]

    min_cells = 20
    min_area_um2 = 5000.0
    buffer_um = 5.0
    simplify_tol = 2.0
    hull_ratio = 0.3

    polygons: list[Polygon] = []
    cluster_polygon_map: dict[str, list[Polygon]] = {}

    for label in np.unique(cluster_labels):
        mask = cluster_labels == label
        pts = xy[mask]
        if len(pts) < min_cells:
            continue

        mp = MultiPoint(pts)
        hull = concave_hull(mp, ratio=hull_ratio)
        if hull.is_empty:
            continue

        # Buffer then simplify for smoother boundaries
        hull = hull.buffer(buffer_um).simplify(simplify_tol)
        if hull.is_empty:
            continue

        # Hull may be Polygon or MultiPolygon after buffering
        parts: list[Polygon] = []
        if isinstance(hull, MultiPolygon):
            parts = [p for p in hull.geoms if p.area >= min_area_um2]
        elif isinstance(hull, Polygon) and hull.area >= min_area_um2:
            parts = [hull]

        if parts:
            polygons.extend(parts)
            cluster_polygon_map[str(label)] = parts

    print(
        f"[{_ts()}] stagate_seg: {len(polygons)} polygons after area filter "
        f"(>= {min_area_um2:.0f} µm²)"
    )

    # ------------------------------------------------------------------
    # 7. Diagnostic plot
    # ------------------------------------------------------------------
    _save_diagnostic(adata, cluster_key, xy, polygons, out)

    return polygons


# ---------------------------------------------------------------------------
# diagnostic visualisation
# ---------------------------------------------------------------------------


def _save_diagnostic(
    adata,
    cluster_key: str,
    xy: np.ndarray,
    polygons: list[Polygon],
    out: Path,
) -> None:
    """Scatter plot of cells coloured by cluster with polygon outlines."""
    labels = adata.obs[cluster_key].values
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    cmap = plt.cm.get_cmap("tab20", max(n_clusters, 1))
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    fig, ax = plt.subplots(figsize=(14, 14))

    # Cell scatter (subsample if > 50 k for speed)
    n = len(xy)
    if n > 50_000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=50_000, replace=False)
    else:
        idx = np.arange(n)

    colours = np.array([label_to_idx[l] for l in labels[idx]])
    ax.scatter(
        xy[idx, 0],
        xy[idx, 1],
        c=colours,
        cmap=cmap,
        s=0.3,
        alpha=0.4,
        rasterized=True,
    )

    # Polygon outlines
    for poly in polygons:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="black", linewidth=1.0)

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(
        f"Spatial clustering ({cluster_key}) — "
        f"{n_clusters} clusters, {len(polygons)} polygons"
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()

    fig.tight_layout()
    png_path = out / "stagate_seg_diagnostic.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}] stagate_seg: saved diagnostic → {png_path}")

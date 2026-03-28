"""Multimodal GNN for semi-supervised villi segmentation.

Combines DAPI morphology features with gene expression via a GATv2-based
graph neural network. Cells are nodes; KNN spatial graph provides edges.
Semi-supervised: trains on GT-annotated cells, propagates to all ~157K cells.
All output polygons are in micron coordinates.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from utils.coords import micron_to_pixel, PIXEL_SIZE_UM
from utils.io import load_cells, load_expression_h5, load_morphology, load_annotations


def _ts() -> str:
    return time.strftime("%H:%M:%S")


# ======================================================================
# Phase A: Feature extraction
# ======================================================================


def _normalize_expression(mat) -> np.ndarray:
    """Total-count normalize per cell, then log1p.

    Parameters
    ----------
    mat : scipy.sparse matrix, shape (n_cells, n_genes)

    Returns
    -------
    np.ndarray, shape (n_cells, n_genes), float32
    """
    import scipy.sparse as sp

    # Total counts per cell
    totals = np.asarray(mat.sum(axis=1)).ravel().astype(np.float64)
    totals[totals == 0] = 1.0  # avoid div-by-zero for empty cells

    # Normalize to median total count (standard scanpy-style)
    median_total = np.median(totals[totals > 1.0]) if (totals > 1.0).any() else 1.0

    if sp.issparse(mat):
        mat = mat.toarray()
    mat = mat.astype(np.float64)
    mat = (mat / totals[:, None]) * median_total
    return np.log1p(mat).astype(np.float32)


def _pca_features(expr: np.ndarray, n_components: int = 20) -> np.ndarray:
    """PCA on normalized expression matrix.

    Returns
    -------
    np.ndarray, shape (n_cells, n_components), float32
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(expr).astype(np.float32)


def _extract_image_features(
    dapi: np.ndarray, centroids_um: np.ndarray, patch_half: int = 16
) -> np.ndarray:
    """Extract per-cell DAPI patch statistics.

    For each cell centroid (in microns), convert to pixel coordinates,
    extract a 32x32 patch, and compute 8 summary statistics.

    Parameters
    ----------
    dapi : np.ndarray
        2-D uint16 DAPI max-projection, shape (H, W).
    centroids_um : np.ndarray
        Cell centroids in micron coords, shape (n_cells, 2) with columns [x, y].
    patch_half : int
        Half-size of patch in pixels (default 16 → 32x32 patches).

    Returns
    -------
    np.ndarray, shape (n_cells, 8), float32
    """
    H, W = dapi.shape
    dapi_f = dapi.astype(np.float32)
    n_cells = len(centroids_um)
    features = np.zeros((n_cells, 8), dtype=np.float32)

    # Convert micron centroids to pixel coordinates
    centroids_px = centroids_um / PIXEL_SIZE_UM  # [x_px, y_px]

    for i in range(n_cells):
        cx_px = int(round(centroids_px[i, 0]))
        cy_px = int(round(centroids_px[i, 1]))

        # Compute patch bounds with boundary handling
        r_start = cy_px - patch_half
        r_end = cy_px + patch_half
        c_start = cx_px - patch_half
        c_end = cx_px + patch_half

        # Clamp to image bounds
        r_start_clamped = max(0, r_start)
        r_end_clamped = min(H, r_end)
        c_start_clamped = max(0, c_start)
        c_end_clamped = min(W, c_end)

        if r_start_clamped >= r_end_clamped or c_start_clamped >= c_end_clamped:
            # Cell completely outside image — leave zeros
            continue

        patch = dapi_f[r_start_clamped:r_end_clamped, c_start_clamped:c_end_clamped]

        if patch.size == 0:
            continue

        # If patch was clipped at boundary, pad with zeros to full size
        if patch.shape != (patch_half * 2, patch_half * 2):
            full_patch = np.zeros((patch_half * 2, patch_half * 2), dtype=np.float32)
            # Where to place the valid data in the full patch
            pr_start = r_start_clamped - r_start
            pc_start = c_start_clamped - c_start
            full_patch[
                pr_start : pr_start + patch.shape[0],
                pc_start : pc_start + patch.shape[1],
            ] = patch
            patch = full_patch

        # Compute gradient magnitude
        gy, gx = np.gradient(patch)
        grad_mag = np.sqrt(gx**2 + gy**2)

        features[i, 0] = patch.mean()
        features[i, 1] = patch.std()
        features[i, 2] = np.median(patch)
        features[i, 3] = np.percentile(patch, 25)
        features[i, 4] = np.percentile(patch, 75)
        features[i, 5] = patch.max()
        features[i, 6] = grad_mag.mean()
        features[i, 7] = grad_mag.std()

    return features


# ======================================================================
# Phase B: Graph construction
# ======================================================================


def _build_knn_graph(coords: np.ndarray, k: int = 10):
    """Build KNN graph from spatial coordinates.

    Parameters
    ----------
    coords : np.ndarray, shape (n_cells, 2)
        Spatial coordinates (microns).
    k : int
        Number of nearest neighbors.

    Returns
    -------
    edge_index : np.ndarray, shape (2, n_edges), int64
        COO format edge index (symmetric / undirected).
    """
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", n_jobs=-1)
    nn.fit(coords)
    distances, indices = nn.kneighbors(coords)

    n = len(coords)
    src = np.repeat(np.arange(n), k)
    dst = indices[:, 1:].ravel()  # skip self (column 0)

    # Make undirected by adding both directions, then deduplicate
    edge_src = np.concatenate([src, dst])
    edge_dst = np.concatenate([dst, src])

    # Deduplicate
    edges = np.stack([edge_src, edge_dst], axis=0)
    edges_sorted = np.sort(edges, axis=0)
    _, unique_idx = np.unique(edges_sorted[0] * n + edges_sorted[1], return_index=True)
    edge_index = edges[:, unique_idx]

    return edge_index.astype(np.int64)


# ======================================================================
# Phase C: Label preparation
# ======================================================================


def _assign_labels(
    centroids_um: np.ndarray,
    gt_polygons: list[Polygon],
    far_threshold_um: float = 100.0,
) -> np.ndarray:
    """Assign semi-supervised labels based on GT polygons.

    Returns
    -------
    labels : np.ndarray, shape (n_cells,), int
        1 = inside GT polygon (villus), 0 = far from any GT polygon (background),
        -1 = near but not inside (unlabeled, excluded from training).
    """
    from shapely.geometry import Point
    from shapely.ops import unary_union
    from shapely import prepared

    n_cells = len(centroids_um)
    labels = np.full(n_cells, -1, dtype=np.int32)

    if not gt_polygons:
        print(f"[{_ts()}]   WARNING: no GT polygons — all cells unlabeled")
        return labels

    # Merge GT polys for distance computation
    gt_union = unary_union(gt_polygons)
    gt_prep = prepared.prep(gt_union)

    # Buffered region: cells within far_threshold of any GT polygon boundary
    gt_buffered = gt_union.buffer(far_threshold_um)

    n_pos = 0
    n_neg = 0
    for i in range(n_cells):
        pt = Point(centroids_um[i, 0], centroids_um[i, 1])
        if gt_prep.contains(pt):
            labels[i] = 1
            n_pos += 1
        elif not gt_buffered.contains(pt):
            # Far from GT polygons → negative
            labels[i] = 0
            n_neg += 1
        # else: near but not inside → stays -1

    print(
        f"[{_ts()}]   labels: {n_pos:,} positive, {n_neg:,} negative, "
        f"{n_cells - n_pos - n_neg:,} unlabeled"
    )
    return labels


# ======================================================================
# Phase D: GNN model & training
# ======================================================================


def _build_and_train(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    n_epochs: int = 200,
    lr: float = 1e-3,
    hidden: int = 64,
):
    """Build GATv2 model, train semi-supervised, return predictions.

    Returns
    -------
    probs : np.ndarray, shape (n_cells,)
        Predicted villus probability for every cell.
    loss_history : list[float]
        Training loss per epoch.
    """
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import GATv2Conv
    from torch_geometric.data import Data

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{_ts()}]   device: {device}")

    # ----- Model -----
    class VillusGNN(torch.nn.Module):
        def __init__(self, in_channels: int, hidden: int = 64):
            super().__init__()
            self.conv1 = GATv2Conv(in_channels, hidden, heads=4, concat=True)
            self.conv2 = GATv2Conv(hidden * 4, hidden, heads=4, concat=True)
            self.conv3 = GATv2Conv(hidden * 4, hidden, heads=1, concat=False)
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(hidden, 32),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(32, 1),
            )

        def forward(self, x, edge_index):
            x = F.elu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.elu(self.conv2(x, edge_index))
            x = F.dropout(x, p=0.3, training=self.training)
            x = self.conv3(x, edge_index)
            return self.classifier(x).squeeze(-1)

    # ----- Data -----
    x = torch.from_numpy(node_features).float()
    ei = torch.from_numpy(edge_index).long()
    y = torch.from_numpy(labels).long()

    data = Data(x=x, edge_index=ei, y=y)
    data = data.to(device)

    # Training mask: only labeled cells (label != -1)
    train_mask = data.y >= 0
    y_train = data.y[train_mask].float()

    # Class imbalance weight
    n_pos = (y_train == 1).sum().item()
    n_neg = (y_train == 0).sum().item()
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], device=device)
    else:
        pos_weight = torch.tensor([1.0], device=device)
    print(
        f"[{_ts()}]   pos_weight: {pos_weight.item():.2f} "
        f"({n_pos:,} pos / {n_neg:,} neg)"
    )

    # ----- Model init -----
    model = VillusGNN(in_channels=node_features.shape[1], hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[{_ts()}]   model parameters: {total_params:,}")

    # ----- Training loop -----
    loss_history = []
    model.train()
    for epoch in range(1, n_epochs + 1):
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[train_mask], y_train)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if epoch % 20 == 0 or epoch == 1:
            # Evaluate on labeled cells
            model.eval()
            with torch.no_grad():
                logits_eval = model(data.x, data.edge_index)
                probs_eval = torch.sigmoid(logits_eval[train_mask])
                preds_eval = (probs_eval > 0.5).long()
                acc = (preds_eval == y_train.long()).float().mean().item()

                # AUC
                try:
                    from sklearn.metrics import roc_auc_score

                    auc = roc_auc_score(y_train.cpu().numpy(), probs_eval.cpu().numpy())
                except Exception:
                    auc = float("nan")

            print(
                f"[{_ts()}]   epoch {epoch:>3d}/{n_epochs} | "
                f"loss={loss.item():.4f} | acc={acc:.4f} | AUC={auc:.4f}"
            )
            model.train()

    # ----- Inference on all cells -----
    model.eval()
    with torch.no_grad():
        logits_all = model(data.x, data.edge_index)
        probs_all = torch.sigmoid(logits_all).cpu().numpy()

    return probs_all, loss_history


# ======================================================================
# Phase E: Polygon extraction
# ======================================================================


def _extract_polygons(
    centroids_um: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    eps_um: float = 30.0,
    min_samples: int = 10,
    hull_ratio: float = 0.3,
    buffer_um: float = 5.0,
    simplify_um: float = 2.0,
    min_area_um2: float = 5000.0,
) -> list[Polygon]:
    """Cluster positive cells with DBSCAN and extract concave hulls.

    Returns
    -------
    list[Polygon]
        Villi polygons in micron coordinates.
    """
    from sklearn.cluster import DBSCAN
    from shapely.geometry import MultiPoint
    from shapely import concave_hull

    pos_mask = probs >= threshold
    n_pos = pos_mask.sum()
    print(f"[{_ts()}]   positive cells: {n_pos:,} / {len(probs):,}")

    if n_pos < min_samples:
        print(f"[{_ts()}]   too few positive cells for clustering")
        return []

    pos_coords = centroids_um[pos_mask]

    # DBSCAN spatial clustering
    db = DBSCAN(eps=eps_um, min_samples=min_samples, n_jobs=-1)
    cluster_labels = db.fit_predict(pos_coords)
    n_clusters = cluster_labels.max() + 1
    print(f"[{_ts()}]   DBSCAN found {n_clusters} clusters (eps={eps_um} µm)")

    polygons: list[Polygon] = []
    for cid in range(n_clusters):
        cluster_mask = cluster_labels == cid
        cluster_coords = pos_coords[cluster_mask]

        if len(cluster_coords) < 4:
            continue

        # Concave hull
        mp = MultiPoint(cluster_coords.tolist())
        try:
            hull = concave_hull(mp, ratio=hull_ratio)
        except Exception:
            hull = mp.convex_hull

        if hull.is_empty or hull.geom_type == "Point" or hull.geom_type == "LineString":
            continue

        # Buffer and simplify
        poly = hull.buffer(buffer_um).simplify(simplify_um)

        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue

        if poly.area >= min_area_um2:
            polygons.append(poly)

    print(
        f"[{_ts()}]   {len(polygons)} polygons after area filter "
        f"(>= {min_area_um2:.0f} µm²)"
    )
    return polygons


# ======================================================================
# Phase F: Diagnostics
# ======================================================================


def _save_diagnostics(
    output_dir: Path,
    centroids_um: np.ndarray,
    probs: np.ndarray,
    gt_polygons: list[Polygon],
    polygons: list[Polygon],
    loss_history: list[float],
) -> None:
    """Save scatter plot of predictions and training loss curve."""
    # --- Scatter plot: cells colored by predicted probability ---
    fig, ax = plt.subplots(figsize=(14, 12))
    sc = ax.scatter(
        centroids_um[:, 0],
        centroids_um[:, 1],
        c=probs,
        cmap="RdYlBu_r",
        s=0.3,
        alpha=0.6,
        vmin=0,
        vmax=1,
        rasterized=True,
    )
    plt.colorbar(sc, ax=ax, label="P(villus)", shrink=0.8)

    # Overlay GT polygon outlines
    for poly in gt_polygons:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="lime", linewidth=1.5, linestyle="--", label="GT")

    # Overlay predicted polygon outlines
    for poly in polygons:
        xs, ys = poly.exterior.xy
        ax.plot(xs, ys, color="cyan", linewidth=1.0, label="Pred")

    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_title(f"Multimodal GNN predictions ({len(polygons)} villi detected)")
    ax.set_aspect("equal")
    ax.invert_yaxis()

    # Deduplicate legend entries
    handles, lbl = ax.get_legend_handles_labels()
    by_label = dict(zip(lbl, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="upper right")

    fig.tight_layout()
    scatter_path = output_dir / "gnn_predictions.png"
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}]   saved predictions scatter → {scatter_path}")

    # --- Training loss curve ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(loss_history) + 1), loss_history, linewidth=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title("GNN Training Loss")
    ax.grid(True, alpha=0.3)

    loss_path = output_dir / "gnn_training_loss.png"
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{_ts()}]   saved loss curve → {loss_path}")


# ======================================================================
# Main entry point
# ======================================================================


def segment(data_path: str, output_dir: str) -> list[Polygon]:
    """Segment villi using a multimodal GNN (DAPI + gene expression).

    Parameters
    ----------
    data_path : str
        Root directory containing Xenium outputs (cells.parquet,
        cell_feature_matrix/, morphology.ome.tif, annotations/).
    output_dir : str
        Directory to write diagnostic PNGs.

    Returns
    -------
    list[Polygon]
        Villi polygons in micron coordinates.
    """
    # Guard: PyTorch Geometric is required
    try:
        import torch
        import torch_geometric  # noqa: F401
    except ImportError as e:
        print(
            f"[{_ts()}] multimodal_gnn: ERROR — PyTorch Geometric not available: {e}\n"
            "  Install with: pip install torch-geometric\n"
            "  Returning empty list."
        )
        return []

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Phase A: Feature extraction
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase A: Feature extraction =====")

    # A1. Load cells
    print(f"[{_ts()}] loading cells ...")
    cells = load_cells(data_path)
    centroids_um = cells[["x_centroid", "y_centroid"]].values.astype(np.float64)
    n_cells = len(cells)
    print(f"[{_ts()}]   {n_cells:,} cells loaded")

    # A2. Gene expression → normalize → PCA
    print(f"[{_ts()}] loading gene expression ...")
    expr_mat, gene_names, cell_ids = load_expression_h5(data_path)
    print(
        f"[{_ts()}]   expression matrix: {expr_mat.shape[0]:,} cells × "
        f"{expr_mat.shape[1]} genes"
    )

    print(f"[{_ts()}] normalizing expression (total-count → log1p) ...")
    expr_norm = _normalize_expression(expr_mat)

    print(f"[{_ts()}] PCA → 20 components ...")
    gene_pca = _pca_features(expr_norm, n_components=20)
    print(f"[{_ts()}]   gene_pca shape: {gene_pca.shape}")

    # A3. DAPI image features
    print(f"[{_ts()}] loading morphology (max-project) ...")
    dapi = load_morphology(data_path, max_project=True)
    print(f"[{_ts()}]   DAPI shape: {dapi.shape}, dtype: {dapi.dtype}")

    print(f"[{_ts()}] extracting per-cell image features (32×32 patches) ...")
    img_features = _extract_image_features(dapi, centroids_um)
    print(f"[{_ts()}]   image features shape: {img_features.shape}")

    # Free DAPI image memory
    del dapi

    # A4. Concatenate features
    node_features = np.hstack([gene_pca, img_features])
    print(f"[{_ts()}]   node_features: {node_features.shape} (20 gene + 8 image)")

    # Standardize features (zero-mean, unit-variance)
    feat_mean = node_features.mean(axis=0, keepdims=True)
    feat_std = node_features.std(axis=0, keepdims=True)
    feat_std[feat_std < 1e-8] = 1.0  # avoid div-by-zero
    node_features = ((node_features - feat_mean) / feat_std).astype(np.float32)

    # ==================================================================
    # Phase B: Graph construction
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase B: Graph construction =====")
    print(f"[{_ts()}] building KNN graph (k=10) ...")
    edge_index = _build_knn_graph(centroids_um, k=10)
    print(
        f"[{_ts()}]   edges: {edge_index.shape[1]:,} "
        f"(~{edge_index.shape[1] / n_cells:.1f} per node)"
    )

    # ==================================================================
    # Phase C: Label preparation
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase C: Label preparation =====")
    print(f"[{_ts()}] loading GT annotations ...")
    gt_polygons = load_annotations(data_path)
    print(f"[{_ts()}]   {len(gt_polygons)} GT polygons loaded")

    print(f"[{_ts()}] assigning semi-supervised labels ...")
    labels = _assign_labels(centroids_um, gt_polygons, far_threshold_um=100.0)

    n_labeled = (labels >= 0).sum()
    if n_labeled == 0:
        print(
            f"[{_ts()}] ERROR: no labeled cells found — cannot train. "
            "Check GT annotations."
        )
        return []

    # ==================================================================
    # Phase D: GNN training
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase D: GNN training (semi-supervised) =====")
    probs, loss_history = _build_and_train(
        node_features=node_features,
        edge_index=edge_index,
        labels=labels,
        n_epochs=200,
        lr=1e-3,
        hidden=64,
    )

    # ==================================================================
    # Phase E: Polygon extraction
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase E: Inference & polygon extraction =====")
    polygons = _extract_polygons(
        centroids_um=centroids_um,
        probs=probs,
        threshold=0.5,
        eps_um=30.0,
        min_samples=10,
        hull_ratio=0.3,
        buffer_um=5.0,
        simplify_um=2.0,
        min_area_um2=5000.0,
    )

    # ==================================================================
    # Phase F: Diagnostics
    # ==================================================================
    print(f"\n[{_ts()}] ===== Phase F: Diagnostics =====")
    _save_diagnostics(
        output_dir=out,
        centroids_um=centroids_um,
        probs=probs,
        gt_polygons=gt_polygons,
        polygons=polygons,
        loss_history=loss_history,
    )

    print(f"\n[{_ts()}] multimodal_gnn: DONE — {len(polygons)} villi polygons returned")
    return polygons

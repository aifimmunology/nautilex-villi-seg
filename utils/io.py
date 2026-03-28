"""Data loading helpers for Xenium outputs + annotations."""

from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import shape, Polygon, MultiPolygon

from utils.coords import pixel_to_micron

# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------


def load_transcripts(data_path: str | Path, min_qv: float = 20.0) -> pd.DataFrame:
    """Load transcripts.parquet, filter by quality value."""
    df = pd.read_parquet(
        Path(data_path) / "transcripts.parquet",
        columns=["x_location", "y_location", "feature_name", "qv"],
    )
    if min_qv > 0:
        df = df[df["qv"] >= min_qv].reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def load_cells(data_path: str | Path) -> pd.DataFrame:
    """Load cells.parquet with centroid x/y."""
    return pd.read_parquet(Path(data_path) / "cells.parquet")


# ---------------------------------------------------------------------------
# Gene expression matrix  (for GNN / STAGATE)
# ---------------------------------------------------------------------------


def load_expression_h5(data_path: str | Path):
    """Load cell-feature matrix from the HDF5 directory export.
    Returns (scipy.sparse.csc_matrix, gene_names, cell_ids).
    """
    import scipy.sparse as sp
    import h5py

    cfm_dir = Path(data_path) / "cell_feature_matrix"
    if not cfm_dir.is_dir():
        # Fall back to the single .h5 file
        cfm_dir = Path(data_path) / "cell_feature_matrix.h5"

    if cfm_dir.is_dir():
        # Xenium directory export: barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz
        from scipy.io import mmread
        import gzip

        mat = mmread(str(cfm_dir / "matrix.mtx.gz")).T.tocsc()  # cells × genes
        with gzip.open(cfm_dir / "features.tsv.gz", "rt") as f:
            genes = [line.strip().split("\t")[1] for line in f]
        with gzip.open(cfm_dir / "barcodes.tsv.gz", "rt") as f:
            cell_ids = [line.strip() for line in f]
        return mat, genes, cell_ids
    else:
        # Single HDF5 file
        h5_path = Path(data_path) / "cell_feature_matrix.h5"
        with h5py.File(h5_path, "r") as f:
            grp = f["matrix"]
            data = grp["data"][:]
            indices = grp["indices"][:]
            indptr = grp["indptr"][:]
            shape_val = grp["shape"][:]
            genes = [x.decode() for x in grp["features/name"][:]]
            cell_ids = [x.decode() for x in grp["barcodes"][:]]
        mat = sp.csc_matrix((data, indices, indptr), shape=shape_val).T
        return mat, genes, cell_ids


# ---------------------------------------------------------------------------
# Morphology image
# ---------------------------------------------------------------------------


def load_morphology(data_path: str | Path, max_project: bool = True) -> np.ndarray:
    """Load morphology.ome.tif and optionally max-project Z-stack.
    Returns uint16 2-D array (Y, X).
    """
    import tifffile

    tif_path = Path(data_path) / "morphology.ome.tif"
    img = tifffile.imread(str(tif_path))  # shape: (Z, Y, X) or (Y, X)

    if img.ndim == 3 and max_project:
        img = img.max(axis=0)
    elif img.ndim > 3:
        # (C, Z, Y, X) or similar — take first channel, max-project
        img = (
            img[0].max(axis=0)
            if img.ndim == 4
            else img.reshape(-1, *img.shape[-2:]).max(axis=0)
        )
    return img


# ---------------------------------------------------------------------------
# Ground-truth annotations  (GeoJSON → list of shapely Polygons in microns)
# ---------------------------------------------------------------------------


def load_annotations(data_path: str | Path) -> list[Polygon]:
    """Load GeoJSON annotations and convert from pixel → micron coordinates.
    Returns list of shapely Polygons in micron space.
    """
    ann_dir = Path(data_path) / "annotations"
    if not ann_dir.exists():
        return []

    polys: list[Polygon] = []
    for gj_file in sorted(ann_dir.glob("*.geojson")):
        with open(gj_file) as f:
            fc = json.load(f)
        features = fc.get("features", [fc]) if "features" in fc else [fc]
        for feat in features:
            geom = shape(feat["geometry"]) if "geometry" in feat else shape(feat)
            if isinstance(geom, MultiPolygon):
                polys.extend(geom.geoms)
            elif isinstance(geom, Polygon):
                polys.append(geom)

    # Convert pixel → micron
    return [pixel_to_micron(p) for p in polys]

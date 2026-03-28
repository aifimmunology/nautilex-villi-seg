"""Validation: match predicted polygons to GT and compute metrics."""

from __future__ import annotations
import numpy as np
from shapely.geometry import Polygon


def iou(a: Polygon, b: Polygon) -> float:
    """Intersection-over-union between two shapely Polygons."""
    if not a.is_valid:
        a = a.buffer(0)
    if not b.is_valid:
        b = b.buffer(0)
    inter = a.intersection(b).area
    union = a.union(b).area
    return inter / union if union > 0 else 0.0


def match_polygons(
    preds: list[Polygon],
    gts: list[Polygon],
    iou_thresh: float = 0.1,
) -> list[dict]:
    """Greedy matching: for each GT polygon, find best-matching prediction.
    Returns list of dicts with gt_idx, pred_idx, iou.
    """
    if not preds or not gts:
        return []

    # Build IoU matrix
    iou_mat = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            # Quick bounding-box pre-filter
            if not g.bounds_intersect(p) if hasattr(g, "bounds_intersect") else True:
                pass
            iou_mat[i, j] = iou(g, p)

    matches = []
    used_preds = set()
    for _ in range(min(len(gts), len(preds))):
        if iou_mat.max() < iou_thresh:
            break
        gi, pj = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
        matches.append(
            {"gt_idx": int(gi), "pred_idx": int(pj), "iou": float(iou_mat[gi, pj])}
        )
        iou_mat[gi, :] = -1
        iou_mat[:, pj] = -1
        used_preds.add(int(pj))

    return matches


def evaluate(
    preds: list[Polygon],
    gts: list[Polygon],
    iou_thresholds: list[float] | None = None,
) -> dict:
    """Compute full evaluation metrics.
    Returns dict with mean_iou, per-threshold precision/recall/f1,
    and per-villus IoU scores.
    """
    if iou_thresholds is None:
        iou_thresholds = [0.25, 0.5, 0.75]

    matches = match_polygons(preds, gts, iou_thresh=0.01)
    per_villus = {m["gt_idx"]: m["iou"] for m in matches}
    mean_iou = np.mean([m["iou"] for m in matches]) if matches else 0.0

    results: dict = {
        "n_gt": len(gts),
        "n_pred": len(preds),
        "n_matched": len(matches),
        "mean_iou": float(mean_iou),
        "per_villus_iou": {str(k): float(v) for k, v in per_villus.items()},
    }

    for thresh in iou_thresholds:
        tp = sum(1 for m in matches if m["iou"] >= thresh)
        prec = tp / len(preds) if preds else 0.0
        rec = tp / len(gts) if gts else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        results[f"P@{thresh}"] = float(prec)
        results[f"R@{thresh}"] = float(rec)
        results[f"F1@{thresh}"] = float(f1)

    return results

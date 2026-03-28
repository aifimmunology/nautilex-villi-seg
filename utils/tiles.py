"""Tile utilities for processing large images in patches."""

from __future__ import annotations
import numpy as np


def extract_tiles(
    image: np.ndarray,
    tile_size: int = 2048,
    overlap: int = 256,
) -> list[tuple[np.ndarray, tuple[int, int, int, int]]]:
    """Extract overlapping tiles from a 2-D image.
    Returns list of (tile, (y0, x0, y1, x1)) in pixel coords.
    """
    H, W = image.shape[:2]
    stride = tile_size - overlap
    tiles = []
    for y0 in range(0, H, stride):
        for x0 in range(0, W, stride):
            y1 = min(y0 + tile_size, H)
            x1 = min(x0 + tile_size, W)
            tile = image[y0:y1, x0:x1]
            # Pad if smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
                padded[: tile.shape[0], : tile.shape[1]] = tile
                tile = padded
            tiles.append((tile, (y0, x0, y1, x1)))
    return tiles


def stitch_masks(
    tiles: list[tuple[np.ndarray, tuple[int, int, int, int]]],
    image_shape: tuple[int, int],
    mode: str = "max",
) -> np.ndarray:
    """Stitch tile masks back into a full image.
    mode='max' takes per-pixel maximum (good for binary/probability masks).
    mode='label' uses center-crop priority to avoid overlap artifacts.
    """
    H, W = image_shape
    if mode == "max":
        out = np.zeros((H, W), dtype=np.float32)
        for mask, (y0, x0, y1, x1) in tiles:
            h, w = y1 - y0, x1 - x0
            out[y0:y1, x0:x1] = np.maximum(
                out[y0:y1, x0:x1], mask[:h, :w].astype(np.float32)
            )
        return out
    else:  # label — center-crop priority
        out = np.zeros((H, W), dtype=np.int32)
        for mask, (y0, x0, y1, x1) in tiles:
            h, w = y1 - y0, x1 - x0
            region = mask[:h, :w]
            empty = out[y0:y1, x0:x1] == 0
            out[y0:y1, x0:x1] = np.where(empty, region, out[y0:y1, x0:x1])
        return out

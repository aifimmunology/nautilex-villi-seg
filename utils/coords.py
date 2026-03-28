"""Coordinate transforms between pixel space and micron space."""

PIXEL_SIZE_UM = 0.2125  # µm per pixel (Xenium default)


def pixel_to_micron(coords, pixel_size=PIXEL_SIZE_UM):
    """Convert (N,2) array or shapely geometry from pixel → micron coords."""
    import numpy as np
    from shapely import affinity

    if hasattr(coords, "geom_type"):  # shapely geometry
        return affinity.scale(coords, xfact=pixel_size, yfact=pixel_size, origin=(0, 0))
    return np.asarray(coords, dtype=np.float64) * pixel_size


def micron_to_pixel(coords, pixel_size=PIXEL_SIZE_UM):
    """Convert (N,2) array or shapely geometry from micron → pixel coords."""
    import numpy as np
    from shapely import affinity

    if hasattr(coords, "geom_type"):
        return affinity.scale(
            coords, xfact=1 / pixel_size, yfact=1 / pixel_size, origin=(0, 0)
        )
    return np.asarray(coords, dtype=np.float64) / pixel_size

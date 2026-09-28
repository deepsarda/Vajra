"""Draft warning polygons from a grid of warning levels, ready for a forecaster to edit."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon

from vajra.grid import Grid


def draft_polygons(levels: np.ndarray, grid: Grid, min_area_km2: float = 50.0, simplify_km: float = 2.0):
    """One polygon per connected area at or above each level, largest level first.

    Returns a list of dicts: {"level": 1..3, "polygon": shapely Polygon in lon/lat, "area_km2": float}.
    """
    import cv2

    tol_deg = simplify_km / 111.0
    out = []
    higher = None
    for level in (3, 2, 1):
        mask = (np.asarray(levels) >= level).astype(np.uint8)
        if not mask.any() or (higher is not None and np.array_equal(mask, higher)):
            higher = mask
            continue
        higher = mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area_km2 = float(cv2.contourArea(c)) * grid.km**2
            if area_km2 < min_area_km2 or len(c) < 3:
                continue
            cols, rows = c[:, 0, 0], c[:, 0, 1]
            lon, lat = grid.lonlat(cols, rows)
            poly = Polygon(np.column_stack([lon, lat])).buffer(0).simplify(tol_deg, preserve_topology=True)
            if poly.is_empty or poly.geom_type != "Polygon":
                continue
            out.append({"level": level, "polygon": poly, "area_km2": area_km2})
    return out

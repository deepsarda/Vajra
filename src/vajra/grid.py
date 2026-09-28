"""A regular latitude-longitude grid and distance-based neighbourhoods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

KM_PER_DEG_LAT = 110.57


@dataclass(frozen=True)
class Grid:
    """Square pixels of ``km`` kilometres. Row 0 is the southern edge, column 0 the western edge.

    ``lon0`` and ``lat0`` give the centre of the pixel at row 0, column 0.
    """

    nx: int
    ny: int
    km: float = 2.0
    lon0: float = 81.5
    lat0: float = 24.5

    @property
    def dlat(self) -> float:
        return self.km / KM_PER_DEG_LAT

    @property
    def dlon(self) -> float:
        lat_mid = self.lat0 + self.dlat * (self.ny - 1) / 2
        return float(self.km / (111.32 * np.cos(np.radians(lat_mid))))

    def lonlat(self, col, row) -> tuple[np.ndarray, np.ndarray]:
        """Longitude and latitude of (fractional) pixel coordinates."""
        col = np.asarray(col, float)
        row = np.asarray(row, float)
        return self.lon0 + col * self.dlon, self.lat0 + row * self.dlat


def disk(radius_px: float) -> np.ndarray:
    """Boolean disk footprint with the given radius in pixels."""
    r = int(np.ceil(radius_px))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return x * x + y * y <= radius_px * radius_px


def within(mask: np.ndarray, radius_km: float, km_per_px: float) -> np.ndarray:
    """True wherever a True pixel of ``mask`` lies within ``radius_km``.

    Works on arrays shaped (..., ny, nx); leading axes are treated independently.
    This turns "lightning here" into "lightning within 10 km", the target Vajra predicts.
    """
    mask = np.asarray(mask, bool)
    radius_px = radius_km / km_per_px
    if radius_px < 0.5:
        return mask.copy()
    fp = disk(radius_px)
    footprint = fp.reshape((1,) * (mask.ndim - 2) + fp.shape)
    grown = ndimage.maximum_filter(mask.astype(np.uint8), footprint=footprint, mode="constant", cval=0)
    return grown.astype(bool)

"""Baselines that the Vajra model must beat. Each returns a (n_steps, ny, nx) chance of lightning."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from vajra.grid import within
from vajra.nowcast.optical_flow import advect, estimate_motion


def persistence(field: np.ndarray, n_steps: int) -> np.ndarray:
    """Tomorrow looks like today: repeat the latest field for every lead time."""
    return np.repeat(np.asarray(field, float)[None], n_steps, axis=0)


def lightning_only(flash_frames: np.ndarray, n_steps: int, radius_px: float) -> np.ndarray:
    """Lightning-only tracking: move the area of recent lightning along with its storm.

    It uses nothing but lightning, so a storm that has not flashed yet stays invisible.
    That blind spot is the gap Vajra is built to close.
    """
    recent = np.asarray(flash_frames[-2:], float)
    density = ndimage.gaussian_filter(recent, sigma=(0, max(radius_px / 2, 1), max(radius_px / 2, 1)))
    area = within(recent[-1] > 0, radius_px, 1.0).astype(float)
    if not area.any():
        return np.zeros((n_steps, *area.shape))
    return np.clip(advect(area, estimate_motion(density), n_steps), 0, 1)


def radar_extrapolation(
    dbz_frames: np.ndarray,
    n_steps: int,
    radius_px: float,
    threshold_dbz: float = 40.0,
    width_db: float = 4.0,
) -> np.ndarray:
    """Move radar echoes with optical flow, then read strong echoes as a chance of lightning nearby.

    Strong echoes appear before the first strike, so this sees new storms that
    lightning-only tracking misses. It cannot foresee storms growing or dying.
    """
    frames = np.asarray(dbz_frames, float)
    moved = advect(frames[-1], estimate_motion(frames), n_steps)
    chance = 1 / (1 + np.exp(-(moved - threshold_dbz) / width_db))
    size = (1, int(2 * round(radius_px) + 1), int(2 * round(radius_px) + 1))
    return ndimage.maximum_filter(chance, size=size, mode="constant")

"""Estimate how storms move, and carry fields forward along that motion.

Motion fields follow the pySTEPS convention: shape (2, ny, nx), where [0] is movement along
columns and [1] along rows, in pixels per time step.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _to_u8(frames: np.ndarray) -> np.ndarray:
    lo, hi = float(np.nanmin(frames)), float(np.nanmax(frames))
    scale = 255.0 / (hi - lo) if hi > lo else 0.0
    return np.nan_to_num((frames - lo) * scale).astype(np.uint8)


def _spread(flow: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Spread motion from where storms are into clear air (normalised convolution).

    Optical flow only sees motion where there is something to track. Without this step,
    storms would stop dead at the edge of their current footprint.
    """
    total = float(weight.sum())
    if total <= 0:
        return np.zeros_like(flow)
    sigma = max(flow.shape[1:]) / 8
    w = ndimage.gaussian_filter(weight, sigma)
    out = []
    for comp in flow:
        mean = float((comp * weight).sum() / total)
        num = ndimage.gaussian_filter(comp * weight, sigma)
        out.append(np.where(w > 1e-6 * w.max(), num / np.maximum(w, 1e-12), mean))
    return np.stack(out)


def _farneback(prev: np.ndarray, curr: np.ndarray) -> np.ndarray:
    import cv2

    a, b = _to_u8(np.stack([prev, curr]))
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 31, 5, 7, 1.5, 0)
    flow = np.stack([flow[..., 0], flow[..., 1]]).astype(np.float64)
    return _spread(flow, np.maximum(a, b).astype(float) / 255.0)


def estimate_motion(frames: np.ndarray, method: str = "auto") -> np.ndarray:
    """Motion field from a (time, ny, nx) stack, newest frame last.

    ``method``: "pysteps" (Lucas-Kanade from pySTEPS), "farneback" (OpenCV), or "auto",
    which uses pySTEPS when it is installed and OpenCV otherwise.
    """
    frames = np.asarray(frames, float)
    if frames.ndim != 3 or frames.shape[0] < 2:
        raise ValueError("need a (time, ny, nx) stack with at least two frames")
    if method in ("auto", "pysteps"):
        try:
            from pysteps import motion
        except ImportError:
            if method == "pysteps":
                raise
        else:
            return np.asarray(motion.get_method("lucaskanade")(frames[-3:]), float)
    if method not in ("auto", "farneback"):
        raise ValueError(f"unknown method {method!r}")
    return _farneback(frames[-2], frames[-1])


def advect(field: np.ndarray, motion: np.ndarray, n_steps: int, fill: float = 0.0) -> np.ndarray:
    """Carry ``field`` forward ``n_steps`` steps along ``motion`` (semi-Lagrangian, backward trajectories).

    Returns a (n_steps, ny, nx) stack; entry k is the forecast k + 1 steps ahead.
    """
    field = np.asarray(field, float)
    u, v = np.asarray(motion, float)
    rows, cols = np.mgrid[0 : field.shape[0], 0 : field.shape[1]].astype(float)
    out = np.empty((n_steps, *field.shape))
    for k in range(n_steps):
        du = ndimage.map_coordinates(u, [rows, cols], order=1, mode="nearest")
        dv = ndimage.map_coordinates(v, [rows, cols], order=1, mode="nearest")
        rows, cols = rows - dv, cols - du
        out[k] = ndimage.map_coordinates(field, [rows, cols], order=1, mode="constant", cval=fill)
    return out


def extrapolate(frames: np.ndarray, n_steps: int, method: str = "auto") -> np.ndarray:
    """Move the newest frame forward along the motion of the recent frames."""
    return advect(np.asarray(frames)[-1], estimate_motion(frames, method), n_steps)

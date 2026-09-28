"""Minutes of warning before the first lightning.

This is Vajra's headline number: how much earlier than lightning-only tracking do people
get warned when a new storm produces its first strike?
"""

from __future__ import annotations

import numpy as np

from vajra.grid import within


def first_event_step(observed: np.ndarray) -> np.ndarray:
    """Index of the first True step for every pixel of a (time, ny, nx) stack; -1 where never True."""
    obs = np.asarray(observed, bool)
    return np.where(obs.any(axis=0), np.argmax(obs, axis=0), -1)


def initiation_mask(observed: np.ndarray, first: np.ndarray, radius_px: float, lookback: int) -> np.ndarray:
    """Pixels whose first lightning came from a new storm.

    A first strike counts as a new storm when no lightning was seen within ``radius_px``
    during the ``lookback`` steps before it. Storms that arrive already flashing are
    excluded, because lightning-only tracking can already see those.
    """
    obs = np.asarray(observed, bool)
    near = within(obs, radius_px, 1.0)
    ny, nx = first.shape
    counts = np.concatenate([np.zeros((1, ny, nx), int), np.cumsum(near, axis=0)], axis=0)
    has = first >= 0
    t = np.where(has, first, 0)
    t0 = np.maximum(0, t - lookback)
    yy, xx = np.indices((ny, nx))
    return has & (counts[t, yy, xx] - counts[t0, yy, xx] == 0)


def warning_lead_minutes(warned: np.ndarray, first: np.ndarray, step_minutes: float, horizon_steps: int):
    """Minutes between the earliest warning and the first lightning, per pixel.

    ``warned`` is a (time, ny, nx) stack: True where the forecast issued at that step expects
    lightning within the next ``horizon_steps``. Only warnings issued inside that horizon
    before the first strike count. Pixels never warned in time score 0; pixels that never
    see lightning are NaN.
    """
    w = np.asarray(warned, bool)
    n_t = w.shape[0]
    has = first >= 0
    t = np.where(has, first, 0)
    yy, xx = np.indices(first.shape)
    lead = np.where(has, 0.0, np.nan)
    assigned = np.zeros(first.shape, bool)
    for k in range(horizon_steps, 0, -1):  # earliest warning first
        s = t - k
        hit = has & ~assigned & (s >= 0) & w[np.clip(s, 0, n_t - 1), yy, xx]
        lead[hit] = k * step_minutes
        assigned |= hit
    return lead


def minutes_gained(
    lead_method: np.ndarray, lead_baseline: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """Average extra minutes of warning the method gives over the baseline."""
    ok = np.isfinite(lead_method) & np.isfinite(lead_baseline)
    if mask is not None:
        ok &= np.asarray(mask, bool)
    return float(np.mean(lead_method[ok] - lead_baseline[ok])) if ok.any() else float("nan")

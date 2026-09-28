"""Verification scores for yes/no and probability forecasts.

All functions take NumPy arrays of matching shape. Observed arrays are yes/no (bool or 0/1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class Contingency:
    """Counts of the four outcomes of a yes/no forecast."""

    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int

    @property
    def csi(self) -> float:
        """Critical success index: hits / (hits + misses + false alarms). 1 is perfect."""
        d = self.hits + self.misses + self.false_alarms
        return self.hits / d if d else float("nan")

    @property
    def pod(self) -> float:
        """Probability of detection: the share of observed events that were forecast."""
        d = self.hits + self.misses
        return self.hits / d if d else float("nan")

    @property
    def far(self) -> float:
        """False alarm ratio: the share of forecast events that did not happen."""
        d = self.hits + self.false_alarms
        return self.false_alarms / d if d else float("nan")

    @property
    def bias(self) -> float:
        """Frequency bias: forecast events divided by observed events. 1 is unbiased."""
        d = self.hits + self.misses
        return (self.hits + self.false_alarms) / d if d else float("nan")

    def __add__(self, other: Contingency) -> Contingency:
        return Contingency(
            self.hits + other.hits,
            self.misses + other.misses,
            self.false_alarms + other.false_alarms,
            self.correct_negatives + other.correct_negatives,
        )


def contingency(forecast, observed) -> Contingency:
    f = np.asarray(forecast, bool)
    o = np.asarray(observed, bool)
    if f.shape != o.shape:
        raise ValueError(f"shape mismatch: {f.shape} vs {o.shape}")
    return Contingency(int(np.sum(f & o)), int(np.sum(~f & o)), int(np.sum(f & ~o)), int(np.sum(~f & ~o)))


def critical_success_index(forecast, observed) -> float:
    return contingency(forecast, observed).csi


def fractions_skill_score(forecast, observed, window: int) -> float:
    """Fractions skill score (Roberts and Lean, 2008) over square neighbourhoods of ``window`` pixels.

    ``forecast`` may be yes/no or probabilities. 1 is perfect and 0 means no skill.
    For stacks shaped (..., ny, nx) the score is pooled over the leading axes.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number of pixels")
    f = np.asarray(forecast, float)
    o = np.asarray(observed, float)
    size = (1,) * (f.ndim - 2) + (window, window)
    pf = ndimage.uniform_filter(f, size=size, mode="constant")
    po = ndimage.uniform_filter(o, size=size, mode="constant")
    ref = np.mean(pf**2) + np.mean(po**2)
    return float(1 - np.mean((pf - po) ** 2) / ref) if ref > 0 else float("nan")


def brier_score(probability, observed) -> float:
    """Mean squared error of probabilities against what happened. 0 is perfect."""
    p = np.asarray(probability, float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(np.mean((p - np.asarray(observed, float)) ** 2))


def brier_skill_score(probability, observed, reference: float | None = None) -> float:
    """Improvement over always forecasting ``reference`` (default: the observed frequency).

    1 is perfect, 0 is no better than the reference, below 0 is worse.
    """
    o = np.asarray(observed, float)
    ref = float(o.mean()) if reference is None else reference
    bs_ref = float(np.mean((ref - o) ** 2))
    return 1 - brier_score(probability, observed) / bs_ref if bs_ref > 0 else float("nan")


def reliability(probability, observed, n_bins: int = 10):
    """Observed event frequency in each forecast-probability bin.

    Returns (bin_centres, observed_frequency, counts). In a reliable forecast,
    observed_frequency sits close to bin_centres: 30% forecasts verify 30% of the time.
    """
    p = np.asarray(probability, float).ravel()
    o = np.asarray(observed, float).ravel()
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sums = np.bincount(idx, weights=o, minlength=n_bins)
    freq = np.full(n_bins, np.nan)
    np.divide(sums, counts, out=freq, where=counts > 0)
    return (edges[:-1] + edges[1:]) / 2, freq, counts

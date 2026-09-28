"""Isotonic calibration: make a 30% forecast come true about 30% of the time."""

from __future__ import annotations

import numpy as np


def _pool_adjacent_violators(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    vals: list[float] = []
    wts: list[float] = []
    sizes: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        vals.append(float(yi))
        wts.append(float(wi))
        sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w_sum = wts[-2] + wts[-1]
            merged = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w_sum
            size = sizes[-2] + sizes[-1]
            del vals[-1], wts[-1], sizes[-1]
            vals[-1], wts[-1], sizes[-1] = merged, w_sum, size
    return np.repeat(vals, sizes)


class IsotonicCalibrator:
    """A monotonic map from raw model probability to observed frequency.

    Fitting bins the raw probabilities first, so it scales to billions of pixels.
    """

    def __init__(self, n_bins: int = 1000):
        self.n_bins = n_bins
        self.x_: np.ndarray | None = None
        self.y_: np.ndarray | None = None

    def fit(self, probability, observed) -> IsotonicCalibrator:
        p = np.clip(np.asarray(probability, float).ravel(), 0, 1)
        o = np.asarray(observed, float).ravel()
        idx = np.minimum((p * self.n_bins).astype(int), self.n_bins - 1)
        counts = np.bincount(idx, minlength=self.n_bins).astype(float)
        sums = np.bincount(idx, weights=o, minlength=self.n_bins)
        psum = np.bincount(idx, weights=p, minlength=self.n_bins)
        used = counts > 0
        centres = psum[used] / counts[used]
        self.x_ = centres
        self.y_ = _pool_adjacent_violators(sums[used] / counts[used], counts[used])
        return self

    def predict(self, probability) -> np.ndarray:
        if self.x_ is None or self.y_ is None:
            raise RuntimeError("fit the calibrator first")
        p = np.asarray(probability, float)
        return np.interp(p, self.x_, self.y_)

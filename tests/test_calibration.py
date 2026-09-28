import numpy as np

from vajra.model import IsotonicCalibrator
from vajra.verify import brier_score


def test_isotonic_calibration_fixes_an_overconfident_forecast():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0, 1, 100_000)
    observed = rng.uniform(size=true_p.size) < true_p
    raw = np.sqrt(true_p)  # says 70% when the truth is 49%
    cal = IsotonicCalibrator().fit(raw, observed)
    assert brier_score(cal.predict(raw), observed) < brier_score(raw, observed)
    assert np.all(np.diff(cal.y_) >= -1e-12)

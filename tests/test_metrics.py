import numpy as np
import pytest

from vajra.verify import (
    brier_score,
    brier_skill_score,
    contingency,
    critical_success_index,
    fractions_skill_score,
    reliability,
)


def test_perfect_forecast_scores():
    obs = np.zeros((20, 20), bool)
    obs[5:9, 5:9] = True
    assert critical_success_index(obs, obs) == 1.0
    assert fractions_skill_score(obs, obs, 5) == pytest.approx(1.0)
    assert brier_score(obs.astype(float), obs) == 0.0


def test_contingency_counts():
    c = contingency(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    assert (c.hits, c.misses, c.false_alarms, c.correct_negatives) == (1, 1, 1, 1)
    assert c.csi == pytest.approx(1 / 3)
    assert (c.pod, c.far, c.bias) == (0.5, 0.5, 1.0)


def test_fss_rewards_near_misses_at_larger_scales():
    obs = np.zeros((40, 40))
    obs[18:22, 18:22] = 1
    displaced = np.roll(obs, 4, axis=1)
    assert fractions_skill_score(displaced, obs, 1) == pytest.approx(0.0)
    assert fractions_skill_score(displaced, obs, 15) > 0.5


def test_brier_skill_and_reliability_of_a_calibrated_forecast():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    o = rng.uniform(0, 1, p.size) < p
    centres, freq, _ = reliability(p, o, 10)
    assert np.allclose(freq, centres, atol=0.01)
    assert brier_skill_score(p, o) == pytest.approx(1 / 3, abs=0.01)
    with pytest.raises(ValueError):
        brier_score(np.array([1.2]), np.array([1]))

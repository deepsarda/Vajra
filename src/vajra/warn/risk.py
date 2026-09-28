"""Risk = chance of lightning x who is likely to be outdoors.

Levels follow IMD's colour codes: yellow (be updated), orange (be prepared), red (take action).
"""

from __future__ import annotations

import numpy as np

LEVELS = ("none", "yellow", "orange", "red")


def outdoor_share(hour_ist: float, farmland_share: np.ndarray) -> np.ndarray:
    """Rough share of people outdoors: more on farmland, and far more in daylight working hours."""
    daylight = float(np.clip(1 - abs(hour_ist - 12.5) / 7.0, 0, 1))
    return np.clip(0.15 + 0.85 * daylight * np.asarray(farmland_share, float), 0, 1)


def exposure(population: np.ndarray, farmland_share: np.ndarray, hour_ist: float) -> np.ndarray:
    """0 to 1 score of how many people are exposed, from population density (people per km^2)."""
    pop = np.log1p(np.asarray(population, float)) / np.log1p(10_000)
    return np.clip(pop, 0, 1) * outdoor_share(hour_ist, farmland_share)


def risk_levels(chance: np.ndarray, exposed: np.ndarray, thresholds=(0.2, 0.4, 0.6)) -> np.ndarray:
    """Warning level per pixel: 0 none, 1 yellow, 2 orange, 3 red.

    A high chance still warns empty areas, just at a lower level than crowded fields.
    """
    r = np.asarray(chance, float) * (0.4 + 0.6 * np.asarray(exposed, float))
    return np.digitize(r, thresholds).astype(np.int8)

import numpy as np

from vajra.nowcast import advect, estimate_motion, lightning_only, persistence, radar_extrapolation


def blob(cx, cy, n=64, s=4.0):
    yy, xx = np.mgrid[0:n, 0:n]
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s * s))


def test_optical_flow_carries_a_moving_storm_forward():
    frames = np.stack([blob(20, 30), blob(23, 32)])  # 3 columns and 2 rows per step
    forecast = advect(frames[-1], estimate_motion(frames, method="farneback"), 3)
    row, col = np.unravel_index(np.argmax(forecast[-1]), forecast[-1].shape)
    assert abs(col - 32) <= 2 and abs(row - 38) <= 2


def test_lightning_only_is_blind_to_a_storm_that_has_not_flashed():
    dbz = np.stack([blob(20, 30) * 55, blob(23, 32) * 55])
    no_flashes = np.zeros((2, 64, 64))
    assert lightning_only(no_flashes, 6, radius_px=5).max() == 0
    assert radar_extrapolation(dbz, 6, radius_px=5).max() > 0.9
    assert persistence(dbz[-1], 4).shape == (4, 64, 64)

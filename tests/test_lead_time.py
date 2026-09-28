import numpy as np

from vajra.verify import first_event_step, initiation_mask, minutes_gained, warning_lead_minutes


def test_lead_time_and_minutes_gained():
    obs = np.zeros((10, 5, 5), bool)
    obs[6:, 2, 2] = True  # first lightning at step 6
    first = first_event_step(obs)
    assert first[2, 2] == 6 and first[0, 0] == -1
    method = np.zeros_like(obs)
    method[3:, 2, 2] = True  # warned from step 3
    baseline = np.zeros_like(obs)
    baseline[5:, 2, 2] = True  # warned from step 5
    lead_m = warning_lead_minutes(method, first, 10, horizon_steps=6)
    lead_b = warning_lead_minutes(baseline, first, 10, horizon_steps=6)
    assert lead_m[2, 2] == 30 and lead_b[2, 2] == 10 and np.isnan(lead_m[0, 0])
    assert minutes_gained(lead_m, lead_b) == 20


def test_initiation_excludes_storms_already_flashing_nearby():
    obs = np.zeros((6, 9, 9), bool)
    obs[1:, 4, 1] = True  # a storm flashing 3 pixels west from step 1
    obs[4:, 4, 4] = True  # first hit at step 4, next to that active storm
    obs[5, 0, 8] = True  # an isolated new storm at step 5
    new = initiation_mask(obs, first_event_step(obs), radius_px=4, lookback=3)
    assert not new[4, 4] and new[0, 8]

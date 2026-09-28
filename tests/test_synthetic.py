import numpy as np

from vajra.data.synthetic import generate


def test_synthetic_storms_are_reproducible():
    a, b = generate(seed=3), generate(seed=3)
    assert np.array_equal(a.flashes, b.flashes)
    assert a.reflectivity.shape == a.flashes.shape == a.cloud_top_k.shape == a.cape.shape


def test_strong_radar_echoes_come_before_the_first_lightning():
    s = generate(seed=0, minutes=300)
    first_echo = np.argmax((s.reflectivity >= 40).any(axis=(1, 2)))
    first_flash = np.argmax((s.flashes > 0).any(axis=(1, 2)))
    assert s.flashes.sum() > 0 and first_echo < first_flash

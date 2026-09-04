import numpy as np

from evolution.config import Config


def test_defaults_match_spec():
    c = Config()
    assert (c.width, c.height) == (128, 128)
    assert c.plant_cap == 20.0
    assert c.brain_cost == 0.02
    assert c.max_age == 2000


def test_config_is_frozen():
    c = Config()
    try:
        c.width = 5
    except Exception:
        return
    raise AssertionError("Config must be frozen so a run cannot be retuned mid-flight")


def test_rng_is_deterministic():
    a = Config(seed=7).rng()
    b = Config(seed=7).rng()
    assert np.array_equal(a.random(10), b.random(10))


def test_different_seeds_differ():
    a = Config(seed=1).rng()
    b = Config(seed=2).rng()
    assert not np.array_equal(a.random(10), b.random(10))

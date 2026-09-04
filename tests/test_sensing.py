import numpy as np

from evolution.config import Config
from evolution.sensing import DIRECTIONS, direction_of, sense_reference
from evolution.world import GridWorld


def make(**kw):
    cfg = Config(width=21, height=21, **kw)
    w = GridWorld(cfg, cfg.rng())
    w.plant[:] = 0.0
    return cfg, w


def test_vector_length_is_twenty():
    cfg, w = make()
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert v.shape == (20,)


def test_bias_is_one():
    cfg, w = make()
    assert sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[19] == 1.0


def test_direction_binning_ties_go_north_south():
    assert direction_of(1, -1) == DIRECTIONS.index("N")
    assert direction_of(-1, 1) == DIRECTIONS.index("S")
    assert direction_of(2, -1) == DIRECTIONS.index("E")
    assert direction_of(-2, 1) == DIRECTIONS.index("W")


def test_plant_to_the_east_lights_the_east_channel_only():
    cfg, w = make()
    w.plant[10, 13] = cfg.plant_cap
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    east = DIRECTIONS.index("E")
    assert v[east] > 0.0
    for d in range(4):
        if d != east:
            assert v[d] == 0.0


def test_plant_beyond_sense_range_is_invisible():
    cfg, w = make()
    w.plant[10, 15] = cfg.plant_cap
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert np.all(v[0:4] == 0.0)


def test_no_agent_in_range_gives_zero_distance_channel():
    cfg, w = make()
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert np.all(v[4:8] == 0.0)


def test_nearest_agent_reports_size_ratio_and_diet():
    cfg, w = make()
    w.place(1, 12, 10, size=2.0, diet=0.75)
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    east = DIRECTIONS.index("E")
    assert v[4 + east] > 0.0
    assert np.isclose(v[8 + east], 2.0)
    assert np.isclose(v[12 + east], 0.75)


def test_closer_agent_reports_larger_proximity():
    cfg, w = make()
    east = DIRECTIONS.index("E")
    w.place(1, 13, 10, 1.0, 0.0)
    far = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[4 + east]
    w.clear(13, 10)
    w.place(2, 11, 10, 1.0, 0.0)
    near = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[4 + east]
    assert near > far

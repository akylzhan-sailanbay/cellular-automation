import numpy as np

from evolution.config import Config
from evolution.world import GridWorld


def make(**kw):
    cfg = Config(width=16, height=16, **kw)
    return GridWorld(cfg, cfg.rng())


def test_wrapping_is_toroidal():
    w = make()
    assert w.wrap(-1, -1) == (15, 15)
    assert w.wrap(16, 16) == (0, 0)


def test_no_wrapping_clamps_when_disabled():
    cfg = Config(width=16, height=16, wrap=False)
    w = GridWorld(cfg, cfg.rng())
    assert w.wrap(-1, 5) == (0, 5)
    assert w.wrap(16, 5) == (15, 5)


def test_plants_grow_and_are_capped():
    w = make()
    for _ in range(10000):
        w.update()
    assert w.plant.max() <= Config().plant_cap + 1e-9
    assert w.plant.min() > 0.0


def test_plants_spread_producing_patchiness():
    """Uniform food makes random wandering optimal and gives evolution no
    gradient to climb (spec 4). Patchiness is the point."""
    w = make()
    w.plant[:] = 0.0
    w.plant[8, 8] = Config().plant_cap
    for _ in range(200):
        w.update()
    assert w.plant.std() > 0.05 * w.plant.mean()


def test_meat_decays_toward_zero():
    w = make()
    w.deposit_meat(3, 3, 100.0)
    for _ in range(500):
        w.update()
    assert w.meat[3, 3] < 1.0


def test_take_plant_never_returns_more_than_present():
    w = make()
    w.plant[2, 2] = 3.0
    assert w.take_plant(2, 2, 10.0) == 3.0
    assert w.plant[2, 2] == 0.0


def test_occupancy_place_move_clear():
    w = make()
    w.place(7, 4, 4, size=2.0, diet=0.25)
    assert not w.is_free(4, 4)
    assert w.occ_id[4, 4] == 7
    assert w.occ_size[4, 4] == 2.0
    assert w.move(7, 4, 4, 5, 4, size=2.0, diet=0.25)
    assert w.is_free(4, 4) and not w.is_free(5, 4)
    w.clear(5, 4)
    assert w.is_free(5, 4)


def test_move_into_occupied_cell_fails():
    w = make()
    w.place(1, 4, 4, 1.0, 0.0)
    w.place(2, 5, 4, 1.0, 0.0)
    assert not w.move(1, 4, 4, 5, 4, 1.0, 0.0)
    assert w.occ_id[4, 4] == 1


def test_growth_mask_confines_growth():
    cfg = Config(width=16, height=16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, 8:] = True
    w = GridWorld(cfg, cfg.rng(), growth_mask=mask)
    w.plant[:] = 0.0
    for _ in range(500):
        w.update()
    assert w.plant[:, :8].sum() == 0.0
    assert w.plant[:, 8:].sum() > 0.0

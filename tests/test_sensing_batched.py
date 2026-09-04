import numpy as np
import pytest

from evolution.config import Config
from evolution.sensing import sense_batch, sense_reference
from evolution.world import GridWorld


def populated_world(seed: int):
    cfg = Config(width=32, height=32, seed=seed)
    rng = cfg.rng()
    w = GridWorld(cfg, rng)
    for i in range(60):
        x, y = int(rng.integers(32)), int(rng.integers(32))
        if w.is_free(x, y):
            w.place(i, x, y, float(rng.uniform(0.5, 3.0)), float(rng.uniform(0, 1)))
    return cfg, w, rng


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_batched_matches_reference_exactly(seed):
    cfg, w, rng = populated_world(seed)
    ys, xs = np.nonzero(w.occ_id != -1)
    sizes = w.occ_size[ys, xs]
    ranges = rng.uniform(1.0, 6.0, size=len(xs))
    energies = rng.uniform(0.0, 200.0, size=len(xs))
    ages = rng.integers(0, 2000, size=len(xs))

    fast = sense_batch(w, xs, ys, sizes, ranges, energies, ages, cfg)
    for i in range(len(xs)):
        slow = sense_reference(
            w, int(xs[i]), int(ys[i]), float(sizes[i]), float(ranges[i]),
            float(energies[i]), int(ages[i]), cfg,
        )
        assert np.allclose(fast[i], slow, atol=1e-12), f"row {i} diverged"


def test_batch_handles_empty_population():
    cfg = Config(width=8, height=8)
    w = GridWorld(cfg, cfg.rng())
    empty = np.array([], dtype=np.int64)
    out = sense_batch(
        w, empty, empty, np.array([]), np.array([]), np.array([]), empty, cfg
    )
    assert out.shape == (0, 20)

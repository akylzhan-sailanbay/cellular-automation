import numpy as np

from evolution.brain import Brain
from evolution.config import Config
from evolution.genome import N_INPUTS, N_OUTPUTS, _add_node, random_genome


def settle(brain: Brain, inputs: np.ndarray, ticks: int = 50) -> np.ndarray:
    out = None
    for _ in range(ticks):
        out = brain.step(inputs)
    return out


def test_step_shapes():
    cfg = Config()
    b = Brain(random_genome(cfg.rng(), cfg))
    out = b.step(np.zeros(N_INPUTS))
    assert out.shape == (N_OUTPUTS,)


def test_outputs_are_bounded_by_tanh():
    cfg = Config()
    b = Brain(random_genome(cfg.rng(), cfg))
    out = settle(b, np.full(N_INPUTS, 1000.0))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_same_genome_same_inputs_same_outputs():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    assert np.array_equal(settle(Brain(g), x), settle(Brain(g), x))


def test_reset_clears_recurrent_state():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    b = Brain(g)
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    first = b.step(x)
    settle(b, x)
    b.reset()
    assert np.allclose(b.step(x), first)


def test_add_node_is_function_preserving_at_steady_state():
    """Spec 5.3. The synchronous update adds one tick of latency per inserted
    node, so pre- and post-split brains agree only after the signal settles.
    Comparing tick 1 to tick 1 would fail for a CORRECT implementation."""
    cfg = Config()
    rng = cfg.rng()
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    for _ in range(20):
        g = random_genome(rng, cfg)
        before = settle(Brain(g), x)
        h = g.copy()
        _add_node(h, rng)
        after = settle(Brain(h), x)
        assert np.allclose(before, after, atol=1e-9), (
            "add_node changed behaviour; structural mutations must arrive neutral"
        )


def test_disabled_connections_do_not_contribute():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    x = np.ones(N_INPUTS)
    live = settle(Brain(g), x)
    for c in g.conns:
        c.enabled = False
    dead = settle(Brain(g), x)
    assert np.allclose(dead, 0.0)
    assert not np.allclose(live, dead)

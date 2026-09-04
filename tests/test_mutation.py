import numpy as np

from evolution.config import Config
from evolution.genome import (
    BODY_GENE_RANGES,
    MUTATION_RATE_RANGE,
    mutate,
    random_genome,
    validate,
)


def test_mutate_does_not_modify_the_parent():
    cfg = Config()
    rng = cfg.rng()
    parent = random_genome(rng, cfg)
    before = [c.weight for c in parent.conns]
    mutate(parent, rng, cfg)
    assert [c.weight for c in parent.conns] == before


def test_mutation_disabled_produces_an_exact_clone():
    cfg = Config(mutation_enabled=False)
    rng = cfg.rng()
    parent = random_genome(rng, cfg)
    child = mutate(parent, rng, cfg)
    assert [c.weight for c in child.conns] == [c.weight for c in parent.conns]
    assert child.body == parent.body
    assert child.mutation_rate == parent.mutation_rate


def test_body_genes_stay_in_range_over_many_generations():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(2000):
        g = mutate(g, rng, cfg)
    for name, (lo, hi) in BODY_GENE_RANGES.items():
        assert lo <= g.body[name] <= hi
    lo, hi = MUTATION_RATE_RANGE
    assert lo <= g.mutation_rate <= hi


def test_genome_stays_valid_over_many_generations():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(2000):
        g = mutate(g, rng, cfg)
        validate(g)


def test_weights_actually_change():
    """Compare summary statistics, not element-wise: structural mutation can
    add and remove connections, so the two arrays need not be the same length
    and an element-wise comparison would raise instead of failing cleanly."""
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    before = np.abs([c.weight for c in g.conns]).mean()
    for _ in range(50):
        g = mutate(g, rng, cfg)
    after = np.abs([c.weight for c in g.conns]).mean()
    assert abs(after - before) > 1e-6

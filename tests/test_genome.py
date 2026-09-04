import pytest

from evolution.config import Config
from evolution.genome import (
    N_INPUTS,
    N_OUTPUTS,
    random_genome,
    validate,
)


def test_random_genome_has_fixed_io():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    assert sum(n.kind == "input" for n in g.nodes) == N_INPUTS
    assert sum(n.kind == "output" for n in g.nodes) == N_OUTPUTS
    assert g.hidden_count() == 0


def test_input_ids_are_zero_through_nineteen():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    assert {n.id for n in g.nodes if n.kind == "input"} == set(range(N_INPUTS))
    assert {n.id for n in g.nodes if n.kind == "output"} == set(
        range(N_INPUTS, N_INPUTS + N_OUTPUTS)
    )


def test_inputs_are_identity_outputs_are_tanh():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    for n in g.nodes:
        if n.kind == "input":
            assert n.activation == "identity"
        if n.kind == "output":
            assert n.activation == "tanh"


def test_random_genome_is_valid():
    cfg = Config()
    validate(random_genome(cfg.rng(), cfg))


def test_copy_is_deep():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    h = g.copy()
    h.conns[0].weight = 999.0
    h.body["diet"] = 0.123
    assert g.conns[0].weight != 999.0
    assert g.body["diet"] != 0.123


def test_validate_rejects_dangling_connection():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    g.conns[0].src = 9999
    with pytest.raises(AssertionError):
        validate(g)


def test_validate_rejects_connection_into_input():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    g.conns[0].dst = 0
    with pytest.raises(AssertionError):
        validate(g)

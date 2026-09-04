from evolution.config import Config
from evolution.genome import (
    _add_link,
    _add_node,
    _change_activation,
    _del_link,
    _del_node,
    random_genome,
    validate,
)


def test_add_node_increases_hidden_count_and_disables_the_split():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    before_enabled = g.enabled_count()
    _add_node(g, rng)
    validate(g)
    assert g.hidden_count() == 1
    # one connection disabled, two added: net +1 enabled
    assert g.enabled_count() == before_enabled + 1


def test_add_node_never_reuses_an_id_after_deletion():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    _add_node(g, rng)
    first_id = [n.id for n in g.nodes if n.kind == "hidden"][0]
    _del_node(g, rng)
    _add_node(g, rng)
    second_id = [n.id for n in g.nodes if n.kind == "hidden"][0]
    assert second_id != first_id
    validate(g)


def test_add_link_never_targets_an_input():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(200):
        _add_link(g, rng)
    validate(g)


def test_del_link_and_del_node_keep_the_genome_valid():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(20):
        _add_node(g, rng)
    for _ in range(50):
        _del_link(g, rng)
        _del_node(g, rng)
        validate(g)


def test_change_activation_never_touches_io_nodes():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(100):
        _change_activation(g, rng)
    validate(g)

from evolution.agent import Agent, metabolic_cost
from evolution.config import Config
from evolution.genome import random_genome


def make_agent(cfg):
    g = random_genome(cfg.rng(), cfg)
    return Agent.create(0, g, x=1, y=1, energy=100.0, depth=0)


def test_agent_caches_body_genes():
    cfg = Config()
    a = make_agent(cfg)
    assert a.size == a.genome.body["size"]
    assert a.diet == a.genome.body["diet"]
    assert a.brain_links == a.genome.enabled_count()


def test_metabolic_cost_includes_brain_rent():
    """Brain cost is what forces complexity to pay for itself (spec 7)."""
    cfg = Config()
    a = make_agent(cfg)
    base = metabolic_cost(a, cfg, moved=False)
    a.brain_links += 100
    # tolerance, not equality: adding 100 links inside the sum associates
    # differently from adding 100 * brain_cost to the result
    assert abs(
        metabolic_cost(a, cfg, moved=False) - (base + 100 * cfg.brain_cost)
    ) < 1e-9


def test_moving_costs_more_than_standing_still():
    cfg = Config()
    a = make_agent(cfg)
    assert metabolic_cost(a, cfg, moved=True) > metabolic_cost(a, cfg, moved=False)


def test_upkeep_scales_with_size_squared():
    cfg = Config(basal=0.0, brain_cost=0.0)
    a = make_agent(cfg)
    a.size = 2.0
    two = metabolic_cost(a, cfg, moved=False)
    a.size = 4.0
    assert abs(metabolic_cost(a, cfg, moved=False) - 4.0 * two) < 1e-9

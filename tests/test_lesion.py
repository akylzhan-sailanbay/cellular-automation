import numpy as np

from evolution.config import Config
from evolution.genome import _add_node
from evolution.lesion import fork, trajectory
from evolution.sim import Simulation


def evolved(seed=1, ticks=2500, attack=True):
    sim = Simulation(Config(width=48, height=48, seed=seed, allow_attack=attack))
    sim.run(ticks)
    return sim


def test_forks_are_faithful():
    """The experiment is worthless unless two unlesioned forks stay identical.
    If forking perturbs the RNG or the world, any difference we later attribute
    to the lesion could just be fork noise."""
    sim = evolved()
    assert sim.agents, "needs a live population"
    a = trajectory(fork(sim, False), 1200)
    b = trajectory(fork(sim, False), 1200)
    assert a["pops"] == b["pops"]
    assert a["final_pop"] == b["final_pop"]
    assert a["births"] == b["births"]


def test_fork_does_not_disturb_the_parent():
    sim = evolved()
    before = (len(sim.agents), sim.tick_count, sim.births)
    trajectory(fork(sim, True), 800)
    assert (len(sim.agents), sim.tick_count, sim.births) == before


def test_lesion_flag_reaches_every_agent_and_every_newborn():
    sim = evolved()
    twin = fork(sim, True)
    assert all(a.brain.lesioned for a in twin.agents)
    twin.run(1500)
    assert all(a.brain.lesioned for a in twin.agents), (
        "newborns must inherit the lesion or it washes out in one generation"
    )


def test_lesion_leaves_brain_cost_untouched():
    """Cost parity is the experiment's core control."""
    sim = evolved()
    before = sorted(a.brain_links for a in sim.agents)
    twin = fork(sim, True)
    assert sorted(a.brain_links for a in twin.agents) == before


def test_lesion_changes_behaviour_when_hidden_nodes_exist():
    """Sanity: if silencing genuinely removes computation, an agent that has
    hidden nodes must act differently. Otherwise the lesion does nothing and a
    null result would be meaningless."""
    cfg = Config()
    rng = cfg.rng()
    from evolution.brain import Brain
    from evolution.genome import random_genome
    g = random_genome(rng, cfg)
    for _ in range(5):
        _add_node(g, rng)
    for c in g.conns:
        c.weight *= 3.0                      # make hidden paths matter
    x = np.linspace(-1, 1, 20)
    plain, cut = Brain(g), Brain(g)
    cut.lesioned = True
    for _ in range(60):
        p = plain.step(x); c = cut.step(x)
    assert not np.allclose(p, c), "lesion had no behavioural effect at all"

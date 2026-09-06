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


def test_scramble_preserves_wiring_and_cost_exactly():
    """The whole point of scrambling over silencing: pathways and rent survive,
    only which weight sits on which edge changes. Silencing severed 64% of the
    sensor-to-motor wiring and killed the population outright."""
    from evolution.lesion import scramble
    sim = evolved(ticks=6000)
    twin = fork(sim, False)
    before_links = sorted(a.brain_links for a in twin.agents)
    before_weights = sorted(
        round(c.weight, 6) for a in twin.agents for c in a.genome.conns if c.enabled
    )
    scramble(twin, np.random.default_rng(0))
    assert sorted(a.brain_links for a in twin.agents) == before_links
    assert sorted(
        round(c.weight, 6) for a in twin.agents for c in a.genome.conns if c.enabled
    ) == before_weights, "scramble must permute weights, not change them"


def _give_hidden_structure(sim, rng):
    """Build the structure directly rather than hoping a short run evolves it.
    These tests previously skipped, and a skipped test verifies nothing."""
    from evolution.brain import Brain
    for a in sim.agents:
        for _ in range(4):
            _add_node(a.genome, rng)
        a.brain = Brain(a.genome)
        a.brain_links = a.genome.enabled_count()
    return sim


def test_scramble_actually_changes_behaviour():
    from evolution.lesion import scramble
    rng = np.random.default_rng(0)
    twin = _give_hidden_structure(fork(evolved(ticks=1200), False), rng)
    assert all(a.genome.hidden_count() == 4 for a in twin.agents)
    before = [[c.weight for c in a.genome.conns] for a in twin.agents]
    scramble(twin, np.random.default_rng(0))
    after = [[c.weight for c in a.genome.conns] for a in twin.agents]
    assert before != after, "scramble had no effect on any genome"


def test_scramble_is_inherited_through_the_genome():
    """Scrambling only the compiled brain would revert at the first birth,
    because children are built from the parent's genome."""
    from evolution.lesion import scramble
    rng = np.random.default_rng(0)
    twin = _give_hidden_structure(fork(evolved(ticks=1200), False), rng)
    tagged = twin.agents[0]
    before = [c.weight for c in tagged.genome.conns]
    scramble(twin, np.random.default_rng(1))
    assert [c.weight for c in tagged.genome.conns] != before


def test_scramble_keeps_every_pathway_connected():
    """The failure mode that broke the first experiment: a lesion that
    disconnects sensors from motors starves the population instead of
    testing it. Scrambling must leave the graph identical."""
    from evolution.lesion import scramble
    rng = np.random.default_rng(0)
    twin = _give_hidden_structure(fork(evolved(ticks=1200), False), rng)
    edges = sorted((c.src, c.dst) for a in twin.agents
                   for c in a.genome.conns if c.enabled)
    scramble(twin, np.random.default_rng(2))
    assert sorted((c.src, c.dst) for a in twin.agents
                  for c in a.genome.conns if c.enabled) == edges

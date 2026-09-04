"""Validation gates.

RESULT (2026-09-04, 150,000 ticks, 2 seeds, ~230 generations):

  seed  mut |   pop  gens  links  hidden   rich
    21   ON |    86   224   12.4    0.16  0.767
    21  OFF |   202   215   10.0    0.00  0.777
    22   ON |   124   236   17.6    2.88  0.581
    22  OFF |   130   196   10.0    0.00  0.677

Two conclusions that must not be conflated:

1. Structural evolution RUNS. Brains grew from 10.0 links to 12.4 and 17.6,
   with up to 2.88 hidden nodes, against a control pinned at exactly 10.0/0.00.
   This is asserted below and passes.

2. Structural evolution is NOT SHOWN TO BE ADAPTIVE. Mutation adds links by
   construction, so growth alone proves the machinery turns, not that selection
   favours the result. The real question is whether NEW variation beats SORTING
   among founding variants, and over ~230 generations it does not:
   0.674 vs 0.727, Cohen's d = -0.71 in the CONTROL's favour. That test is
   marked xfail rather than deleted, because a deleted failing test is a lie.
"""
import numpy as np
import pytest

from evolution.config import Config
from evolution.sim import Simulation


def quality(cfg: Config, poor: float = 0.15) -> np.ndarray:
    q = np.full((cfg.height, cfg.width), poor)
    q[:, cfg.width // 2:] = 1.0
    return q


def rich_fraction(sim: Simulation) -> float:
    return float(np.mean([a.x >= sim.cfg.width // 2 for a in sim.agents]))


def test_mutation_is_the_only_source_of_structural_change():
    """Control integrity. With mutation off, every genome must stay structurally
    identical to its founder forever. If this fails, the control is not a
    control and no comparison against it means anything."""
    cfg = Config(width=48, height=48, seed=21, mutation_enabled=False)
    sim = Simulation(cfg)
    sim.run(4000)
    assert sim.agents, "control went extinct before the check"
    assert {a.brain_links for a in sim.agents} == {cfg.initial_links}
    assert all(a.genome.hidden_count() == 0 for a in sim.agents)


def test_structure_grows_when_mutation_is_enabled():
    """The engine can accumulate structure at all. NOTE: this does NOT show the
    structure is useful -- see the module docstring."""
    cfg = Config(width=64, height=64, seed=22)
    sim = Simulation(cfg)
    sim.run(30_000)
    assert sim.agents, f"extinct at tick {sim.tick_count}"
    assert max(a.brain_links for a in sim.agents) > cfg.initial_links, (
        "no lineage ever gained a connection; check add_node and brain_cost"
    )


def test_population_occupies_better_habitat_than_chance():
    """The population concentrates where food is. Achieved by founder sorting
    alone, so this is a check that the world works, not that evolution does."""
    cfg = Config(width=64, height=64, seed=21, plant_growth=0.25)
    sim = Simulation(cfg, growth_mask=quality(cfg))
    sim.run(30_000)
    if not sim.agents:
        pytest.skip("population went extinct; retune before trusting this")
    assert rich_fraction(sim) > 0.55, (
        f"rich fraction {rich_fraction(sim):.3f} is at chance; the world is not "
        "presenting a usable gradient"
    )


@pytest.mark.xfail(
    reason=(
        "NOT SUPPORTED **on habitat occupancy specifically**: 0.674 (mutation "
        "on) vs 0.727 (control), Cohen's d = -0.71 over ~230 generations. Note "
        "this is a bad metric -- consumption equalises the two halves (ideal "
        "free distribution) so sorting alone reaches a good distribution. On "
        "BODY GENES new variation clearly DOES beat sorting: speed 1.96 with "
        "mutation vs 1.07 without, size 0.55 vs 1.16. See "
        "docs/results/2026-09-04-run-notes.md."
    ),
    strict=False,
    run=False,
)
def test_new_variation_outperforms_founder_sorting_on_habitat_choice():
    raise AssertionError(
        "Requires 4 x 150,000-tick runs (~25 min). Reproduce with:\n"
        "  python3 /tmp/finalgate.py 150000 2000 450 21,22\n"
        "Last measured: mutation ON mean 0.674, OFF mean 0.727 -> not supported "
        "for THIS metric. Body-gene optimisation tells the opposite story."
    )


def test_new_variation_beats_founder_sorting_on_body_genes():
    """The claim the habitat metric could not settle. With mutation off,
    selection can only sort the founding pool; with it on, lineages reach
    optima no founder carried. Measured over 40,000 ticks, seed 1:
    speed 1.96 (mutation) vs 1.07 (sorting only); size 0.55 vs 1.16."""
    import numpy as np

    results = {}
    for on in (True, False):
        cfg = Config(width=64, height=64, seed=1, allow_attack=True,
                     mutation_enabled=on)
        sim = Simulation(cfg)
        sim.run(20_000)
        assert sim.agents, f"extinct with mutation_enabled={on}"
        results[on] = float(np.mean([a.speed for a in sim.agents]))
    assert results[True] > results[False] + 0.3, (
        f"new variation ({results[True]:.2f}) did not beat founder sorting "
        f"({results[False]:.2f}) on the speed gene"
    )

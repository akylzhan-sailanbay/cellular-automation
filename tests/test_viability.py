from evolution.config import Config
from evolution.sim import Simulation


def test_population_survives_and_does_not_saturate():
    """The world must be able to carry life without carrying all possible life.
    Plant energy is the only brake in the system (spec 4); if this fails, the
    constants are wrong, not the code."""
    cfg = Config(width=64, height=64, seed=0)
    sim = Simulation(cfg)
    sim.run(20_000)
    cells = cfg.width * cfg.height
    assert len(sim.agents) > 0, f"extinct at tick {sim.tick_count}"
    assert len(sim.agents) < 0.5 * cells, "population saturated the grid"


def test_reproduction_actually_happens():
    sim = Simulation(Config(width=64, height=64, seed=0))
    sim.run(5_000)
    assert sim.births > 0, "no agent ever reproduced; check the energy economy"


def test_survives_across_several_seeds():
    for seed in (1, 2, 3):
        sim = Simulation(Config(width=64, height=64, seed=seed))
        sim.run(10_000)
        assert len(sim.agents) > 0, f"extinct on seed {seed}"

import numpy as np
import pytest

from evolution.config import Config
from evolution.sim import Simulation


def mean_x(sim: Simulation) -> float:
    return float(np.mean([a.x for a in sim.agents])) if sim.agents else float("nan")


def eastern_mask(cfg: Config) -> np.ndarray:
    mask = np.zeros((cfg.height, cfg.width), dtype=bool)
    mask[:, cfg.width // 2:] = True
    return mask


def test_null_run_shows_no_directional_drift():
    """Control. With mutation off, reproduction copies genomes exactly, so no
    lineage can get better at anything. If THIS run trends, the trend in the
    real run is an artefact of the harness and proves nothing."""
    cfg = Config(width=64, height=64, wrap=False, seed=21, mutation_enabled=False)
    sim = Simulation(cfg, growth_mask=eastern_mask(cfg))
    sim.run(10_000)
    if not sim.agents:
        pytest.skip("control population went extinct; retune before trusting the gate")
    assert mean_x(sim) < 0.60 * cfg.width, (
        f"control drifted east to {mean_x(sim):.1f} without mutation; "
        "something other than evolution is moving the population"
    )


def test_rigged_world_gate_population_evolves_eastward():
    """Gate. Plants grow only in the eastern half and the world does not wrap,
    so the optimal strategy is knowable in advance: go east. If the engine
    cannot solve this, it cannot solve anything (spec 11)."""
    cfg = Config(width=64, height=64, wrap=False, seed=21)
    sim = Simulation(cfg, growth_mask=eastern_mask(cfg))
    sim.run(10_000)
    assert sim.agents, f"population went extinct at tick {sim.tick_count}"
    assert mean_x(sim) > 0.65 * cfg.width, (
        f"mean x reached only {mean_x(sim):.1f} of {cfg.width}; "
        "the population did not learn to exploit the food gradient"
    )

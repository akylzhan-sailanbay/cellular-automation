import csv

import numpy as np

from evolution.config import Config
from evolution.sim import Simulation
from evolution.stats import StatsWriter, bimodality, collect


def test_bimodality_separates_unimodal_from_bimodal():
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0.5, 0.1, 5000)
    bimodal = np.concatenate(
        [rng.normal(0.1, 0.03, 2500), rng.normal(0.9, 0.03, 2500)]
    )
    assert bimodality(unimodal) < 0.555
    assert bimodality(bimodal) > 0.555


def test_collect_returns_the_documented_keys():
    sim = Simulation(Config(width=32, height=32, initial_agents=30))
    sim.run(20)
    row = collect(sim)
    for key in (
        "tick", "population", "births", "deaths_starved", "deaths_killed",
        "deaths_age", "mean_links", "max_links", "mean_hidden", "mean_diet",
        "std_diet", "diet_bimodality", "mean_size", "mean_sense_range",
        "mean_speed", "mean_mutation_rate", "diversity", "total_plant",
        "total_meat", "total_agent_energy", "mean_depth",
    ):
        assert key in row, f"missing metric {key}"


def test_collect_on_an_empty_population_does_not_crash():
    cfg = Config(width=16, height=16, initial_agents=4,
                 initial_energy=1.0, plant_growth=0.0)
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(100)
    assert collect(sim)["population"] == 0


def test_writer_round_trips(tmp_path):
    path = tmp_path / "stats.csv"
    w = StatsWriter(path)
    w.write({"tick": 1, "population": 5})
    w.write({"tick": 2, "population": 7})
    w.close()
    rows = list(csv.DictReader(path.open()))
    assert [r["population"] for r in rows] == ["5", "7"]

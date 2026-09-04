import csv
from pathlib import Path

import numpy as np

from evolution.sim import Simulation


def bimodality(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient. Above ~0.555 suggests two modes."""
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < 4:
        return 0.0
    sd = values.std()
    if sd < 1e-12:
        return 0.0
    z = (values - values.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    if kurt <= 0.0:
        return 0.0
    return (skew * skew + 1.0) / kurt


def _diversity(sim: Simulation, rng: np.random.Generator, k: int = 100) -> float:
    """Mean pairwise distance in body-gene space over a random sample."""
    n = len(sim.agents)
    if n < 2:
        return 0.0
    idx = rng.choice(n, size=min(k, n), replace=False)
    m = np.array(
        [
            [
                sim.agents[i].diet,
                sim.agents[i].size / 3.0,
                sim.agents[i].sense_range / 6.0,
                sim.agents[i].speed / 2.0,
            ]
            for i in idx
        ]
    )
    diffs = m[:, None, :] - m[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=2))
    iu = np.triu_indices(len(m), k=1)
    return float(d[iu].mean()) if len(iu[0]) else 0.0


def collect(sim: Simulation) -> dict[str, float]:
    agents = sim.agents
    n = len(agents)
    diet = np.array([a.diet for a in agents]) if n else np.zeros(0)
    links = np.array([a.brain_links for a in agents]) if n else np.zeros(0)
    return {
        "tick": sim.tick_count,
        "population": n,
        "births": sim.births,
        "deaths_starved": sim.deaths["starved"],
        "deaths_killed": sim.deaths["killed"],
        "deaths_age": sim.deaths["age"],
        "mean_links": float(links.mean()) if n else 0.0,
        "max_links": int(links.max()) if n else 0,
        "mean_hidden": (
            float(np.mean([a.genome.hidden_count() for a in agents])) if n else 0.0
        ),
        "mean_diet": float(diet.mean()) if n else 0.0,
        "std_diet": float(diet.std()) if n else 0.0,
        "diet_bimodality": bimodality(diet) if n else 0.0,
        "mean_size": float(np.mean([a.size for a in agents])) if n else 0.0,
        "mean_sense_range": (
            float(np.mean([a.sense_range for a in agents])) if n else 0.0
        ),
        "mean_speed": float(np.mean([a.speed for a in agents])) if n else 0.0,
        "mean_mutation_rate": (
            float(np.mean([a.genome.mutation_rate for a in agents])) if n else 0.0
        ),
        # seeded from tick_count, NOT sim.rng: drawing from the simulation's own
        # generator would let measurement change the trajectory being measured
        "diversity": _diversity(sim, np.random.default_rng(sim.tick_count)),
        "total_plant": sim.world.total_plant(),
        "total_meat": sim.world.total_meat(),
        "total_agent_energy": float(sum(a.energy for a in agents)),
        "mean_depth": float(np.mean([a.depth for a in agents])) if n else 0.0,
        "mean_x": float(np.mean([a.x for a in agents])) if n else 0.0,
    }


class StatsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

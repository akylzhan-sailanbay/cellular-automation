from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    # world
    width: int = 128
    height: int = 128
    wrap: bool = True

    # population seeding
    initial_agents: int = 200
    initial_energy: float = 150.0
    initial_links: int = 10
    initial_mutation_rate: float = 0.1

    # plant cellular automaton
    plant_cap: float = 20.0
    plant_growth: float = 0.15
    plant_spread: float = 0.05
    plant_seed_min: float = 1.0

    # feeding
    eat_rate: float = 5.0
    plant_energy: float = 1.0
    meat_energy: float = 1.5
    meat_per_size: float = 30.0
    meat_decay: float = 0.05

    # metabolism
    basal: float = 0.5
    move_cost: float = 0.2
    upkeep: float = 0.1
    brain_cost: float = 0.02
    attack_cost: float = 2.0

    # life cycle
    repro_threshold: float = 120.0
    max_age: int = 2000

    # staging switches
    allow_attack: bool = False
    mutation_enabled: bool = True

    # bookkeeping
    stats_interval: int = 50
    seed: int = 0

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

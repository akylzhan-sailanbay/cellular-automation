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
    # Growth is driven by seeded NEIGHBOURS, not by a flat rate everywhere --
    # a flat rate saturates the whole grid and destroys the patchiness the
    # plant layer exists to create (spec 4).
    plant_spread: float = 0.125
    plant_spontaneous: float = 0.01
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
    # 0.02 purged neutral structure ~7x faster than mutation supplied it
    # (0.06 hidden nodes where the rates alone predict 0.44). At 0.005 the
    # same run reaches 1.04. Complexity must be cheap enough to drift in
    # before it can ever pay off, since a split node arrives neutral.
    brain_cost: float = 0.005
    attack_cost: float = 2.0

    # life cycle
    # 120 boom-busted to extinction by tick 2251; 450 survives 3/3 seeds.
    # Higher is NOT better -- 900 loses a seed because bottlenecked
    # populations then reproduce too slowly to recover.
    repro_threshold: float = 450.0
    max_age: int = 2000

    # staging switches
    allow_attack: bool = False
    mutation_enabled: bool = True

    # bookkeeping
    stats_interval: int = 50
    seed: int = 0

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

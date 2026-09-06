"""Ablation experiment: does the evolved brain structure actually do anything?

The project's headline is that predation drives brain complexity. But mutation
adds connections by construction, so "brains got bigger" shows the machinery
turns, not that the structure earns its keep. This asks the harder question:
silence the hidden nodes and see whether the population still copes.

Two design decisions carry the whole experiment:

1. The lesion SILENCES rather than deletes. Hidden nodes stay in the genome and
   keep costing brain rent every tick. Deleting them would refund 13-20% of a
   reproduction budget, and lesioned agents might then do BETTER for reasons
   with nothing to do with computation.

2. Both arms are forked from one evolved population with a deep copy, so they
   share an identical world, identical agents, and an identical RNG state. Any
   divergence is caused by the lesion and nothing else. `forks_are_faithful`
   in the tests asserts two unlesioned forks stay byte-identical.
"""
import copy
from dataclasses import replace

import numpy as np

from evolution.config import Config
from evolution.sim import Simulation


def fork(sim: Simulation, lesion: bool) -> Simulation:
    """Deep-copy a simulation, optionally silencing every brain's hidden nodes."""
    twin = copy.deepcopy(sim)
    twin.lesion = lesion            # so newborns inherit it too
    if lesion:
        for a in twin.agents:
            a.brain.lesioned = True
    return twin


def trajectory(sim: Simulation, ticks: int, every: int = 100) -> dict:
    pops, energy = [], []
    births0 = sim.births
    deaths0 = dict(sim.deaths)
    for _ in range(ticks):
        sim.tick()
        if not sim.agents:
            break
        if sim.tick_count % every == 0:
            pops.append(len(sim.agents))
            energy.append(float(np.mean([a.energy for a in sim.agents])))
    return {
        "final_pop": len(sim.agents),
        "pops": pops,
        "mean_energy": float(np.mean(energy)) if energy else 0.0,
        "births": sim.births - births0,
        "starved": sim.deaths["starved"] - deaths0["starved"],
        "survived": bool(sim.agents),
        "ticks_lasted": sim.tick_count,
    }


def run_arm(seed: int, attack: bool, evolve: int, test: int) -> dict:
    """Evolve once, then run intact and lesioned twins from that same state."""
    cfg = Config(width=64, height=64, seed=seed, allow_attack=attack)
    sim = Simulation(cfg)
    sim.run(evolve)
    if not sim.agents:
        return {"extinct_before_test": True, "attack": attack, "seed": seed}

    hidden = float(np.mean([a.genome.hidden_count() for a in sim.agents]))
    links = float(np.mean([a.brain_links for a in sim.agents]))
    intact = trajectory(fork(sim, False), test)
    lesioned = trajectory(fork(sim, True), test)
    return {
        "extinct_before_test": False,
        "attack": attack, "seed": seed,
        "hidden_at_lesion": round(hidden, 3),
        "links_at_lesion": round(links, 2),
        "pop_at_lesion": len(sim.agents),
        "intact": intact, "lesioned": lesioned,
    }


def experiment(seeds=(1, 2, 3), evolve: int = 30_000, test: int = 6_000) -> list[dict]:
    """2x2: predation on/off crossed with intact/lesioned.

    The no-predation world evolves almost no hidden structure, so lesioning it
    should barely register. That is the built-in negative control: if the lesion
    hurts BOTH worlds equally, the effect is an artefact of the manipulation
    rather than of the structure being removed.
    """
    out = []
    for seed in seeds:
        for attack in (True, False):
            out.append(run_arm(seed, attack, evolve, test))
    return out

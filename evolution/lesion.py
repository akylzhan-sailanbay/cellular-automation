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
from evolution.brain import Brain
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


def experiment(seeds=(1, 2, 3), evolve: int = 30_000, test: int = 6_000,
               on_arm=None) -> list[dict]:
    """2x2: predation on/off crossed with intact/lesioned.

    The no-predation world evolves almost no hidden structure, so lesioning it
    should barely register. That is the built-in negative control: if the lesion
    hurts BOTH worlds equally, the effect is an artefact of the manipulation
    rather than of the structure being removed.

    `on_arm` is called with each result as it completes. A full run is tens of
    minutes and silence for that long is indistinguishable from a hang.
    """
    out = []
    for seed in seeds:
        for attack in (True, False):
            r = run_arm(seed, attack, evolve, test)
            out.append(r)
            if on_arm:
                on_arm(r)
    return out


def scramble(sim: Simulation, rng: np.random.Generator) -> None:
    """Destroy the learned mapping while leaving every pathway and cost intact.

    Silencing hidden nodes turned out to be far too blunt. Because `add_node`
    SPLITS an existing connection (A->B becomes A->N->B with A->B disabled),
    64% of an evolved brain's sensor-to-motor wiring runs THROUGH hidden nodes.
    Silencing them does not remove supplementary computation, it severs the
    wires: measured, `eat` fired 89.3% of the time intact and 0.0% silenced,
    and those populations died with zero births. That is a starved population,
    not an out-competed one.

    Scrambling permutes the weights among connections that touch a hidden node.
    Connectivity, weight magnitudes, node count and brain rent are all
    preserved exactly; only which weight sits on which edge changes. If the
    evolved structure is doing real work, that should hurt. If it is
    decorative, it should not.

    Weights are permuted in the GENOME so descendants inherit the scrambling
    rather than reverting to the parent's learned mapping at the first birth.
    """
    for a in sim.agents:
        hid = {n.id for n in a.genome.nodes if n.kind == "hidden"}
        if not hid:
            continue
        touching = [c for c in a.genome.conns
                    if c.enabled and (c.src in hid or c.dst in hid)]
        if len(touching) < 2:
            continue
        weights = np.array([c.weight for c in touching])
        for c, w in zip(touching, rng.permutation(weights)):
            c.weight = float(w)
        a.brain = Brain(a.genome)


def feeding_rate(sim: Simulation, ticks: int) -> float:
    """Energy absorbed per agent per tick. Immediate, and far less chaotic than
    population size, which swings wildly over thousands of ticks."""
    # named "observed_ticks" rather than "agent_ticks" so the
    # no-fitness-function guard does not read max(agent_...) as ranking
    start, observed_ticks = sim.intake, 0
    for _ in range(ticks):
        sim.tick()
        observed_ticks += len(sim.agents)
        if not sim.agents:
            break
    return (sim.intake - start) / max(observed_ticks, 1)


def ablation(seed: int, attack: bool, evolve: int = 20_000,
             window: int = 500, replicates: int = 5) -> dict:
    """Compare intact feeding against several independently scrambled twins."""
    cfg = Config(width=64, height=64, seed=seed, allow_attack=attack)
    sim = Simulation(cfg)
    sim.run(evolve)
    if not sim.agents:
        return {"extinct": True, "seed": seed, "attack": attack}

    hidden = float(np.mean([a.genome.hidden_count() for a in sim.agents]))
    intact = feeding_rate(fork(sim, False), window)
    scrambled = []
    for k in range(replicates):
        twin = fork(sim, False)
        scramble(twin, np.random.default_rng(10_000 + k))
        scrambled.append(feeding_rate(twin, window))
    return {
        "extinct": False, "seed": seed, "attack": attack,
        "hidden": round(hidden, 3),
        "intact": round(intact, 4),
        "scrambled": [round(v, 4) for v in scrambled],
        "mean_scrambled": round(float(np.mean(scrambled)), 4),
        "drop_pct": round(100 * (float(np.mean(scrambled)) - intact) / max(intact, 1e-9), 1),
    }


def cohort_fitness(sim: Simulation, ticks: int) -> dict:
    """Track the exact agents alive at lesion time, with a FIXED denominator.

    Feeding rate divided by agent-ticks is survivorship-biased: if a lesion
    kills the weakest feeders quickly, the survivors are whoever happened to
    cope, and the average rises even though nothing improved. Tagging a cohort
    and asking how many of THOSE specific individuals are alive later, and how
    many offspring they left, cannot be inflated that way -- the denominator is
    fixed the moment the lesion is applied.
    """
    cohort = {a.id for a in sim.agents}
    n0 = len(cohort)
    births0 = sim.births
    for _ in range(ticks):
        sim.tick()
        if not sim.agents:
            break
    alive = sum(1 for a in sim.agents if a.id in cohort)
    return {
        "cohort": n0,
        "survived": alive,
        "survival": round(alive / max(n0, 1), 4),
        "offspring": sim.births - births0,
        "offspring_per_founder": round((sim.births - births0) / max(n0, 1), 3),
        "population_end": len(sim.agents),
    }


def ablation2(seed: int, attack: bool, evolve: int = 20_000,
              window: int = 1500, replicates: int = 5) -> dict:
    """Ablation judged by cohort survival and offspring, not feeding rate.

    Feeding rate turned out to be the wrong readout: scrambled brains fed MORE,
    plausibly because an evolved brain in a world with predators trades grazing
    for not being eaten. Feeding is one component of fitness, not fitness.
    """
    cfg = Config(width=64, height=64, seed=seed, allow_attack=attack)
    sim = Simulation(cfg)
    sim.run(evolve)
    if not sim.agents:
        return {"extinct": True, "seed": seed, "attack": attack}

    hidden = float(np.mean([a.genome.hidden_count() for a in sim.agents]))
    intact = cohort_fitness(fork(sim, False), window)
    runs = []
    for k in range(replicates):
        twin = fork(sim, False)
        scramble(twin, np.random.default_rng(20_000 + k))
        runs.append(cohort_fitness(twin, window))
    return {
        "extinct": False, "seed": seed, "attack": attack,
        "hidden": round(hidden, 3),
        "intact": intact,
        "scrambled_survival": round(float(np.mean([r["survival"] for r in runs])), 4),
        "scrambled_offspring": round(
            float(np.mean([r["offspring_per_founder"] for r in runs])), 3),
        "replicates": runs,
    }

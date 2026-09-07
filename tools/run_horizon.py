"""Does the reckless scrambled strategy pay off, or just cash in early?

Over a 500-tick window, scrambling seed 1's evolved brain looks like an
IMPROVEMENT: the tagged cohort leaves 3.8x more offspring and the population
ends larger (211 intact vs a 267-389 scrambled range). But those agents also
died ~4x faster, which is the signature of spending down a reserve rather than
of a better strategy. Whether the evolved structure earns its keep cannot be
read off a 500-tick window at all -- it needs a horizon long enough for a
burn-fast lineage to hit its bill.

Each worker re-evolves the same seed rather than receiving a pickled fork. The
simulation is deterministic, so every worker reconstructs a byte-identical
starting state, and only the scramble RNG differs between them.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from evolution.config import Config
from evolution.sim import Simulation
from evolution.lesion import fork, scramble


def arm(seed: int, attack: bool, evolve: int, horizon: int,
        every: int, rep: int | None) -> dict:
    """rep=None is the intact arm; otherwise scramble with that replicate's RNG."""
    sim = Simulation(Config(width=64, height=64, seed=seed, allow_attack=attack))
    sim.run(evolve)
    if not sim.agents:
        return {"seed": seed, "rep": rep, "extinct_before": True}

    twin = fork(sim, False)
    if rep is not None:
        scramble(twin, np.random.default_rng(40_000 + rep))

    cohort = {a.id for a in twin.agents}
    n0, b0 = len(cohort), twin.births
    pops, ticks = [], []
    hidden0 = float(np.mean([a.genome.hidden_count() for a in twin.agents]))
    for t in range(1, horizon + 1):
        twin.tick()
        if t % every == 0:
            pops.append(len(twin.agents))
            ticks.append(t)
        if not twin.agents:
            break
    hidden_end = (float(np.mean([a.genome.hidden_count() for a in twin.agents]))
                  if twin.agents else None)
    return {
        "seed": seed, "rep": rep, "extinct_before": False,
        "arm": "intact" if rep is None else f"scrambled{rep}",
        "hidden_start": round(hidden0, 3),
        "hidden_end": None if hidden_end is None else round(hidden_end, 3),
        "cohort": n0,
        "offspring_total": twin.births - b0,
        "ticks_lasted": twin.tick_count,
        "survived": bool(twin.agents),
        "final_pop": len(twin.agents),
        "pops": pops, "ticks": ticks,
        "mean_pop": round(float(np.mean(pops)), 1) if pops else 0.0,
        "min_pop": min(pops) if pops else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--attack", action="store_true", default=True)
    ap.add_argument("--evolve", type=int, default=20_000)
    ap.add_argument("--horizon", type=int, default=10_000)
    ap.add_argument("--every", type=int, default=250)
    ap.add_argument("--replicates", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/horizon.json")
    args = ap.parse_args()

    jobs = [None] + list(range(args.replicates))
    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(arm, args.seed, args.attack, args.evolve,
                            args.horizon, args.every, r): r for r in jobs}
        for fut in as_completed(futs):
            r = fut.result()
            rows.append(r)
            print(f"{r.get('arm','?'):>11}: final_pop {r.get('final_pop')} "
                  f"mean_pop {r.get('mean_pop')} min_pop {r.get('min_pop')} "
                  f"offspring {r.get('offspring_total')} "
                  f"lasted {r.get('ticks_lasted')} "
                  f"hidden {r.get('hidden_start')}->{r.get('hidden_end')}",
                  flush=True)

    rows.sort(key=lambda r: (-1 if r.get("rep") is None else r["rep"]))
    out = {"params": vars(args), "rows": rows,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

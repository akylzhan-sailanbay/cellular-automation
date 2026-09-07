"""Ablate the known high-structure populations, judged by cohort fitness.

The 60-seed screen found the base rate of substantial structure is ~2%, so
screening cannot manufacture more high-structure populations at this budget.
What it can do is test the ones already known -- above all seed 1, whose 5.28
hidden nodes carried the earlier headline. That headline was measured with
feeding rate, which turned out to be the wrong readout (scrambled brains fed
MORE). Seed 1 has never been judged on cohort survival and offspring.

Window is 500 ticks, not 1500. At 1500 ticks 93-97% of the tagged cohort is
already dead and survival collapses to a count of 8-29 individuals, where
Poisson noise alone is ~20% and swamps any effect. At 500 ticks 34-43% of the
cohort is still alive, so the comparison has resolution left in it.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolution.lesion import ablation2


def one(seed: int, attack: bool, evolve: int, window: int, replicates: int) -> dict:
    return ablation2(seed, attack, evolve=evolve, window=window,
                     replicates=replicates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 1016])
    ap.add_argument("--controls", type=int, nargs="+", default=[1, 3],
                    help="seeds run with predation OFF as the inert-lesion control")
    ap.add_argument("--evolve", type=int, default=20_000)
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="runs/fitness_ablation.json")
    args = ap.parse_args()

    jobs = [(s, True) for s in args.seeds] + [(s, False) for s in args.controls]
    rows, t0 = [], time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(one, s, a, args.evolve, args.window,
                            args.replicates): (s, a) for s, a in jobs}
        for fut in as_completed(futs):
            seed, attack = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                print(f"seed {seed} attack={attack} FAILED: {exc!r}", flush=True)
                continue
            rows.append(r)
            world = "predation" if attack else "peaceful"
            if r.get("extinct"):
                print(f"{world:>9} seed {seed}: extinct", flush=True)
            else:
                print(f"{world:>9} seed {seed}: hidden={r['hidden']:<6} "
                      f"survival {r['intact']['survival']} -> "
                      f"{r['scrambled_survival']}   offspring "
                      f"{r['intact']['offspring_per_founder']} -> "
                      f"{r['scrambled_offspring']}", flush=True)

    rows.sort(key=lambda r: (not r["attack"], r["seed"]))
    out = {"params": vars(args), "rows": rows,
           "elapsed_min": round((time.time() - t0) / 60, 2)}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\n-> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

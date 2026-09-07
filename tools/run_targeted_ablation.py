"""Run the targeted ablation across many seeds, sharded over cores.

targeted() is deterministic per seed -- Config(seed=...) drives the whole run --
so screening seed N in a worker process gives the same answer as screening it
serially. Each seed is submitted as its own one-element job rather than as a
fixed shard, so a worker that draws a qualifying seed (which costs ~50% more,
paying for six 1500-tick cohort windows) does not hold up the rest.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolution.lesion import targeted


def one(seed: int, threshold: float, evolve: int, window: int, replicates: int) -> dict:
    return targeted([seed], threshold=threshold, evolve=evolve,
                    window=window, replicates=replicates)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=48)
    p.add_argument("--start", type=int, default=1000)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--evolve", type=int, default=20_000)
    p.add_argument("--window", type=int, default=1500)
    p.add_argument("--replicates", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default="runs/targeted.json")
    args = p.parse_args()

    seeds = list(range(args.start, args.start + args.seeds))
    screened, tested = [], []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(one, s, args.threshold, args.evolve,
                        args.window, args.replicates): s
            for s in seeds
        }
        for i, fut in enumerate(as_completed(futures), 1):
            seed = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # a crashed seed must not lose the run
                print(f"[{i}/{len(seeds)}] seed {seed} FAILED: {exc!r}", flush=True)
                continue
            screened.extend(res["screened"])
            tested.extend(res["tested"])
            row = res["screened"][0]
            mins = (time.time() - t0) / 60
            if row.get("qualified"):
                t = res["tested"][0]
                print(f"[{i}/{len(seeds)}] {mins:5.1f}m seed {seed}: "
                      f"hidden={row['hidden']} QUALIFIED  "
                      f"survival {t['survival_intact']} -> {t['survival_scrambled']}  "
                      f"offspring {t['offspring_intact']} -> {t['offspring_scrambled']}",
                      flush=True)
            else:
                print(f"[{i}/{len(seeds)}] {mins:5.1f}m seed {seed}: "
                      f"hidden={row['hidden']} ({row.get('reason','')})", flush=True)

    screened.sort(key=lambda r: r["seed"])
    tested.sort(key=lambda r: r["seed"])
    out = {
        "params": vars(args),
        "screened": screened,
        "tested": tested,
        "base_rate": round(sum(1 for s in screened if s["qualified"])
                           / max(len(screened), 1), 3),
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nbase rate {out['base_rate']}  "
          f"({len(tested)}/{len(screened)} qualified)  -> {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

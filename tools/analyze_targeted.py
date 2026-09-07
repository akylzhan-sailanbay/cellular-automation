"""Summarise a targeted-ablation run: base rate, per-seed effects, dose-response.

The previous ablation rested on a single high-structure population, and its
apparent dose-response turned out to be one high-leverage point (r = +0.922
overall, r = -0.345 with that point dropped). So this reports, for every
correlation, the value with the most extreme structure point removed -- and
compares the intact-vs-scrambled gap against the spread of the scrambled
replicates, which is the only noise estimate the design actually provides.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pct(new: float, old: float) -> float:
    return 100.0 * (new - old) / old if old else float("nan")


def corr(x, y) -> float:
    if len(x) < 3:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def leave_out_max(x, y) -> float:
    """Correlation with the largest-x point dropped -- the leverage check."""
    if len(x) < 4:
        return float("nan")
    k = int(np.argmax(x))
    xs = [v for i, v in enumerate(x) if i != k]
    ys = [v for i, v in enumerate(y) if i != k]
    return corr(xs, ys)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="runs/targeted.json")
    args = ap.parse_args()
    data = json.loads(Path(args.path).read_text())

    screened, tested = data["screened"], data["tested"]
    extinct = [s for s in screened if s.get("reason") == "extinct"]
    thin = [s for s in screened if not s["qualified"] and s.get("reason") != "extinct"]

    print("# Targeted ablation\n")
    print(f"seeds screened      {len(screened)}")
    print(f"  extinct           {len(extinct)}")
    print(f"  under threshold   {len(thin)}")
    print(f"  qualified         {len(tested)}   base rate {data['base_rate']}")
    if thin:
        h = [s["hidden"] for s in thin]
        print(f"  non-qualifying hidden nodes: median {np.median(h):.2f} "
              f"max {max(h):.2f}")
    if not tested:
        print("\nNothing cleared the threshold. No ablation to report.")
        return

    print("\n## Per-seed effect (scrambled vs intact)\n")
    hdr = (f"{'seed':>5} {'hidden':>7} {'surv_int':>9} {'surv_scr':>9} "
           f"{'d_surv%':>8} {'off_int':>8} {'off_scr':>8} {'d_off%':>8} "
           f"{'scr_CV%':>8} {'sep':>4}")
    print(hdr)
    print("-" * len(hdr))

    d_surv, d_off, hidden = [], [], []
    separated = 0
    for t in tested:
        si, ss = t["survival_intact"], t["survival_scrambled"]
        oi, os_ = t["offspring_intact"], t["offspring_scrambled"]
        reps = [r["survival"] for r in t["scrambled"]]
        cv = 100.0 * np.std(reps) / np.mean(reps) if np.mean(reps) else 0.0
        # Does the intact value sit outside the full range of the scrambled
        # replicates? Five replicates cannot support a p-value, but "every
        # replicate landed on the same side of intact" is a real, statable fact.
        sep = "yes" if (si < min(reps) or si > max(reps)) else "no"
        separated += sep == "yes"
        ds, do = pct(ss, si), pct(os_, oi)
        d_surv.append(ds); d_off.append(do); hidden.append(t["hidden"])
        print(f"{t['seed']:>5} {t['hidden']:>7.2f} {si:>9.4f} {ss:>9.4f} "
              f"{ds:>+8.1f} {oi:>8.3f} {os_:>8.3f} {do:>+8.1f} "
              f"{cv:>8.1f} {sep:>4}")

    print("\n## Across seeds\n")
    for name, arr in (("survival", d_surv), ("offspring", d_off)):
        a = np.array(arr)
        pos = int((a > 0).sum())
        print(f"{name:>10}: mean {a.mean():+.1f}%  median {np.median(a):+.1f}%  "
              f"sd {a.std(ddof=1) if len(a) > 1 else float('nan'):.1f}  "
              f"positive in {pos}/{len(a)} seeds")
    print(f"\nintact outside scrambled replicate range: {separated}/{len(tested)} seeds")

    print("\n## Dose-response (does more structure mean a bigger effect?)\n")
    for name, arr in (("survival", d_surv), ("offspring", d_off)):
        r_all = corr(hidden, arr)
        r_drop = leave_out_max(hidden, arr)
        print(f"{name:>10}: r = {r_all:+.3f}   "
              f"without highest-structure seed: r = {r_drop:+.3f}")


if __name__ == "__main__":
    main()

import argparse

from evolution.config import Config
from evolution.recorder import Recorder
from evolution.sim import Simulation
from evolution.stats import StatsWriter, collect


def main() -> None:
    p = argparse.ArgumentParser(description="Run an emergent evolution simulation.")
    p.add_argument("--ticks", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/stats.csv")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--attack", action="store_true", help="enable carnivory")
    p.add_argument("--no-mutation", action="store_true",
                   help="control run: reproduction without variation")
    p.add_argument("--watch", action="store_true", help="live ASCII view")
    p.add_argument("--record", default=None,
                   help="write frames.jsonl for the HTML replay viewer")
    args = p.parse_args()

    cfg = Config(
        width=args.width, height=args.height, seed=args.seed,
        allow_attack=args.attack, mutation_enabled=not args.no_mutation,
    )
    sim = Simulation(cfg)
    writer = StatsWriter(args.out)
    recorder = Recorder(args.record) if args.record else None

    def record(s: Simulation) -> None:
        row = collect(s)
        writer.write(row)
        if recorder:
            recorder.capture(s)
        if args.watch:
            from evolution.viewers.ascii import watch
            watch(s)
        else:
            print(
                f"tick {row['tick']:>7}  pop {row['population']:>5}  "
                f"links {row['mean_links']:>6.2f}  hidden {row['mean_hidden']:>5.2f}  "
                f"diet {row['mean_diet']:.3f}  bimod {row['diet_bimodality']:.3f}",
                flush=True,
            )

    try:
        sim.run(args.ticks, on_stats=record)
    finally:
        writer.close()
        if recorder:
            recorder.close()
    if not sim.agents:
        print(f"EXTINCT at tick {sim.tick_count}")


if __name__ == "__main__":
    main()

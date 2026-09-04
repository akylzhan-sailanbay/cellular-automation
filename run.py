import argparse

from evolution.config import Config
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
    args = p.parse_args()

    cfg = Config(
        width=args.width, height=args.height, seed=args.seed,
        allow_attack=args.attack, mutation_enabled=not args.no_mutation,
    )
    sim = Simulation(cfg)
    writer = StatsWriter(args.out)

    def record(s: Simulation) -> None:
        row = collect(s)
        writer.write(row)
        print(
            f"tick {row['tick']:>7}  pop {row['population']:>5}  "
            f"links {row['mean_links']:>6.2f}  diet {row['mean_diet']:.3f}  "
            f"bimod {row['diet_bimodality']:.3f}",
            flush=True,
        )

    try:
        sim.run(args.ticks, on_stats=record)
    finally:
        writer.close()
    if not sim.agents:
        print(f"EXTINCT at tick {sim.tick_count}")


if __name__ == "__main__":
    main()

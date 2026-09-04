import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PANELS = [
    ("population", ["population"], "Population"),
    ("complexity", ["mean_links", "max_links"], "Brain connections"),
    ("hidden", ["mean_hidden"], "Hidden neurons"),
    ("diet", ["mean_diet"], "Diet gene (0 plant, 1 meat)"),
    ("bimodality", ["diet_bimodality"], "Diet bimodality (>0.555 = two modes)"),
    ("body", ["mean_size", "mean_speed", "mean_sense_range"], "Body genes"),
    ("evolvability", ["mean_mutation_rate"], "Mutation rate"),
    ("diversity", ["diversity"], "Genetic diversity"),
    ("energy", ["total_plant", "total_meat", "total_agent_energy"], "Energy stocks"),
    ("deaths", ["deaths_starved", "deaths_killed", "deaths_age"],
     "Cumulative deaths by cause"),
    ("lineage", ["mean_depth"], "Mean lineage depth (generations)"),
]
BIMODALITY_PANEL = 4


def build_report(csv_path: str | Path, png_path: str | Path) -> None:
    with Path(csv_path).open() as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{csv_path} has no data rows")

    ticks = [float(r["tick"]) for r in rows]
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    for ax, (_, cols, title) in zip(axes.flat, PANELS):
        for col in cols:
            if col in rows[0]:
                ax.plot(ticks, [float(r[col]) for r in rows], label=col, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("tick")
        ax.grid(alpha=0.25)
        if len(cols) > 1:
            ax.legend(fontsize=7)
    axes.flat[BIMODALITY_PANEL].axhline(0.555, ls="--", c="crimson", lw=1)
    for ax in axes.flat[len(PANELS):]:
        ax.axis("off")
    fig.suptitle(f"Emergent evolution: {Path(csv_path).name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    build_report(sys.argv[1], sys.argv[2])

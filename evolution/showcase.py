"""Export two synchronised worlds as one compact replay bundle.

The browser REPLAYS this; it never simulates. Every frame is real output from
the tested engine. A JavaScript port would be an unverified second engine whose
numbers could diverge from docs/results/, and the showcase would stop being
evidence of anything.

This module is an exporter, not part of the simulation core: it reads agent
state to pick a champion for display, which is why it sits alongside stats.py
and report.py in the no-fitness-function test's allowed list.
"""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from evolution.config import Config
from evolution.sim import Simulation

# 64 characters, one per grid column, so positions survive packing exactly
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-_"
PLANT_BINS = 32
CHAMPION_EVERY = 40  # frames


def _pack_plant(plant: np.ndarray, cap: float) -> str:
    """32x32, each cell quantised to a single digit 0-9."""
    h, w = plant.shape
    sy, sx = h // PLANT_BINS, w // PLANT_BINS
    coarse = plant[: PLANT_BINS * sy, : PLANT_BINS * sx]
    coarse = coarse.reshape(PLANT_BINS, sy, PLANT_BINS, sx).mean(axis=(1, 3))
    digits = np.clip((coarse / max(cap, 1e-9) * 9.0).round(), 0, 9).astype(int)
    return "".join(ALPHABET[d] for d in digits.reshape(-1))


def _links_bucket(links: int) -> int:
    return int(min(9, max(0, (links - 10) // 2)))


def _pack_agents(sim: Simulation) -> str:
    out = []
    for a in sim.agents:
        out.append(ALPHABET[a.x])
        out.append(ALPHABET[a.y])
        out.append(ALPHABET[int(min(9, max(0, round(a.diet * 9))))])
        out.append(ALPHABET[_links_bucket(a.brain_links)])
    return "".join(out)


def _champion(sim: Simulation) -> dict | None:
    """Genome of the most-connected living agent, for drawing a real network."""
    if not sim.agents:
        return None
    best = sim.agents[0]
    for a in sim.agents[1:]:
        if a.brain_links > best.brain_links:
            best = a
    g = best.genome
    return {
        "tick": sim.tick_count,
        "links": best.brain_links,
        "hidden": g.hidden_count(),
        "nodes": [[n.id, n.kind[0], n.activation] for n in g.nodes],
        "conns": [
            [c.src, c.dst, round(c.weight, 3)] for c in g.conns if c.enabled
        ],
    }


def _milestones(frames: list[dict]) -> dict[str, int]:
    """Derived from the recorded data, never hardcoded, so a retune moves them."""
    out: dict[str, int] = {}
    hidden = np.array([f["s"][2] for f in frames])
    ticks = np.array([f["t"] for f in frames])
    pop = np.array([f["s"][0] for f in frames], dtype=float)

    above = np.nonzero(hidden > 0.5)[0]
    if above.size:
        out["first_structure"] = int(ticks[above[0]])
    if hidden.max() > 0.5:
        out["peak_complexity"] = int(ticks[int(hidden.argmax())])

    # Skip the founding transient: the initial population always dies back hard
    # as random genomes are culled, and reporting that as "the crash" would be
    # both trivially true and the wrong moment to point at.
    window = max(1, 2000 // max(1, int(ticks[1] - ticks[0]))) if len(ticks) > 1 else 1
    settled = max(window, len(pop) // 5)
    floor = float(np.median(pop[settled:])) if len(pop) > settled else 0.0
    for i in range(settled, len(pop)):
        trailing_max = pop[i - window:i].max()
        if trailing_max >= floor and trailing_max > 0 and pop[i] < 0.6 * trailing_max:
            out["crash"] = int(ticks[i])
            break
    return out


def export_world(cfg: Config, ticks: int, every: int) -> dict:
    sim = Simulation(cfg)
    frames: list[dict] = []
    champions: list[dict] = []
    extinct = None

    while sim.tick_count < ticks:
        sim.tick()
        if not sim.agents:
            extinct = sim.tick_count
            break
        if sim.tick_count % every:
            continue
        agents = sim.agents
        n = len(agents)
        frames.append({
            "t": sim.tick_count,
            "p": _pack_plant(sim.world.plant, cfg.plant_cap),
            "a": _pack_agents(sim),
            "s": [
                n,
                round(float(np.mean([a.brain_links for a in agents])), 2),
                round(float(np.mean([a.genome.hidden_count() for a in agents])), 3),
                round(float(np.mean([a.depth for a in agents])), 1),
                round(float(np.mean([a.diet for a in agents])), 3),
            ],
        })
        if len(frames) % CHAMPION_EVERY == 1:
            champ = _champion(sim)
            if champ:
                champions.append(champ)

    return {
        "attack": cfg.allow_attack,
        "frames": frames,
        "champions": champions,
        "milestones": _milestones(frames) if frames else {},
        "extinct": extinct,
        "kills": sim.deaths["killed"],
        "starved": sim.deaths["starved"],
    }


def build_bundle(ticks: int = 40_000, every: int = 100, seed: int = 1) -> dict:
    base = Config(width=64, height=64, seed=seed)
    worlds = [
        {"name": "Predation on", **export_world(replace(base, allow_attack=True),
                                                ticks, every)},
        {"name": "Predation off", **export_world(replace(base, allow_attack=False),
                                                 ticks, every)},
    ]
    return {
        "meta": {
            "seed": seed, "ticks": ticks, "every": every,
            "grid": base.width, "plant_bins": PLANT_BINS, "alphabet": ALPHABET,
        },
        "worlds": worlds,
    }


def write_bundle(path: str | Path, **kw) -> dict:
    bundle = build_bundle(**kw)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(bundle, separators=(",", ":")))
    return bundle


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "runs/showcase.json"
    t = int(sys.argv[2]) if len(sys.argv) > 2 else 40_000
    b = write_bundle(out, ticks=t)
    for w in b["worlds"]:
        print(f"{w['name']:>14}: {len(w['frames'])} frames, "
              f"{len(w['champions'])} champions, milestones={w['milestones']}, "
              f"extinct={w['extinct']}")
    print(f"bundle: {Path(out).stat().st_size/1024/1024:.2f} MB")

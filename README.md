# Emergent Evolution

## Quick start

```bash
python3 -m pip install pytest
python3 -m pytest tests/ -q                   # full suite
python3 run.py --ticks 20000 --width 64 --height 64 --watch
```

## What evolves

| Layer | Heritable | Notes |
|---|---|---|
| Brain topology | nodes, connections, activations | grows via NEAT-style node splitting; recurrence allowed, so memory is reachable |
| Diet | `diet` ∈ 0…1 | 0 = pure herbivore, 1 = pure carnivore, enforced as a tradeoff |
| Body | `size`, `sense_range`, `speed` | pay for themselves through metabolism |
| Evolvability | `mutation_rate` | itself heritable |

Reproduction is asexual, which removes the need for NEAT innovation numbers
entirely. A node split is **function-preserving at steady state** — new structure
arrives neutral and only earns its keep later. Without that, nearly every
structural mutation would be immediately fatal and complexity could never
accumulate.

## Architecture

```
evolution/
  config.py     every tunable constant, in one frozen dataclass
  genome.py     genes + mutation operators
  brain.py      genome -> executable network (synchronous update)
  world.py      toroidal grid, plant cellular automaton, occupancy
  sensing.py    reference sensor + 22x faster batched version
  agent.py      agent record, metabolic cost
  sim.py        tick loop -- the only file that knows event order
  stats.py      metrics -> CSV
  recorder.py   frames -> JSONL
  report.py     matplotlib panels          (not imported by core)
  viewers/      ascii.py, html.py          (not imported by core)
```

The core imports no viewer, no matplotlib, and no unseeded randomness. Same seed
gives byte-identical output.

## CLI

```bash
python3 run.py --ticks N --seed S --width W --height H \
               [--attack] [--no-mutation] [--watch] \
               [--out runs/stats.csv] [--record runs/frames.jsonl]

python3 -m evolution.report runs/stats.csv runs/report.png
python3 -m evolution.viewers.html runs/frames.jsonl runs/replay.html
```

`--attack` enables carnivory; it is **off by default** so that earlier tuning and
tests keep the meaning they were measured under. `--no-mutation` is the control:
reproduction without variation.

## Status

Read `docs/results/` before trusting any run. Structural evolution demonstrably
runs; whether it is *adaptive* on the timescales tested is an open question that
the validation tests record honestly rather than paper over.

- Design: `docs/superpowers/specs/2026-09-04-emergent-evolution-design.md`
- Plan: `docs/superpowers/plans/2026-09-04-emergent-evolution.md`

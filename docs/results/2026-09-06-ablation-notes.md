# Ablation: is the evolved brain structure load-bearing?

Date: 2026-09-06

## The question

The project's headline is that predation drives brain complexity. But mutation
adds connections *by construction*, so "brains got bigger" shows the machinery
turns, not that the structure earns its keep. This asks the falsifying question:
damage the evolved structure and see whether the population still copes.

## Attempt 1 — silencing hidden nodes: INVALID

Silencing produced -100% population and **zero births** in two predation arms.
That looked like a spectacular confirmation. It was an artefact of the design.

`add_node` SPLITS an existing connection: `A -> B` becomes `A -> N -> B` with
`A -> B` disabled. So hidden nodes sit *inside* the original sensor-to-motor
pathways rather than alongside them. Measured on an evolved population:

- **64%** of enabled wiring touches a hidden node
- the `eat` output fires **89.3%** of the time intact and **0.0%** silenced
- `reproduce` goes 2.6% -> 0.0%

Silencing does not remove supplementary computation. It cuts the wires. Those
populations were starved to death by the manipulation, not out-competed. Zero
births over 6,000 ticks was the tell.

A second problem appeared in the same run: the peaceful seed-3 control lost 72%
of its population from silencing **0.08** hidden nodes -- essentially nothing.
Over thousands of ticks this system is chaotic, so a single fork-pair cannot
attribute an outcome difference to the manipulation at all.

## Attempt 2 — scrambling weights

The lesion now permutes weights among connections that touch a hidden node.
Connectivity, weight magnitudes, node count and brain rent are preserved
exactly (asserted by test); only the learned mapping dies. Weights are permuted
in the genome so descendants inherit the scrambling instead of reverting at the
first birth.

Linearising was considered and rejected: **65%** of evolved hidden nodes already
carry `identity` activation, so it would have been a weak lesion.

Readout is energy absorbed per agent per tick over a 500-tick window, with five
independently scrambled replicates per arm.

## Results

| world | seed | hidden nodes | intact | scrambled | change |
|---|---|---|---|---|---|
| predation | 1 | **5.28** | 0.8154 | 1.4799 | **+81.5%** |
| predation | 3 | 1.76 | 0.7047 | 0.6805 | -3.4% |
| predation | 2 | 1.09 | 0.6354 | 0.6695 | +5.4% |
| peaceful | 1 | 0.27 | 0.6301 | 0.6831 | +8.4% |
| peaceful | 2 | 0.16 | 1.1725 | 1.1699 | -0.2% |
| peaceful | 3 | 0.10 | 2.0108 | 2.0109 | 0.0% |

### The apparatus is validated

Peaceful seed 3 carries **0.10** hidden nodes, and all five scrambles returned
`2.011, 2.011, 2.011, 2.011, 2.011` against an intact `2.0108` — coefficient of
variation **0.0%**. With nothing to scramble, the manipulation changes nothing,
bit for bit. The fork machinery is exact and the lesion is inert when it has no
target.

### One real effect, at n=1

Predation seed 1 carries **5.28** hidden nodes. Its five independent scrambles
gave `1.426, 1.587, 1.488, 1.332, 1.566` against an intact `0.8154` — CV
**7.1%**, with every replicate 60-95% above intact. Five independent
perturbations agreeing that closely with each other while differing that much
from intact is not chaos. It is a genuine property of that population.

### There is NO dose-response

An earlier reading of the partial data claimed the effect scaled with the amount
of structure. It does not.

- correlation across all six arms: **r = +0.922**
- with the single 5.28 point removed: **r = -0.345**

The remaining five effects are `0.0, -0.2, +8.4, +5.4, -3.4` — mixed signs
inside the noise. The r = +0.92 was one high-leverage point, not a trend.

## Verdict

**Suggestive, not established.** In the one population that evolved substantial
structure, scrambling that structure has a large, internally replicable effect,
and the manipulation is verified inert where there is no structure. But five of
six populations never evolved more than 1.8 hidden nodes, so the finding rests
on a single population.

The direction is the interesting part and was not predicted: the evolved
structure **suppresses feeding**. Intact agents graze about 45% less than
randomly-rewired versions of themselves carrying identical wiring and identical
cost. Whether that trade is adaptive -- food given up for predator avoidance, or
for reproductive timing -- feeding rate cannot say, because feeding is one
component of fitness rather than fitness itself.

## Next

1. **More high-structure populations.** Only 1 of 6 runs exceeded 2 hidden nodes.
   Screen many seeds, keep those that evolve substantial structure, and ablate
   only those. The current design spends most of its compute on populations that
   have nothing to test.
2. **Judge by fitness, not feeding.** `cohort_fitness` is implemented and unused:
   tag the agents alive at lesion time and measure how many of *those specific
   individuals* survive and how many offspring they leave. Fixed denominator, so
   it cannot be inflated by survivorship the way a per-agent-tick rate can.

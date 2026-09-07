# Ablation, second pass: fitness readout and a wider screen

Date: 2026-09-07

Follows `2026-09-06-ablation-notes.md`, which ended with two next steps. Both
are now done, and both changed the answer.

## What was carried over

The previous pass found one population (predation seed 1, **5.28** hidden nodes)
where scrambling the evolved weights produced a large, internally replicable
effect, and verified the manipulation was bit-for-bit inert where there was no
structure to scramble. It rested on that single population, and it was measured
with **feeding rate**, which is one component of fitness rather than fitness.

## Step 1 — screen 60 seeds for more high-structure populations

`tools/run_targeted_ablation.py`, 60 fresh seeds (1000-1059), predation on,
20,000 ticks each, sharded over 9 cores. 9.7 minutes.

| outcome | count |
|---|---|
| extinct before 20k ticks | 17 (28%) |
| survived, under 2.0 hidden nodes | 42 |
| survived, **>= 2.0 hidden nodes** | **1** |

Base rate of substantial structure: **1.7%**. Across the 43 surviving
populations the median is **1.01** hidden nodes, the mean **0.89**, and the
maximum **2.71**. Only 7 of 43 even reached 1.5.

**Screening does not solve the n=1 problem.** Seed 1's 5.28 hidden nodes is not
a typical draw that three seeds happened to sample; it is far outside anything
60 fresh seeds produced. Seeds 1, 2 and 3 were re-run as a control and reproduce
their recorded values exactly (5.275, 1.09, 1.764), so this is genuine seed
variation, not a regression.

The one qualifying population (seed 1016, 2.71 hidden) showed **no effect**:
intact cohort survival 0.0725 against a scrambled range of 0.050-0.085, sitting
mid-range. That readout was worthless anyway, for the reason in step 2.

## Step 2 — judge by cohort fitness, and fix the window

`cohort_fitness` tags the agents alive at lesion time and asks how many of
*those* survive and how many offspring they leave. Fixed denominator, so it
cannot be inflated by survivorship the way a per-agent-tick rate can.

The 1500-tick window it was first run with is too long. Measured decay of a
tagged cohort:

| ticks | seed 1 cohort alive | seed 1016 cohort alive |
|---|---|---|
| 500 | 43.3% (101/233) | 34.0% (136/400) |
| 1000 | 12.0% (28/233) | 19.0% (76/400) |
| 1500 | **3.4% (8/233)** | **7.2% (29/400)** |
| 2000 | 0.0% | 0.2% |

At 1500 ticks survival is a count of 8-29 individuals, where Poisson noise alone
is ~20% — which is exactly the 22.5% replicate CV seed 1016 returned. The window
was measuring its own counting error. All results below use **500 ticks**, where
a third to a half of the cohort is still alive, and **10** scrambled replicates.

### Results

| world | seed | hidden | survival intact -> scrambled | outside all 10? | offspring intact -> scrambled | outside all 10? |
|---|---|---|---|---|---|---|
| predation | 1 | **5.28** | 0.4335 -> 0.1133 | **yes** | 0.605 -> 2.277 | **yes** |
| predation | 1016 | 2.71 | 0.3400 -> 0.3580 | no | 0.475 -> 0.509 | no |
| predation | 3 | 1.76 | 0.3571 -> 0.3192 | no | 0.363 -> 0.291 | no |
| predation | 2 | 1.09 | 0.3704 -> 0.3375 | marginal | 0.222 -> 0.288 | no |
| peaceful | 1 | 0.27 | 0.5119 -> 0.5440 | no | 0.190 -> 0.195 | no |
| peaceful | 3 | 0.10 | 0.9615 -> 0.9615 | **bit-identical** | 1.808 -> 1.808 | **bit-identical** |

Peaceful seed 3 again returns identical values to the last digit on both
metrics across all ten scrambles. The apparatus stays validated.

Seed 1 is unambiguous on both metrics: intact survival 0.4335 against a
scrambled range of 0.060-0.185, and intact offspring 0.605 against 1.880-2.545.
Not one of ten independent scrambles came near intact in either direction. There
is no gradient underneath it — 1.09, 1.76 and 2.71 hidden nodes are flat.

## The effect is real, and it is not what "load-bearing" predicted

The two components of cohort fitness point in **opposite directions**. Scrambled
agents survive about 4x worse and leave about 3.8x more offspring. At the end of
the 500-tick window the scrambled populations are *larger*: **211** intact
against a scrambled range of **267-389**.

So over 500 ticks, destroying the evolved structure **helps**. This is the same
direction the feeding result pointed last time (scrambled brains fed more), now
confirmed on reproduction as well. The evolved structure implements a slow,
conservative life history: feed less, breed less, live longer.

## Does the fast strategy hit a bill later? No — 10,000 ticks

If scrambled agents were simply spending down a reserve, a longer horizon should
punish them. `tools/run_horizon.py`, seed 1, 10,000 ticks past the lesion:

| arm | mean pop | final pop | offspring | hidden start -> end |
|---|---|---|---|---|
| intact | **173.9** | 253 | 2935 | 5.275 -> **3.028** |
| scrambled 0-4 | 176.4-207.8 | 58-217 | 3583-4547 | 5.275 -> 4.54-6.66 |

**Zero extinctions in any arm.** The intact mean population (173.9) sits just
below *all five* scrambled runs (176.4-207.8). Intact ends higher (253, above
four of five), but final population is a single noisy snapshot where mean
population is not.

The scrambled populations do not collapse. They sustain slightly larger average
populations while producing ~36% more offspring.

One unplanned observation: over those 10,000 ticks the **intact** population
sheds structure, 5.28 -> 3.03 hidden nodes, while the scrambled arms hold or
grow theirs. Selection is trimming the very structure this experiment set out to
test. Not interpreted here — it is a single seed and could be drift.

## Verdict

**The structure is real but not shown to be adaptive.** Three claims, at three
different confidence levels:

1. **Solid.** The lesion is exact and inert with no target — bit-identical
   output across ten replicates on a 0.10-hidden-node population.
2. **Solid within one population.** In seed 1, scrambling the evolved weights
   changes behaviour enormously: 4x worse individual survival, 3.8x more
   offspring, every one of ten replicates outside intact on both metrics. This
   structure is doing work. It is not decorative.
3. **Not established.** That work is not demonstrably *worth* anything. Over
   500 ticks the scrambled populations end larger; over 10,000 ticks none go
   extinct and their mean populations run slightly higher. The evolved
   structure buys individual longevity and pays for it in fecundity, and the
   trade does not come out ahead on any population measure tested.

And it is still one population. 60 fresh seeds produced nothing above 2.71
hidden nodes, so this cannot currently be replicated at all.

## Next

The comparison so far is between *separate worlds*, which is the weakest form of
the question — it compares one deterministic intact trajectory against five
scrambled ones and asks whether the numbers overlap. The decisive version is
**direct competition in a single world**: scramble half the population in place,
tag both groups, and let the lineages compete for the same food under the same
predators. Relative fitness is then read off which lineage's descendants take
over, with no cross-world comparison at all.

That needs a heritable lineage tag on `Agent` propagated through `_reproduce` —
about three lines of core instrumentation that does not touch the dynamics, and
deliberately not added here, since it is a new experiment rather than a step in
the one that was planned.

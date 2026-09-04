# Emergent Evolution: Design Spec

Date: 2026-09-04
Status: Approved, ready for implementation planning

## 1. Goal

Build an agent-based world in which **behavior, body plan, and ecological role are
discovered by evolution rather than authored by the programmer**.

The success criterion is falsifiable and stated up front:

> Starting from a population of near-empty random brains, the run must produce
> (a) measurable growth in brain complexity, (b) a diet-gene distribution that
> separates into distinct modes, and (c) at least one behavioral strategy that
> was never coded.

If a run does not produce these, the run failed. This is a claim we can check,
not a vibe we can admire.

### Explicit non-goals

- No fitness function anywhere in the codebase. This is a hard invariant, not a
  preference. Fitness is whether a lineage still exists.
- No hand-coded predators, prey, herding, or foraging rules.
- No labeled species. Species, if any, are an observed clustering, not a type.
- No genetic algorithm generation loop (evaluate-population, rank, select
  top-K, breed). Reproduction is continuous and local.

## 2. Core design decisions

| Decision | Choice | Why |
|---|---|---|
| What evolves | Neural brains (topology + weights) plus body genes | Highest ratio of visible surprise to build cost; ecological speciation and morphology tradeoffs come along for free |
| World substrate | Discrete toroidal grid, behind a `World` interface | Keeps the CA lineage, exact rules, fast; continuous world can swap in later without a rewrite |
| Genome | Growing topology, asexual reproduction | Complexity is unbounded; asexual means no genome alignment and no innovation numbers |
| Selection | Resource scarcity plus emergent predation | Nothing is labeled predator or prey; carnivory must be discovered |
| Observation | Headless core, CSV stats, matplotlib report, ASCII live viewer, HTML replay | Evidence over eyeballing; core stays fast and testable |

### Rejected alternatives, and why

- **Fixed-topology MLP.** Simpler and faster, but the complexity ceiling would be
  set by the programmer. That is the exact failure mode this project exists to avoid.
- **Avida-style self-replicating programs.** Most genuinely open-ended, but the
  output is instruction listings. Time would go to reading disassembly instead of
  watching behavior.
- **Scripted predators.** Produces fast, legible evasion pressure, but half the
  ecosystem would then be authored rather than evolved.
- **Sexual reproduction.** Would reintroduce innovation numbers and genome
  alignment, roughly doubling genome complexity, in exchange for faster adaptation
  and a real interbreeding species concept. Deliberately deferred; revisit only
  after the asexual engine is validated.

## 3. Module layout

```
evolution/
  genome.py      Genome: brain genes + body genes; mutate(), copy()
  brain.py       compile(Genome) -> Brain; Brain.step(inputs) -> outputs
  world.py       World interface + GridWorld (plant CA lives here)
  agent.py       Agent: genome, brain, energy, age, position
  sim.py         tick loop, birth/death bookkeeping, seeded RNG
  stats.py       sampled metrics -> stats.csv
  recorder.py    frame log -> frames.jsonl
  report.py      matplotlib panel figure from stats.csv
  viewers/
    ascii.py     live terminal view
    html.py      standalone replay/scrub page from frames.jsonl
tests/
```

Dependency direction is strictly one-way:
`genome -> brain -> agent -> sim -> {stats, recorder}`, and `sim -> world`.
Nothing in the core imports a viewer, matplotlib, or any I/O beyond writing its
own output files.

### The `World` interface

`sim.py` may only touch the world through:

```
wrap(x, y) -> (x, y)
neighbors(x, y) -> iterable of coords
occupant(x, y) -> Agent | None
move(agent, dx, dy) -> bool
free_adjacent(x, y) -> (x, y) | None
plant_at(x, y) -> float
meat_at(x, y) -> float
take_plant(x, y, amount) -> float
take_meat(x, y, amount) -> float
deposit_meat(x, y, amount) -> None
sense(agent) -> ndarray[20]
update() -> None
```

`GridWorld` implements it now. A future `ContinuousWorld` implements the same
interface; only `sense` and `move` change meaning. `sim.py` is not modified.

## 4. The plant layer is a cellular automaton

Food is not a uniform regrowth field. Plants spread to neighbors, producing
drifting patches:

```
for each cell c:
    n = count of 8-neighbors of c with plant > PLANT_SEED_MIN
    plant[c] += PLANT_GROWTH * (PLANT_SPONTANEOUS + PLANT_SPREAD * n)
    plant[c]  = min(plant[c], PLANT_CAP)
```

**Growth must be driven by neighbours, not by a flat rate.** An earlier
version of this formula read `PLANT_GROWTH * (1 + PLANT_SPREAD * n)`, which
gives every cell the full growth rate whether or not any plant is nearby; the
neighbour term was only a bonus. Under that rule an isolated empty cell fills to
the cap in 133 ticks, the equilibrium of the whole grid is a uniform lawn at
`PLANT_CAP`, and the layer is not a cellular automaton at all. The bug was
caught by `test_plants_spread_producing_patchiness`. With `PLANT_SPONTANEOUS`
small, an isolated cell instead needs ~13,000 ticks, so empty regions stay empty
and plants genuinely spread as a front.

**Rationale, because this looks like a cosmetic choice and is not:** with food
spread uniformly, random wandering is close to optimal, no navigation strategy
outperforms any other, and there is no selection gradient for the brain to climb.
Patchy food is what makes navigation, memory, and territory worth evolving. The
CA layer is the reason the brains have anything to think about.

Total world energy input per tick is bounded by `PLANT_CAP` and the growth rate.
This bound is the carrying capacity, and it is the only population control in the
system. There is no population cap.

## 5. Genome

### 5.1 Body genes

| Gene | Range | Effect |
|---|---|---|
| `diet` | 0.0 - 1.0 | 0 = pure herbivore, 1 = pure carnivore |
| `size` | 0.5 - 3.0 | attack success, move cost, corpse value, upkeep |
| `sense_range` | 1 - 6 | how far sensors reach (float, used as a distance scale) |
| `speed` | 0.5 - 2.0 | actions per tick, via an accumulator (see below) |
| `mutation_rate` | 0.01 - 0.5 | scales every mutation probability except its own (see 5.3) |

`mutation_rate` being itself heritable means evolvability can evolve. Expect it
to fall in stable environments and rise after a shock; this is a metric worth
plotting.

The diet tradeoff is enforced in the economy, not the genome:

```
plant_gain = eaten * (1 - diet) * PLANT_ENERGY
meat_gain  = eaten *      diet  * MEAT_ENERGY
```

No agent can be efficient at both. A mid-range `diet` is a genuinely worse
omnivore, not a free generalist, so the population is under pressure to
resolve toward one end or the other. That pressure is what can drive the
distribution bimodal.

### 5.2 Brain genes

- **Nodes:** `(id, kind, activation)` where `kind` is `input | hidden | output`
  and `activation` is one of `identity | tanh | relu | sin | gauss`.
  `identity` exists specifically so that `add_node` can be exactly
  function-preserving (see 5.3); without it, test 2 could only ever pass
  approximately.
- **Connections:** `(src_id, dst_id, weight, enabled)`.
- Recurrent and self connections are permitted, so memory is reachable.

Node IDs are unique within a single genome and are allocated from a per-genome
counter. Because reproduction is asexual, two genomes are never aligned or
crossed, so historical markings (NEAT innovation numbers) are unnecessary.

### 5.3 Mutation operators

All probabilities below are multiplied by that genome's own `mutation_rate` `m`.

| Operator | Probability | Effect |
|---|---|---|
| weight jitter | `0.8m` per connection | `w += N(0, 0.5)` |
| add link | `0.5m` | random src -> dst, if not already present |
| add node | `0.2m` | split an enabled connection |
| delete link | `0.3m` | remove a random connection |
| delete node | `0.1m` | remove a hidden node and its connections |
| toggle enable | `0.1m` | flip a connection's enabled flag |
| change activation | `0.1m` | reroll one hidden node's activation |
| body jitter | `m` per gene | `g += N(0, 0.1 * range)`, clamped to range |
| meta-mutation | `0.1` (unscaled) | `m *= lognormal(0, 0.1)`, clamped |

Body jitter applies to `diet`, `size`, `sense_range` and `speed` only.
`mutation_rate` is mutated exclusively by the meta-mutation row, at a rate that
is deliberately *not* scaled by itself. Scaling it by itself would create a
runaway feedback loop in both directions: a lineage that drifted to a high rate
would mutate its rate faster still.

**`add_node` must be function-preserving.** Splitting `A -> B` (weight `w`)
produces `A -> N` with weight 1.0, `N -> B` with weight `w`, and disables
`A -> B`. The new node's activation is `identity`, which makes the split exactly
function-preserving rather than merely approximately so. Later mutation may
change that activation, at which point the node starts doing real work.

This is not an optimization; it is load-bearing. If structural mutations changed
behavior on arrival, nearly every one would be immediately fatal, structure would
never be retained, and brain complexity would stay flat forever. Neutral arrival
lets structure accumulate first and become useful later.

**Function-preserving means at steady state, not tick-for-tick.** Because brain
evaluation is synchronous (6.1), inserting a node onto a path adds exactly one
tick of propagation delay, so the pre-split and post-split brains do *not* agree
on the transient. They agree once the signal has settled. Test 2 must therefore
hold a constant input for enough ticks to settle (50 is ample for an acyclic
genome) and compare the settled outputs. Comparing tick 1 against tick 1 would
fail for a correct implementation, and chasing that phantom failure would waste
a lot of time.

Input and output nodes are never deleted and never change activation.

## 6. Fixed I/O, growing middle

Sensors are a **fixed 20-element vector**, independent of `sense_range`:

```
each cell within Chebyshev distance <= sense_range is assigned to exactly one
of the 4 directions by the dominant axis of its offset from the agent
(|dx| > |dy| -> E or W; |dy| > |dx| -> N or S; ties on |dx| == |dy| go to N/S).
The agent's own cell is excluded. Then, for each direction:

    plant_density   summed plant in that direction, scaled by sense_range
    agent_distance  distance to nearest agent, normalized, 0 if none in range
    size_ratio      that agent's size / own size
    their_diet      that agent's diet gene
                                                     = 16 values
own energy (normalized), own age (normalized), own size, bias = 1.0
                                                     =  4 values
                                                     total 20
```

Outputs are a **fixed 8-element vector**:

```
0-4  move N, S, E, W, stay   -> argmax selects the action
5    eat        -> acts if > 0.5
6    attack     -> acts if > 0.5
7    reproduce  -> acts if > 0.5 and energy >= threshold
```

**Why fixed size matters:** if `sense_range` changed the *number* of inputs, then
mutating it would shift the index-to-meaning mapping of every connection in the
genome at once, silently scrambling the brain. `sense_range` therefore scales
sensor *reach* only, never sensor *count*.

Sensor slot 4 (`their_diet`) means an agent can perceive whether a neighbor is
carnivorous. This provides the raw signal from which predator avoidance can
evolve. The signal is provided; the behavior is not.

### Brain evaluation

Synchronous update. Every node computes its new value from the previous tick's
values of its inputs:

```
new[i] = activation_i( sum over enabled (j -> i) of weight * prev[j] )
```

Uniform, cycle-safe, and gives recurrence and memory for free. The cost is that a
signal takes N ticks to traverse N layers. This is accepted, and is arguably more
biologically honest than instantaneous propagation.

## 7. Economy

```
INCOME
  eat:     intake is BUDGETED by diet, not discounted after the fact:
             plant_taken = take_plant(cell, EAT_RATE * (1 - diet))
             meat_taken  = take_meat(cell,  EAT_RATE *      diet)
             energy += plant_taken * PLANT_ENERGY + meat_taken * MEAT_ENERGY
           Both forms encode the same tradeoff, but budgeting stops a pure
           carnivore from stripping a cell of plants it cannot digest and
           gaining nothing -- an invisible trampling effect that would distort
           the plant CA wherever carnivores walk.

OUTFLOW per tick
  basal        BASAL
  movement     MOVE_COST * size          (only on a tick the agent moves)
  upkeep       UPKEEP * size^2
  brain        BRAIN_COST * enabled_connection_count

DEATH
  energy <= 0            cause = starved
  killed by an attacker  cause = killed
  age > MAX_AGE          cause = age
  -> deposits MEAT_PER_SIZE * size of meat in its cell; meat decays MEAT_DECAY/tick

ATTACK
  target = the occupant of the cell named by the movement argmax (outputs 0-4).
    This reuses the existing direction outputs rather than adding new ones, and
    means an attack is always aimed. If that cell is empty or the argmax is
    'stay', the attack is a no-op and still costs ATTACK_COST.
  P(success) = size_self / (size_self + size_target)
  on success the target dies immediately (cause = killed)
  the killer gains NO energy directly. The victim becomes meat in its own cell,
    which someone must then choose to eat. Killing and feeding are therefore two
    separate behaviors that both have to evolve, and the killer is not guaranteed
    to be the one that benefits.

BIRTH
  requires energy >= REPRO_THRESHOLD * size, a free adjacent cell,
  and output 7 firing
  the child is placed in the first free cell found by scanning the 8 neighbors
    in an order shuffled by the seeded RNG, so placement carries no directional
    bias
  child gets half the parent's energy; parent keeps the other half
  child genome = mutated copy of parent's
```

**Brain cost is the single most important constant in the system.** Without a
per-tick charge proportional to enabled connections, `add_node` is free, genomes
accumulate junk structure indefinitely, and by late in a run the population
carries large brains that do nothing except slow the simulation down. Charging
rent every tick means structure survives only if it pays for itself. This is the
difference between evolution and accumulation.

## 8. Tick order

Deterministic given a seed. Agent order is shuffled each tick by the seeded RNG
so no agent has a permanent positional advantage.

1. World update: plant CA growth and spread, meat decay.
2. For each agent in shuffled order, accumulate `budget += speed`, then
   `while budget >= 1 and actions_this_tick < 2:` sense -> brain step ->
   act (move / eat / attack / reproduce), `budget -= 1`.
   So `speed` 0.5 acts every other tick, `speed` 1.0 every tick, and `speed` 2.0
   twice per tick. Each action is charged separately, so speed is not free: a
   fast agent pays MOVE_COST twice as often. The hard cap of 2 actions per tick
   matches the gene's upper bound.
3. Apply metabolic costs; increment age.
4. Resolve deaths; deposit corpses as meat.
5. Resolve queued births.
6. Every `STATS_INTERVAL` ticks, append a stats row.

Because there are no discrete generations, "generation" is reported as **mean
lineage depth** (a child's depth is its parent's plus one).

## 9. Parameters

Starting values. These are tuning knobs, expected to change; they live in one
`Config` dataclass, not scattered as literals.

```
GRID              128 x 128, toroidal
INITIAL_AGENTS    200
INITIAL_ENERGY    150

PLANT_CAP         20.0
PLANT_GROWTH      0.15
PLANT_SPREAD      0.125     per plant-bearing 8-neighbor
PLANT_SPONTANEOUS 0.01      growth with no neighbours; deliberately near-zero
PLANT_SEED_MIN    1.0

EAT_RATE          5.0
PLANT_ENERGY      1.0
MEAT_ENERGY       1.5       meat is denser than plants
MEAT_PER_SIZE     30.0
MEAT_DECAY        0.05

BASAL             0.5
MOVE_COST         0.2       x size
UPKEEP            0.1       x size^2
BRAIN_COST        0.02      x enabled connections
ATTACK_COST       2.0

REPRO_THRESHOLD   120.0     x size
MAX_AGE           2000

STATS_INTERVAL    50 ticks
DEFAULT_RUN       50,000 ticks
```

Initial genomes: all 20 inputs and 8 outputs present, no hidden nodes, and 10
random input-to-output connections with weights from `N(0, 1)`. Brains start
nearly empty on purpose. Any structure observed later was built by mutation.

## 10. Metrics written to stats.csv

- tick, population, births, deaths split by cause (starved / killed / age)
- brain: mean and max enabled connections, mean hidden node count
- body: mean and stdev of `diet`, `size`, `sense_range`, `speed`, `mutation_rate`
- `diet` histogram, 20 bins, plus a bimodality coefficient
- genetic diversity: mean pairwise body-gene distance over a random sample of 100
- energy budget: total plant, total meat, total agent energy
- mean lineage depth

The report figure plots these as a panel grid. The bimodality coefficient over
time is the primary evidence for the speciation claim; mean enabled connections
over time is the primary evidence for the complexity claim.

## 11. Testing strategy

Correctness tests:

1. **Mutation validity.** After any mutation sequence, a genome has no dangling
   connection endpoints, no duplicate connections, no input node as a `dst`, and
   all body genes within range.
2. **`add_node` preserves function.** For a random genome, outputs before and
   after an `add_node` mutation match within tolerance.
3. **Brain determinism.** Same genome and same input sequence gives identical
   outputs.
4. **World wrapping.** Movement across every edge wraps correctly; `neighbors`
   returns 8 distinct cells everywhere including corners.
5. **Energy accounting.** Track total system energy as
   `sum(plant) + sum(meat) + sum(agent energy)`. Over a run with births and
   deaths disabled, the tick-over-tick change in that total must equal
   `(plant grown) - (metabolism paid) - (meat decayed)` within floating point
   tolerance. Eating must show up as a transfer, not as a gain: an agent eating
   1.0 of plant with `diet = 0.5` gains 0.5 energy while the world loses 1.0, so
   the diet tradeoff is a real loss of energy from the system, not a rounding
   artifact. Nothing is created from nothing.
6. **Run reproducibility.** Two runs with the same seed produce byte-identical
   `stats.csv`.

Validation tests, which check that evolution actually works rather than that the
code merely runs:

7. **Null run.** Mutation rate forced to zero. Population statistics must stay
   flat apart from drift and extinction. This proves that any change seen in a
   normal run comes from mutation and not from a bug in the harness. Without this
   control, a trending chart proves nothing.
8. **Rigged world (gate).** Plants grow only in the eastern half of the grid
   (the world does not wrap for this test; the eastern edge is a wall). Within
   10,000 ticks, the mean agent x-position must exceed 0.65 * width, versus a
   mutation-off control run on the same seed which must stay near 0.5 * width.
   The test asserts on that numeric threshold, not on a chart looking right. If the engine cannot solve a
   problem this easy, it cannot solve a hard one, and no amount of watching an
   interesting-looking run would reveal that.

Test 8 is a **gate**: carnivory and the full economy are not enabled until it
passes.

## 12. Staged delivery

Each stage ends with passing tests.

| Stage | Deliverable |
|---|---|
| 0 | Repo scaffold, `Config`, test harness, seeded RNG |
| 1 | `genome.py` and mutation operators; tests 1 and 2 |
| 2 | `brain.py` compile and step; test 3 |
| 3 | `world.py`, GridWorld, plant CA, sensors; test 4 |
| 4 | `agent.py`, `sim.py`, economy, births and deaths; tests 5 and 6 |
| 5 | **Gate:** null run and rigged-world validation; tests 7 and 8 |
| 6 | `stats.py` and `report.py` panel figure |
| 7 | `viewers/ascii.py` live terminal view |
| 8 | Enable carnivory; long run; analyze for speciation |
| 9 | `viewers/html.py` replay and scrub page |

Stages 1 through 5 deliberately run **herbivores only**, with attack disabled.
The evolution engine is proved on the simpler world before predation is
introduced, so that when something misbehaves later there is only one new
variable to blame.

## 13. Known risks

| Risk | Signal | Response |
|---|---|---|
| Extinction in the first few hundred ticks | population hits 0 early | Raise `INITIAL_ENERGY`, lower `BASAL`, raise `PLANT_GROWTH`. A `--min-population` respawn flag exists for debugging and is **off by default**; it is a crutch that masks a real imbalance and must never be on for a run whose results are reported. |
| Population explosion to grid saturation | population approaches cell count | Expected to self-correct via the plant energy cap. If it does not, raise `REPRO_THRESHOLD`. Do not add a hard population cap; that would be an authored selection pressure. |
| Brain bloat | mean connections climbing while population is flat | Raise `BRAIN_COST`. |
| Genetic convergence to one genome | diversity metric collapsing | Spatial patchiness should prevent it. If not, this is a finding worth reporting, not necessarily a bug. |
| Complexity never grows | mean connections flat across a long run | Most likely `BRAIN_COST` too high, or `add_node` not actually function-preserving. Test 2 covers the second case. |

# Emergent Evolution Simulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a grid world in which neural-brained agents reproduce, mutate, and
are selected by resource scarcity alone, such that brain complexity and ecological
role are discovered rather than programmed.

**Architecture:** A headless, seeded simulation core (`genome -> brain -> agent ->
sim`, plus `sim -> world`) that writes CSV statistics and an optional frame log.
Viewers and plotting read those files and are never imported by the core. The
genome is a variable-topology neural network with asexual reproduction, so no
genome alignment or innovation numbers are needed. All selection is implicit:
there is no fitness function anywhere in the codebase.

**Tech Stack:** Python 3.13, numpy 2.2 (already installed), matplotlib 3.10
(already installed), pytest (must be installed in Task 1). No other dependencies.
No pygame.

**Spec:** `docs/superpowers/specs/2026-09-04-emergent-evolution-design.md`.
Read it before starting. Section references below (§5.3 etc.) point into it.

## Global Constraints

- **No fitness function.** No code may rank, sort, or score agents for the purpose
  of choosing who reproduces. Reproduction is a brain output. Violating this
  invalidates the entire project. A test enforces it (Task 17).
- **No hand-coded roles.** No `if agent.is_predator`, no species labels, no
  behavior lookup tables. Behavior comes only from brain outputs.
- **Determinism.** Every random draw goes through a single
  `numpy.random.Generator` created from `Config.seed`. No `random` module, no
  global `np.random`, no `Date`/time-dependent values. Same seed must give
  byte-identical `stats.csv`.
- **Core purity.** Nothing under `evolution/` except `report.py` and `viewers/`
  may import matplotlib, curses, or perform terminal output.
- **Fixed I/O.** `N_INPUTS = 20`, `N_OUTPUTS = 8`, permanently. Input node IDs are
  `0..19`, output node IDs are `20..27`. These never change, are never deleted,
  and their activations are never mutated (§5.3).
- **Node ID rule.** Node IDs are unique within a genome and allocated from a
  per-genome counter that only ever increases, including across deletions.
- **Python 3.13, numpy 2.2 API.** Use `rng.integers` (not `randint`),
  `rng.random` (not `random_sample`).
- **Style.** Type hints on all public functions. Dataclasses for records.
  No comments restating what code says; comments only for the non-obvious "why".

## File Structure

| File | Responsibility |
|---|---|
| `evolution/config.py` | `Config` dataclass. Every tunable number in the system lives here and nowhere else. |
| `evolution/genome.py` | Gene records, `random_genome`, `mutate`, `validate`. Knows nothing about worlds or agents. |
| `evolution/brain.py` | Compiles a `Genome` into an executable network. Pure numpy, no simulation concepts. |
| `evolution/world.py` | `World` protocol, `GridWorld`, plant CA, meat, movement, occupancy. |
| `evolution/sensing.py` | Reference and batched sensor implementations. Split from `world.py` because it is the performance-critical part and needs its own oracle test. |
| `evolution/agent.py` | `Agent` record and metabolic cost calculation. |
| `evolution/sim.py` | Tick loop, action dispatch, births, deaths. The only file that knows the tick order. |
| `evolution/stats.py` | Metric collection and CSV writing. |
| `evolution/recorder.py` | Frame log writing (JSONL). |
| `evolution/report.py` | matplotlib panel figure from `stats.csv`. Not imported by core. |
| `evolution/viewers/ascii.py` | Live terminal view. Not imported by core. |
| `evolution/viewers/html.py` | Standalone replay page from `frames.jsonl`. Not imported by core. |
| `run.py` | CLI entry point. |
| `tests/` | One test module per core module, plus `tests/test_validation.py` for the null-run and rigged-world gates. |

---

### Task 1: Project scaffold and Config

**Files:**
- Create: `evolution/__init__.py`, `evolution/config.py`, `pyproject.toml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `Config` frozen dataclass with the fields listed below;
  `Config.rng()` returning `numpy.random.Generator`.

- [ ] **Step 1: Install pytest**

```bash
python3 -m pip install pytest
python3 -m pytest --version
```

- [ ] **Step 2: Create the package and pyproject**

```bash
mkdir -p evolution/viewers tests
touch evolution/__init__.py evolution/viewers/__init__.py
```

`pyproject.toml`:

```toml
[project]
name = "evolution"
version = "0.1.0"
requires-python = ">=3.13"
dependencies = ["numpy>=2.2", "matplotlib>=3.10"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Write the failing test**

`tests/test_config.py`:

```python
import numpy as np
from evolution.config import Config


def test_defaults_match_spec():
    c = Config()
    assert (c.width, c.height) == (128, 128)
    assert c.plant_cap == 20.0
    assert c.brain_cost == 0.02
    assert c.max_age == 2000


def test_config_is_frozen():
    c = Config()
    try:
        c.width = 5
    except Exception:
        return
    raise AssertionError("Config must be frozen so a run cannot be retuned mid-flight")


def test_rng_is_deterministic():
    a = Config(seed=7).rng()
    b = Config(seed=7).rng()
    assert np.array_equal(a.random(10), b.random(10))


def test_different_seeds_differ():
    a = Config(seed=1).rng()
    b = Config(seed=2).rng()
    assert not np.array_equal(a.random(10), b.random(10))
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.config'`

- [ ] **Step 5: Implement Config**

`evolution/config.py`:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    # world
    width: int = 128
    height: int = 128
    wrap: bool = True

    # population seeding
    initial_agents: int = 200
    initial_energy: float = 150.0
    initial_links: int = 10
    initial_mutation_rate: float = 0.1

    # plant cellular automaton
    plant_cap: float = 20.0
    plant_growth: float = 0.15
    # Growth is driven by seeded NEIGHBOURS, not by a flat rate everywhere --
    # a flat rate saturates the whole grid and destroys the patchiness the
    # plant layer exists to create (spec 4).
    plant_spread: float = 0.125
    plant_spontaneous: float = 0.01
    plant_seed_min: float = 1.0

    # feeding
    eat_rate: float = 5.0
    plant_energy: float = 1.0
    meat_energy: float = 1.5
    meat_per_size: float = 30.0
    meat_decay: float = 0.05

    # metabolism
    basal: float = 0.5
    move_cost: float = 0.2
    upkeep: float = 0.1
    brain_cost: float = 0.02
    attack_cost: float = 2.0

    # life cycle
    repro_threshold: float = 120.0
    max_age: int = 2000

    # staging switches
    allow_attack: bool = False
    mutation_enabled: bool = True

    # bookkeeping
    stats_interval: int = 50
    seed: int = 0

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)
```

`allow_attack` defaults to `False` on purpose: Tasks 1-12 run a herbivores-only
world so the evolution engine can be validated before predation adds a second
moving part. Task 16 turns it on.

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml evolution tests
git commit -m "feat: project scaffold and Config"
```

---

### Task 2: Genome records and construction

**Files:**
- Create: `evolution/genome.py`
- Test: `tests/test_genome.py`

**Interfaces:**
- Consumes: `Config` from Task 1.
- Produces:
  - Constants `N_INPUTS = 20`, `N_OUTPUTS = 8`, `ACTIVATIONS`,
    `BODY_GENE_RANGES`, `MUTATION_RATE_RANGE`
  - `NodeGene(id: int, kind: str, activation: str)`
  - `ConnGene(src: int, dst: int, weight: float, enabled: bool)`
  - `Genome` with `.nodes`, `.conns`, `.body: dict[str, float]`,
    `.mutation_rate: float`, `.next_node_id: int`, and methods
    `.copy() -> Genome`, `.enabled_count() -> int`, `.hidden_count() -> int`
  - `random_genome(rng, cfg) -> Genome`
  - `validate(g: Genome) -> None` raising `AssertionError`

- [ ] **Step 1: Write the failing test**

`tests/test_genome.py`:

```python
import pytest
from evolution.config import Config
from evolution.genome import (
    N_INPUTS, N_OUTPUTS, Genome, random_genome, validate,
)


def test_random_genome_has_fixed_io():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    assert sum(n.kind == "input" for n in g.nodes) == N_INPUTS
    assert sum(n.kind == "output" for n in g.nodes) == N_OUTPUTS
    assert g.hidden_count() == 0


def test_input_ids_are_zero_through_nineteen():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    assert {n.id for n in g.nodes if n.kind == "input"} == set(range(N_INPUTS))
    assert {n.id for n in g.nodes if n.kind == "output"} == set(
        range(N_INPUTS, N_INPUTS + N_OUTPUTS)
    )


def test_inputs_are_identity_outputs_are_tanh():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    for n in g.nodes:
        if n.kind == "input":
            assert n.activation == "identity"
        if n.kind == "output":
            assert n.activation == "tanh"


def test_random_genome_is_valid():
    cfg = Config()
    validate(random_genome(cfg.rng(), cfg))


def test_copy_is_deep():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    h = g.copy()
    h.conns[0].weight = 999.0
    h.body["diet"] = 0.123
    assert g.conns[0].weight != 999.0
    assert g.body["diet"] != 0.123


def test_validate_rejects_dangling_connection():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    g.conns[0].src = 9999
    with pytest.raises(AssertionError):
        validate(g)


def test_validate_rejects_connection_into_input():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    g.conns[0].dst = 0
    with pytest.raises(AssertionError):
        validate(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_genome.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.genome'`

- [ ] **Step 3: Implement genome records**

`evolution/genome.py`:

```python
from dataclasses import dataclass, field

import numpy as np

from evolution.config import Config

N_INPUTS = 20
N_OUTPUTS = 8
ACTIVATIONS = ("identity", "tanh", "relu", "sin", "gauss")

BODY_GENE_RANGES: dict[str, tuple[float, float]] = {
    "diet": (0.0, 1.0),
    "size": (0.5, 3.0),
    "sense_range": (1.0, 6.0),
    "speed": (0.5, 2.0),
}
MUTATION_RATE_RANGE = (0.01, 0.5)


@dataclass
class NodeGene:
    id: int
    kind: str
    activation: str


@dataclass
class ConnGene:
    src: int
    dst: int
    weight: float
    enabled: bool = True


@dataclass
class Genome:
    nodes: list[NodeGene] = field(default_factory=list)
    conns: list[ConnGene] = field(default_factory=list)
    body: dict[str, float] = field(default_factory=dict)
    mutation_rate: float = 0.1
    next_node_id: int = N_INPUTS + N_OUTPUTS

    def copy(self) -> "Genome":
        return Genome(
            nodes=[NodeGene(n.id, n.kind, n.activation) for n in self.nodes],
            conns=[ConnGene(c.src, c.dst, c.weight, c.enabled) for c in self.conns],
            body=dict(self.body),
            mutation_rate=self.mutation_rate,
            next_node_id=self.next_node_id,
        )

    def enabled_count(self) -> int:
        return sum(1 for c in self.conns if c.enabled)

    def hidden_count(self) -> int:
        return sum(1 for n in self.nodes if n.kind == "hidden")


def random_genome(rng: np.random.Generator, cfg: Config) -> Genome:
    nodes = [NodeGene(i, "input", "identity") for i in range(N_INPUTS)]
    nodes += [
        NodeGene(N_INPUTS + j, "output", "tanh") for j in range(N_OUTPUTS)
    ]
    conns: list[ConnGene] = []
    seen: set[tuple[int, int]] = set()
    while len(conns) < cfg.initial_links:
        src = int(rng.integers(0, N_INPUTS))
        dst = N_INPUTS + int(rng.integers(0, N_OUTPUTS))
        if (src, dst) in seen:
            continue
        seen.add((src, dst))
        conns.append(ConnGene(src, dst, float(rng.normal(0.0, 1.0))))
    body = {
        name: float(rng.uniform(lo, hi))
        for name, (lo, hi) in BODY_GENE_RANGES.items()
    }
    return Genome(
        nodes=nodes,
        conns=conns,
        body=body,
        mutation_rate=cfg.initial_mutation_rate,
        next_node_id=N_INPUTS + N_OUTPUTS,
    )


def validate(g: Genome) -> None:
    ids = {n.id for n in g.nodes}
    assert len(ids) == len(g.nodes), "duplicate node ids"
    inputs = {n.id for n in g.nodes if n.kind == "input"}
    outputs = {n.id for n in g.nodes if n.kind == "output"}
    assert inputs == set(range(N_INPUTS)), "input node set corrupted"
    assert outputs == set(
        range(N_INPUTS, N_INPUTS + N_OUTPUTS)
    ), "output node set corrupted"
    for n in g.nodes:
        assert n.activation in ACTIVATIONS, f"unknown activation {n.activation}"
        if n.kind == "input":
            assert n.activation == "identity"
        if n.kind == "output":
            assert n.activation == "tanh"
    seen: set[tuple[int, int]] = set()
    for c in g.conns:
        assert c.src in ids, "dangling connection source"
        assert c.dst in ids, "dangling connection target"
        assert c.dst not in inputs, "connection into an input node"
        assert (c.src, c.dst) not in seen, "duplicate connection"
        seen.add((c.src, c.dst))
    for name, (lo, hi) in BODY_GENE_RANGES.items():
        assert name in g.body, f"missing body gene {name}"
        assert lo <= g.body[name] <= hi, f"body gene {name} out of range"
    lo, hi = MUTATION_RATE_RANGE
    assert lo <= g.mutation_rate <= hi, "mutation_rate out of range"
    assert g.next_node_id > max(ids), "next_node_id would collide"
```

Output nodes are `tanh`, so every output is in `[-1, 1]` and the `> 0.5`
thresholds in §6 are reachable. Input nodes are `identity` because they are
written directly from the sensor vector and must not be squashed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_genome.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/genome.py tests/test_genome.py
git commit -m "feat: genome records, construction, and validation"
```

---

### Task 3: Weight and body mutation

**Files:**
- Modify: `evolution/genome.py` (append)
- Test: `tests/test_mutation.py`

**Interfaces:**
- Consumes: everything from Task 2.
- Produces: `mutate(g: Genome, rng: np.random.Generator, cfg: Config) -> Genome`
  returning a NEW genome; the input is never modified.

- [ ] **Step 1: Write the failing test**

`tests/test_mutation.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.genome import (
    BODY_GENE_RANGES, MUTATION_RATE_RANGE, mutate, random_genome, validate,
)


def test_mutate_does_not_modify_the_parent():
    cfg = Config()
    rng = cfg.rng()
    parent = random_genome(rng, cfg)
    before = [c.weight for c in parent.conns]
    mutate(parent, rng, cfg)
    assert [c.weight for c in parent.conns] == before


def test_mutation_disabled_produces_an_exact_clone():
    cfg = Config(mutation_enabled=False)
    rng = cfg.rng()
    parent = random_genome(rng, cfg)
    child = mutate(parent, rng, cfg)
    assert [c.weight for c in child.conns] == [c.weight for c in parent.conns]
    assert child.body == parent.body
    assert child.mutation_rate == parent.mutation_rate


def test_body_genes_stay_in_range_over_many_generations():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(2000):
        g = mutate(g, rng, cfg)
    for name, (lo, hi) in BODY_GENE_RANGES.items():
        assert lo <= g.body[name] <= hi
    lo, hi = MUTATION_RATE_RANGE
    assert lo <= g.mutation_rate <= hi


def test_genome_stays_valid_over_many_generations():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(2000):
        g = mutate(g, rng, cfg)
        validate(g)


def test_weights_actually_change():
    """Compare summary statistics, not element-wise: structural mutation can
    add and remove connections, so the two arrays need not be the same length
    and an element-wise comparison would raise instead of failing cleanly."""
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    before = np.abs([c.weight for c in g.conns]).mean()
    for _ in range(50):
        g = mutate(g, rng, cfg)
    after = np.abs([c.weight for c in g.conns]).mean()
    assert abs(after - before) > 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_mutation.py -v`
Expected: FAIL with `ImportError: cannot import name 'mutate'`

- [ ] **Step 3: Implement mutate with weight and body operators only**

Append to `evolution/genome.py`. The structural helpers `_add_link`,
`_add_node`, `_del_link`, `_del_node`, `_toggle_enable`, `_change_activation`
are written in Task 4; for now define them as no-ops so this task's tests run
in isolation, and Task 4 replaces the bodies.

```python
def _add_link(g: Genome, rng: np.random.Generator) -> None:
    return


def _add_node(g: Genome, rng: np.random.Generator) -> None:
    return


def _del_link(g: Genome, rng: np.random.Generator) -> None:
    return


def _del_node(g: Genome, rng: np.random.Generator) -> None:
    return


def _toggle_enable(g: Genome, rng: np.random.Generator) -> None:
    return


def _change_activation(g: Genome, rng: np.random.Generator) -> None:
    return


def mutate(g: Genome, rng: np.random.Generator, cfg: Config) -> Genome:
    child = g.copy()
    if not cfg.mutation_enabled:
        return child

    m = child.mutation_rate

    for c in child.conns:
        if rng.random() < 0.8 * m:
            c.weight += float(rng.normal(0.0, 0.5))

    if rng.random() < 0.5 * m:
        _add_link(child, rng)
    if rng.random() < 0.2 * m:
        _add_node(child, rng)
    if rng.random() < 0.3 * m:
        _del_link(child, rng)
    if rng.random() < 0.1 * m:
        _del_node(child, rng)
    if rng.random() < 0.1 * m:
        _toggle_enable(child, rng)
    if rng.random() < 0.1 * m:
        _change_activation(child, rng)

    for name, (lo, hi) in BODY_GENE_RANGES.items():
        if rng.random() < m:
            jitter = float(rng.normal(0.0, 0.1 * (hi - lo)))
            child.body[name] = float(np.clip(child.body[name] + jitter, lo, hi))

    # Deliberately NOT scaled by m: a self-scaling meta-mutation rate is a
    # runaway feedback loop in both directions (spec 5.3).
    if rng.random() < 0.1:
        lo, hi = MUTATION_RATE_RANGE
        scaled = child.mutation_rate * float(np.exp(rng.normal(0.0, 0.1)))
        child.mutation_rate = float(np.clip(scaled, lo, hi))

    return child
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_mutation.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/genome.py tests/test_mutation.py
git commit -m "feat: weight and body-gene mutation"
```

---

### Task 4: Structural mutation

**Files:**
- Modify: `evolution/genome.py` (replace the six no-op helpers from Task 3)
- Test: `tests/test_structural_mutation.py`

**Interfaces:**
- Consumes: `Genome`, `NodeGene`, `ConnGene`, `ACTIVATIONS` from Task 2.
- Produces: working `_add_link`, `_add_node`, `_del_link`, `_del_node`,
  `_toggle_enable`, `_change_activation`. No new public names.

The `add_node` operator is the single most important function in this task.
Read §5.3 before implementing it.

- [ ] **Step 1: Write the failing test**

`tests/test_structural_mutation.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.genome import (
    N_INPUTS, N_OUTPUTS, _add_link, _add_node, _change_activation, _del_link,
    _del_node, random_genome, validate,
)


def test_add_node_increases_hidden_count_and_disables_the_split():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    before_enabled = g.enabled_count()
    _add_node(g, rng)
    validate(g)
    assert g.hidden_count() == 1
    # one connection disabled, two added: net +1 enabled
    assert g.enabled_count() == before_enabled + 1


def test_add_node_never_reuses_an_id_after_deletion():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    _add_node(g, rng)
    first_id = [n.id for n in g.nodes if n.kind == "hidden"][0]
    _del_node(g, rng)
    _add_node(g, rng)
    second_id = [n.id for n in g.nodes if n.kind == "hidden"][0]
    assert second_id != first_id
    validate(g)


def test_add_link_never_targets_an_input():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(200):
        _add_link(g, rng)
    validate(g)


def test_del_link_and_del_node_keep_the_genome_valid():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(20):
        _add_node(g, rng)
    for _ in range(50):
        _del_link(g, rng)
        _del_node(g, rng)
        validate(g)


def test_change_activation_never_touches_io_nodes():
    cfg = Config()
    rng = cfg.rng()
    g = random_genome(rng, cfg)
    for _ in range(100):
        _change_activation(g, rng)
    validate(g)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_structural_mutation.py -v`
Expected: FAIL — `test_add_node_increases_hidden_count_and_disables_the_split`
fails because the helper is still a no-op from Task 3.

- [ ] **Step 3: Replace the six no-op helpers**

```python
def _add_link(g: Genome, rng: np.random.Generator) -> None:
    existing = {(c.src, c.dst) for c in g.conns}
    sources = [n.id for n in g.nodes]
    targets = [n.id for n in g.nodes if n.kind != "input"]
    for _ in range(20):
        src = sources[int(rng.integers(len(sources)))]
        dst = targets[int(rng.integers(len(targets)))]
        if (src, dst) not in existing:
            g.conns.append(ConnGene(src, dst, float(rng.normal(0.0, 1.0))))
            return


def _add_node(g: Genome, rng: np.random.Generator) -> None:
    candidates = [c for c in g.conns if c.enabled]
    if not candidates:
        return
    old = candidates[int(rng.integers(len(candidates)))]
    new_id = g.next_node_id
    g.next_node_id += 1
    # identity activation and unit input weight make the split exactly
    # function-preserving at steady state (spec 5.3)
    g.nodes.append(NodeGene(new_id, "hidden", "identity"))
    old.enabled = False
    g.conns.append(ConnGene(old.src, new_id, 1.0))
    g.conns.append(ConnGene(new_id, old.dst, old.weight))


def _del_link(g: Genome, rng: np.random.Generator) -> None:
    if not g.conns:
        return
    del g.conns[int(rng.integers(len(g.conns)))]


def _del_node(g: Genome, rng: np.random.Generator) -> None:
    hidden = [n for n in g.nodes if n.kind == "hidden"]
    if not hidden:
        return
    victim = hidden[int(rng.integers(len(hidden)))]
    g.nodes.remove(victim)
    g.conns = [
        c for c in g.conns if c.src != victim.id and c.dst != victim.id
    ]


def _toggle_enable(g: Genome, rng: np.random.Generator) -> None:
    if not g.conns:
        return
    c = g.conns[int(rng.integers(len(g.conns)))]
    c.enabled = not c.enabled


def _change_activation(g: Genome, rng: np.random.Generator) -> None:
    hidden = [n for n in g.nodes if n.kind == "hidden"]
    if not hidden:
        return
    node = hidden[int(rng.integers(len(hidden)))]
    node.activation = ACTIVATIONS[int(rng.integers(len(ACTIVATIONS)))]
```

`_del_node` does not decrease `next_node_id`. That is what makes
`test_add_node_never_reuses_an_id_after_deletion` pass, and it prevents a
resurrected ID from silently inheriting connections that referenced the dead
node.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/ -v`
Expected: all pass, including the Task 3 suite which now exercises real
structural operators.

- [ ] **Step 5: Commit**

```bash
git add evolution/genome.py tests/test_structural_mutation.py
git commit -m "feat: structural mutation operators"
```

---

### Task 5: Brain compilation and evaluation

**Files:**
- Create: `evolution/brain.py`
- Test: `tests/test_brain.py`

**Interfaces:**
- Consumes: `Genome`, `N_INPUTS`, `N_OUTPUTS` from Task 2.
- Produces:
  - `ACT_FN: dict[str, Callable]`
  - `Brain(genome: Genome)` with `.step(inputs: np.ndarray) -> np.ndarray`
    mapping shape `(20,)` to shape `(8,)`, and `.reset() -> None`

- [ ] **Step 1: Write the failing test**

`tests/test_brain.py`:

```python
import numpy as np
from evolution.brain import Brain
from evolution.config import Config
from evolution.genome import N_INPUTS, N_OUTPUTS, _add_node, random_genome


def settle(brain: Brain, inputs: np.ndarray, ticks: int = 50) -> np.ndarray:
    out = None
    for _ in range(ticks):
        out = brain.step(inputs)
    return out


def test_step_shapes():
    cfg = Config()
    b = Brain(random_genome(cfg.rng(), cfg))
    out = b.step(np.zeros(N_INPUTS))
    assert out.shape == (N_OUTPUTS,)


def test_outputs_are_bounded_by_tanh():
    cfg = Config()
    b = Brain(random_genome(cfg.rng(), cfg))
    out = settle(b, np.full(N_INPUTS, 1000.0))
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_same_genome_same_inputs_same_outputs():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    assert np.array_equal(settle(Brain(g), x), settle(Brain(g), x))


def test_reset_clears_recurrent_state():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    b = Brain(g)
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    first = b.step(x)
    settle(b, x)
    b.reset()
    assert np.allclose(b.step(x), first)


def test_add_node_is_function_preserving_at_steady_state():
    """Spec 5.3. The synchronous update adds one tick of latency per inserted
    node, so pre- and post-split brains agree only after the signal settles.
    Comparing tick 1 to tick 1 would fail for a CORRECT implementation."""
    cfg = Config()
    rng = cfg.rng()
    x = np.linspace(-1.0, 1.0, N_INPUTS)
    for _ in range(20):
        g = random_genome(rng, cfg)
        before = settle(Brain(g), x)
        h = g.copy()
        _add_node(h, rng)
        after = settle(Brain(h), x)
        assert np.allclose(before, after, atol=1e-9), (
            "add_node changed behaviour; structural mutations must arrive neutral"
        )


def test_disabled_connections_do_not_contribute():
    cfg = Config()
    g = random_genome(cfg.rng(), cfg)
    x = np.ones(N_INPUTS)
    live = settle(Brain(g), x)
    for c in g.conns:
        c.enabled = False
    dead = settle(Brain(g), x)
    assert np.allclose(dead, 0.0)
    assert not np.allclose(live, dead)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_brain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.brain'`

- [ ] **Step 3: Implement Brain**

`evolution/brain.py`:

```python
from typing import Callable

import numpy as np

from evolution.genome import Genome

ACT_FN: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda x: x,
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
    "sin": np.sin,
    "gauss": lambda x: np.exp(-(x * x)),
}


class Brain:
    """A genome compiled into flat numpy arrays.

    Evaluation is synchronous: every node reads the PREVIOUS tick's values, so
    cycles need no special handling and recurrence gives memory for free. The
    cost is one tick of latency per layer (spec 6.1).
    """

    def __init__(self, genome: Genome) -> None:
        index = {n.id: i for i, n in enumerate(genome.nodes)}
        self.n = len(genome.nodes)

        self.input_idx = np.array(
            [index[n.id] for n in genome.nodes if n.kind == "input"], dtype=np.int64
        )
        self.output_idx = np.array(
            [index[n.id] for n in genome.nodes if n.kind == "output"], dtype=np.int64
        )

        live = [c for c in genome.conns if c.enabled]
        self.src = np.array([index[c.src] for c in live], dtype=np.int64)
        self.dst = np.array([index[c.dst] for c in live], dtype=np.int64)
        self.w = np.array([c.weight for c in live], dtype=np.float64)

        groups: dict[str, list[int]] = {}
        for i, node in enumerate(genome.nodes):
            groups.setdefault(node.activation, []).append(i)
        self.act_groups = {
            name: np.array(idxs, dtype=np.int64) for name, idxs in groups.items()
        }

        self.state = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        self.state[:] = 0.0

    def step(self, inputs: np.ndarray) -> np.ndarray:
        prev = self.state
        prev[self.input_idx] = inputs

        sums = np.zeros(self.n, dtype=np.float64)
        if self.src.size:
            np.add.at(sums, self.dst, self.w * prev[self.src])

        new = np.empty(self.n, dtype=np.float64)
        for name, idxs in self.act_groups.items():
            new[idxs] = ACT_FN[name](sums[idxs])
        new[self.input_idx] = inputs

        self.state = new
        return new[self.output_idx]
```

Inputs are re-stamped onto `new` after activation so that an input node always
holds its raw sensor value rather than a squashed one.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_brain.py -v`
Expected: 6 passed

If `test_add_node_is_function_preserving_at_steady_state` fails, the bug is
almost certainly in `_add_node` (Task 4) and not here: check that the
`src -> new` weight is exactly `1.0`, the `new -> dst` weight is the original
weight, the old connection is disabled, and the new node's activation is
`identity`.

- [ ] **Step 5: Commit**

```bash
git add evolution/brain.py tests/test_brain.py
git commit -m "feat: brain compilation and synchronous evaluation"
```

---

### Task 6: GridWorld, plant CA, meat, occupancy

**Files:**
- Create: `evolution/world.py`
- Test: `tests/test_world.py`

**Interfaces:**
- Consumes: `Config` from Task 1.
- Produces: `GridWorld(cfg, rng, growth_mask=None)` with
  - fields `plant: np.ndarray (h, w)`, `meat: np.ndarray (h, w)`,
    `occ_id: np.ndarray (h, w) int64` (-1 for empty),
    `occ_size: np.ndarray (h, w)`, `occ_diet: np.ndarray (h, w)`
  - `wrap(x, y) -> tuple[int, int]`
  - `is_free(x, y) -> bool`
  - `place(agent_id, x, y, size, diet) -> None`
  - `clear(x, y) -> None`
  - `move(agent_id, x, y, nx, ny, size, diet) -> bool`
  - `free_adjacent(x, y, rng) -> tuple[int, int] | None`
  - `take_plant(x, y, amount) -> float`
  - `take_meat(x, y, amount) -> float`
  - `deposit_meat(x, y, amount) -> None`
  - `update() -> None`
  - `total_plant() -> float`, `total_meat() -> float`

`growth_mask` is an optional boolean array of shape `(h, w)`; where it is
`False`, plants never grow. It exists for the rigged-world gate in Task 13 and
is `None` (grow everywhere) in normal runs.

- [ ] **Step 1: Write the failing test**

`tests/test_world.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.world import GridWorld


def make(**kw):
    cfg = Config(width=16, height=16, **kw)
    return GridWorld(cfg, cfg.rng())


def test_wrapping_is_toroidal():
    w = make()
    assert w.wrap(-1, -1) == (15, 15)
    assert w.wrap(16, 16) == (0, 0)


def test_no_wrapping_clamps_when_disabled():
    cfg = Config(width=16, height=16, wrap=False)
    w = GridWorld(cfg, cfg.rng())
    assert w.wrap(-1, 5) == (0, 5)
    assert w.wrap(16, 5) == (15, 5)


def test_plants_grow_and_are_capped():
    w = make()
    for _ in range(10000):
        w.update()
    assert w.plant.max() <= Config().plant_cap + 1e-9
    assert w.plant.min() > 0.0


def test_plants_spread_producing_patchiness():
    """Uniform food makes random wandering optimal and gives evolution no
    gradient to climb (spec 4). Patchiness is the point."""
    w = make()
    w.plant[:] = 0.0
    w.plant[8, 8] = Config().plant_cap
    for _ in range(200):
        w.update()
    assert w.plant.std() > 0.05 * w.plant.mean()


def test_meat_decays_toward_zero():
    w = make()
    w.deposit_meat(3, 3, 100.0)
    for _ in range(500):
        w.update()
    assert w.meat[3, 3] < 1.0


def test_take_plant_never_returns_more_than_present():
    w = make()
    w.plant[2, 2] = 3.0
    assert w.take_plant(2, 2, 10.0) == 3.0
    assert w.plant[2, 2] == 0.0


def test_occupancy_place_move_clear():
    w = make()
    w.place(7, 4, 4, size=2.0, diet=0.25)
    assert not w.is_free(4, 4)
    assert w.occ_id[4, 4] == 7
    assert w.occ_size[4, 4] == 2.0
    assert w.move(7, 4, 4, 5, 4, size=2.0, diet=0.25)
    assert w.is_free(4, 4) and not w.is_free(5, 4)
    w.clear(5, 4)
    assert w.is_free(5, 4)


def test_move_into_occupied_cell_fails():
    w = make()
    w.place(1, 4, 4, 1.0, 0.0)
    w.place(2, 5, 4, 1.0, 0.0)
    assert not w.move(1, 4, 4, 5, 4, 1.0, 0.0)
    assert w.occ_id[4, 4] == 1


def test_growth_mask_confines_growth():
    cfg = Config(width=16, height=16)
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, 8:] = True
    w = GridWorld(cfg, cfg.rng(), growth_mask=mask)
    w.plant[:] = 0.0
    for _ in range(500):
        w.update()
    assert w.plant[:, :8].sum() == 0.0
    assert w.plant[:, 8:].sum() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_world.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.world'`

- [ ] **Step 3: Implement GridWorld**

`evolution/world.py`:

```python
import numpy as np

from evolution.config import Config

NEIGHBOR_OFFSETS = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if not (dx == 0 and dy == 0)
]


class GridWorld:
    """Toroidal grid holding the plant CA, meat, and agent occupancy.

    Arrays are indexed [y, x] throughout. Occupancy is mirrored into parallel
    float grids (occ_size, occ_diet) so sensing can be done with vectorised
    array gathers instead of per-agent object lookups.
    """

    def __init__(
        self,
        cfg: Config,
        rng: np.random.Generator,
        growth_mask: np.ndarray | None = None,
    ) -> None:
        self.cfg = cfg
        self.rng = rng
        shape = (cfg.height, cfg.width)
        self.plant = rng.uniform(0.0, cfg.plant_cap, size=shape)
        self.meat = np.zeros(shape)
        self.occ_id = np.full(shape, -1, dtype=np.int64)
        self.occ_size = np.zeros(shape)
        self.occ_diet = np.zeros(shape)
        self.growth_mask = growth_mask
        if growth_mask is not None:
            self.plant *= growth_mask

    def wrap(self, x: int, y: int) -> tuple[int, int]:
        if self.cfg.wrap:
            return x % self.cfg.width, y % self.cfg.height
        return (
            min(max(x, 0), self.cfg.width - 1),
            min(max(y, 0), self.cfg.height - 1),
        )

    def is_free(self, x: int, y: int) -> bool:
        return bool(self.occ_id[y, x] == -1)

    def place(self, agent_id: int, x: int, y: int, size: float, diet: float) -> None:
        self.occ_id[y, x] = agent_id
        self.occ_size[y, x] = size
        self.occ_diet[y, x] = diet

    def clear(self, x: int, y: int) -> None:
        self.occ_id[y, x] = -1
        self.occ_size[y, x] = 0.0
        self.occ_diet[y, x] = 0.0

    def move(
        self, agent_id: int, x: int, y: int, nx: int, ny: int,
        size: float, diet: float,
    ) -> bool:
        nx, ny = self.wrap(nx, ny)
        if (nx, ny) == (x, y) or not self.is_free(nx, ny):
            return False
        self.clear(x, y)
        self.place(agent_id, nx, ny, size, diet)
        return True

    def free_adjacent(
        self, x: int, y: int, rng: np.random.Generator
    ) -> tuple[int, int] | None:
        order = rng.permutation(len(NEIGHBOR_OFFSETS))
        for i in order:
            dx, dy = NEIGHBOR_OFFSETS[i]
            nx, ny = self.wrap(x + dx, y + dy)
            if self.is_free(nx, ny):
                return nx, ny
        return None

    def take_plant(self, x: int, y: int, amount: float) -> float:
        taken = min(float(self.plant[y, x]), amount)
        self.plant[y, x] -= taken
        return taken

    def take_meat(self, x: int, y: int, amount: float) -> float:
        taken = min(float(self.meat[y, x]), amount)
        self.meat[y, x] -= taken
        return taken

    def deposit_meat(self, x: int, y: int, amount: float) -> None:
        self.meat[y, x] += amount

    def update(self) -> None:
        cfg = self.cfg
        seeded = (self.plant > cfg.plant_seed_min).astype(np.float64)
        neighbours = np.zeros_like(seeded)
        for dx, dy in NEIGHBOR_OFFSETS:
            neighbours += np.roll(np.roll(seeded, dy, axis=0), dx, axis=1)
        growth = cfg.plant_growth * (
            cfg.plant_spontaneous + cfg.plant_spread * neighbours
        )
        if self.growth_mask is not None:
            growth = growth * self.growth_mask
        np.minimum(self.plant + growth, cfg.plant_cap, out=self.plant)
        self.meat *= 1.0 - cfg.meat_decay

    def total_plant(self) -> float:
        return float(self.plant.sum())

    def total_meat(self) -> float:
        return float(self.meat.sum())
```

`np.roll` implements the 8-neighbour count toroidally in one vectorised pass.
For a non-wrapping world the roll wraps anyway, which only affects the growth
*bonus* at the edges by a negligible amount and never leaks plants across the
`growth_mask`, so the rigged-world gate is unaffected.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_world.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/world.py tests/test_world.py
git commit -m "feat: GridWorld with plant cellular automaton and occupancy"
```

---

### Task 7: Sensing, reference implementation

**Files:**
- Create: `evolution/sensing.py`
- Test: `tests/test_sensing.py`

**Interfaces:**
- Consumes: `GridWorld` from Task 6, `N_INPUTS` from Task 2.
- Produces: `sense_reference(world, x, y, size, sense_range, energy, age, cfg)
  -> np.ndarray` of shape `(20,)`.

This is the obviously-correct, slow version. Task 8 writes a fast version and
tests it against this one. Do not optimise anything here; its only job is to be
right.

Layout, per §6, in this exact order:

```
index  0..3    plant_density   for N, S, E, W
index  4..7    agent_distance  for N, S, E, W
index  8..11   size_ratio      for N, S, E, W
index 12..15   their_diet      for N, S, E, W
index 16       own energy / repro threshold at size 1
index 17       own age / max_age
index 18       own size / max size (3.0)
index 19       bias, always 1.0
```

Direction binning: a cell at offset `(dx, dy)` with Chebyshev distance
`<= int(sense_range)` and not `(0, 0)` belongs to `N` if `dy < 0 and |dy| >= |dx|`,
`S` if `dy > 0 and |dy| >= |dx|`, `E` if `dx > 0 and |dx| > |dy|`, `W` if
`dx < 0 and |dx| > |dy|`. Ties on `|dx| == |dy|` go to N/S, as the spec requires.

- [ ] **Step 1: Write the failing test**

`tests/test_sensing.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.sensing import DIRECTIONS, direction_of, sense_reference
from evolution.world import GridWorld


def make(**kw):
    cfg = Config(width=21, height=21, **kw)
    w = GridWorld(cfg, cfg.rng())
    w.plant[:] = 0.0
    return cfg, w


def test_vector_length_is_twenty():
    cfg, w = make()
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert v.shape == (20,)


def test_bias_is_one():
    cfg, w = make()
    assert sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[19] == 1.0


def test_direction_binning_ties_go_north_south():
    assert direction_of(1, -1) == DIRECTIONS.index("N")
    assert direction_of(-1, 1) == DIRECTIONS.index("S")
    assert direction_of(2, -1) == DIRECTIONS.index("E")
    assert direction_of(-2, 1) == DIRECTIONS.index("W")


def test_plant_to_the_east_lights_the_east_channel_only():
    cfg, w = make()
    w.plant[10, 13] = cfg.plant_cap
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    east = DIRECTIONS.index("E")
    assert v[east] > 0.0
    for d in range(4):
        if d != east:
            assert v[d] == 0.0


def test_plant_beyond_sense_range_is_invisible():
    cfg, w = make()
    w.plant[10, 15] = cfg.plant_cap
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert np.all(v[0:4] == 0.0)


def test_no_agent_in_range_gives_zero_distance_channel():
    cfg, w = make()
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    assert np.all(v[4:8] == 0.0)


def test_nearest_agent_reports_size_ratio_and_diet():
    cfg, w = make()
    w.place(1, 12, 10, size=2.0, diet=0.75)
    v = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)
    east = DIRECTIONS.index("E")
    assert v[4 + east] > 0.0
    assert np.isclose(v[8 + east], 2.0)
    assert np.isclose(v[12 + east], 0.75)


def test_closer_agent_reports_larger_proximity():
    cfg, w = make()
    east = DIRECTIONS.index("E")
    w.place(1, 13, 10, 1.0, 0.0)
    far = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[4 + east]
    w.clear(13, 10)
    w.place(2, 11, 10, 1.0, 0.0)
    near = sense_reference(w, 10, 10, 1.0, 3.0, 50.0, 0, cfg)[4 + east]
    assert near > far
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sensing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.sensing'`

- [ ] **Step 3: Implement the reference sensor**

`evolution/sensing.py`:

```python
import numpy as np

from evolution.config import Config
from evolution.genome import N_INPUTS
from evolution.world import GridWorld

DIRECTIONS = ("N", "S", "E", "W")
MAX_SIZE = 3.0


def direction_of(dx: int, dy: int) -> int:
    """Bin an offset into one of four directions. Ties go to N/S (spec 6)."""
    if abs(dy) >= abs(dx):
        return 0 if dy < 0 else 1
    return 2 if dx > 0 else 3


def sense_reference(
    world: GridWorld,
    x: int,
    y: int,
    size: float,
    sense_range: float,
    energy: float,
    age: int,
    cfg: Config,
) -> np.ndarray:
    """Slow, obviously-correct sensor. The oracle for the fast version."""
    v = np.zeros(N_INPUTS)
    r = int(sense_range)

    plant_sum = np.zeros(4)
    cell_count = np.zeros(4)
    best_dist = np.full(4, np.inf)
    best_size = np.zeros(4)
    best_diet = np.zeros(4)

    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            cx, cy = world.wrap(x + dx, y + dy)
            d = direction_of(dx, dy)
            plant_sum[d] += world.plant[cy, cx]
            cell_count[d] += 1.0
            if world.occ_id[cy, cx] != -1:
                dist = float(max(abs(dx), abs(dy)))
                if dist < best_dist[d]:
                    best_dist[d] = dist
                    best_size[d] = world.occ_size[cy, cx]
                    best_diet[d] = world.occ_diet[cy, cx]

    with np.errstate(invalid="ignore", divide="ignore"):
        v[0:4] = np.where(
            cell_count > 0, plant_sum / (cell_count * cfg.plant_cap), 0.0
        )
    seen = np.isfinite(best_dist)
    # proximity, not distance: 1.0 is adjacent, 0.0 is nothing in range
    v[4:8] = np.where(seen, 1.0 - best_dist / (r + 1.0), 0.0)
    v[8:12] = np.where(seen, best_size / max(size, 1e-9), 0.0)
    v[12:16] = np.where(seen, best_diet, 0.0)

    v[16] = energy / cfg.repro_threshold
    v[17] = age / cfg.max_age
    v[18] = size / MAX_SIZE
    v[19] = 1.0
    return v
```

Channel 4-7 is proximity rather than raw distance so that "nothing there" and
"something adjacent" are at opposite ends of the range. Encoding it as raw
distance would make an empty direction (0) indistinguishable from an adjacent
neighbour (0), which would be a silent, permanent blind spot.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sensing.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/sensing.py tests/test_sensing.py
git commit -m "feat: reference sensor implementation"
```

---

### Task 8: Batched sensing, verified against the reference

**Files:**
- Modify: `evolution/sensing.py` (append)
- Test: `tests/test_sensing_batched.py`

**Interfaces:**
- Consumes: `sense_reference`, `direction_of`, `DIRECTIONS`, `MAX_SIZE` from Task 7.
- Produces: `sense_batch(world, xs, ys, sizes, ranges, energies, ages, cfg)
  -> np.ndarray` of shape `(n_agents, 20)`, row `i` equal to
  `sense_reference` for agent `i`.

Why this task exists: per-agent Python sensing costs roughly a millisecond per
agent per tick once numpy call overhead is counted. At 1000 agents and 50,000
ticks that is over ten hours. Batching turns thousands of tiny gathers into one
gather per sense-radius group, of which there are at most six.

- [ ] **Step 1: Write the failing test**

`tests/test_sensing_batched.py`:

```python
import numpy as np
import pytest
from evolution.config import Config
from evolution.sensing import sense_batch, sense_reference
from evolution.world import GridWorld


def populated_world(seed: int):
    cfg = Config(width=32, height=32, seed=seed)
    rng = cfg.rng()
    w = GridWorld(cfg, rng)
    for i in range(60):
        x, y = int(rng.integers(32)), int(rng.integers(32))
        if w.is_free(x, y):
            w.place(i, x, y, float(rng.uniform(0.5, 3.0)), float(rng.uniform(0, 1)))
    return cfg, w, rng


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_batched_matches_reference_exactly(seed):
    cfg, w, rng = populated_world(seed)
    ys, xs = np.nonzero(w.occ_id != -1)
    sizes = w.occ_size[ys, xs]
    ranges = rng.uniform(1.0, 6.0, size=len(xs))
    energies = rng.uniform(0.0, 200.0, size=len(xs))
    ages = rng.integers(0, 2000, size=len(xs))

    fast = sense_batch(w, xs, ys, sizes, ranges, energies, ages, cfg)
    for i in range(len(xs)):
        slow = sense_reference(
            w, int(xs[i]), int(ys[i]), float(sizes[i]), float(ranges[i]),
            float(energies[i]), int(ages[i]), cfg,
        )
        assert np.allclose(fast[i], slow, atol=1e-12), f"row {i} diverged"


def test_batch_handles_empty_population():
    cfg = Config(width=8, height=8)
    w = GridWorld(cfg, cfg.rng())
    empty = np.array([], dtype=np.int64)
    out = sense_batch(
        w, empty, empty, np.array([]), np.array([]), np.array([]), empty, cfg
    )
    assert out.shape == (0, 20)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sensing_batched.py -v`
Expected: FAIL with `ImportError: cannot import name 'sense_batch'`

- [ ] **Step 3: Implement the batched sensor**

Append to `evolution/sensing.py`:

```python
_PATCH_CACHE: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _patch_masks(r: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For radius r return (dir_mask (4,s,s), dist (s,s), centre_mask (s,s))."""
    if r in _PATCH_CACHE:
        return _PATCH_CACHE[r]
    size = 2 * r + 1
    dir_mask = np.zeros((4, size, size), dtype=bool)
    dist = np.zeros((size, size))
    centre = np.zeros((size, size), dtype=bool)
    for iy, dy in enumerate(range(-r, r + 1)):
        for ix, dx in enumerate(range(-r, r + 1)):
            dist[iy, ix] = max(abs(dx), abs(dy))
            if dx == 0 and dy == 0:
                centre[iy, ix] = True
                continue
            dir_mask[direction_of(dx, dy), iy, ix] = True
    _PATCH_CACHE[r] = (dir_mask, dist, centre)
    return _PATCH_CACHE[r]


def sense_batch(
    world: GridWorld,
    xs: np.ndarray,
    ys: np.ndarray,
    sizes: np.ndarray,
    ranges: np.ndarray,
    energies: np.ndarray,
    ages: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    n = len(xs)
    out = np.zeros((n, N_INPUTS))
    if n == 0:
        return out

    radii = np.clip(ranges.astype(np.int64), 1, 6)
    for r in np.unique(radii):
        sel = np.nonzero(radii == r)[0]
        r = int(r)
        dir_mask, dist, _ = _patch_masks(r)
        offsets = np.arange(-r, r + 1)

        if cfg.wrap:
            py = (ys[sel, None] + offsets[None, :]) % cfg.height
            px = (xs[sel, None] + offsets[None, :]) % cfg.width
        else:
            py = np.clip(ys[sel, None] + offsets[None, :], 0, cfg.height - 1)
            px = np.clip(xs[sel, None] + offsets[None, :], 0, cfg.width - 1)

        rows = py[:, :, None]
        cols = px[:, None, :]
        plant = world.plant[rows, cols]
        occ = world.occ_id[rows, cols] != -1
        osize = world.occ_size[rows, cols]
        odiet = world.occ_diet[rows, cols]

        counts = dir_mask.sum(axis=(1, 2)).astype(np.float64)
        plant_sum = np.tensordot(plant, dir_mask, axes=([1, 2], [1, 2]))
        out[np.ix_(sel, np.arange(0, 4))] = plant_sum / (counts * cfg.plant_cap)

        flat_dist = dist.reshape(-1)
        for d in range(4):
            visible = occ & dir_mask[d]
            masked = np.where(visible, dist[None, :, :], np.inf).reshape(len(sel), -1)
            best = np.argmin(masked, axis=1)
            found = np.isfinite(masked[np.arange(len(sel)), best])
            bd = flat_dist[best]
            out[sel, 4 + d] = np.where(found, 1.0 - bd / (r + 1.0), 0.0)
            fs = osize.reshape(len(sel), -1)[np.arange(len(sel)), best]
            fd = odiet.reshape(len(sel), -1)[np.arange(len(sel)), best]
            out[sel, 8 + d] = np.where(found, fs / np.maximum(sizes[sel], 1e-9), 0.0)
            out[sel, 12 + d] = np.where(found, fd, 0.0)

    out[:, 16] = energies / cfg.repro_threshold
    out[:, 17] = ages / cfg.max_age
    out[:, 18] = sizes / MAX_SIZE
    out[:, 19] = 1.0
    return out
```

Note the deliberate ordering trap avoided here: `np.argmin` over an all-`inf`
row returns index 0, which is a real cell. The `found` mask is what prevents
that from being reported as a sighting. Removing it would make every agent
believe there is a neighbour due north at all times.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_sensing_batched.py -v`
Expected: 6 passed

- [ ] **Step 5: Benchmark and record the number**

```bash
python3 - <<'BENCH'
import time
import numpy as np
from evolution.config import Config
from evolution.sensing import sense_batch
from evolution.world import GridWorld

cfg = Config(seed=0)
rng = cfg.rng()
w = GridWorld(cfg, rng)
n = 1000
xs = rng.integers(0, cfg.width, n)
ys = rng.integers(0, cfg.height, n)
for i in range(n):
    w.place(i, int(xs[i]), int(ys[i]), 1.0, 0.5)
sizes = np.ones(n)
ranges = rng.uniform(1, 6, n)
en = np.full(n, 50.0)
ages = np.zeros(n, dtype=np.int64)

sense_batch(w, xs, ys, sizes, ranges, en, ages, cfg)
t = time.perf_counter()
for _ in range(20):
    sense_batch(w, xs, ys, sizes, ranges, en, ages, cfg)
ms = (time.perf_counter() - t) / 20 * 1000
print(f"{ms:.2f} ms per tick for {n} agents -> {1000/ms:.0f} sense-ticks/sec")
BENCH
```

Expected: under 20 ms per tick. If it is above 50 ms, a 50,000-tick run will
take over 40 minutes of sensing alone; stop and reduce `width`/`height` or the
target population in `Config` before continuing, and record the decision in the
commit message.

- [ ] **Step 6: Commit**

```bash
git add evolution/sensing.py tests/test_sensing_batched.py
git commit -m "perf: batched sensing verified against the reference implementation"
```

---

### Task 9: Agent record

**Files:**
- Create: `evolution/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `Genome` (Task 2), `Brain` (Task 5), `Config` (Task 1).
- Produces: `Agent` dataclass with fields `id, genome, brain, x, y, energy, age,
  budget, depth, alive, moved, brain_links, size, diet, speed, sense_range`, and
  `metabolic_cost(agent, cfg) -> float`.

Body-gene values are copied onto the agent at construction. They are read every
tick and the genome dict lookup is measurably slower than an attribute read.
They are never mutated in place; a mutation produces a new genome and therefore a
new agent.

- [ ] **Step 1: Write the failing test**

`tests/test_agent.py`:

```python
from evolution.agent import Agent, metabolic_cost
from evolution.brain import Brain
from evolution.config import Config
from evolution.genome import random_genome


def make_agent(cfg, **kw):
    g = random_genome(cfg.rng(), cfg)
    return Agent.create(0, g, x=1, y=1, energy=100.0, depth=0)


def test_agent_caches_body_genes():
    cfg = Config()
    a = make_agent(cfg)
    assert a.size == a.genome.body["size"]
    assert a.diet == a.genome.body["diet"]
    assert a.brain_links == a.genome.enabled_count()


def test_metabolic_cost_includes_brain_rent():
    """Brain cost is what forces complexity to pay for itself (spec 7)."""
    cfg = Config()
    a = make_agent(cfg)
    base = metabolic_cost(a, cfg, moved=False)
    a.brain_links += 100
    assert metabolic_cost(a, cfg, moved=False) == base + 100 * cfg.brain_cost


def test_moving_costs_more_than_standing_still():
    cfg = Config()
    a = make_agent(cfg)
    assert metabolic_cost(a, cfg, moved=True) > metabolic_cost(a, cfg, moved=False)


def test_upkeep_scales_with_size_squared():
    cfg = Config(basal=0.0, brain_cost=0.0)
    a = make_agent(cfg)
    a.size = 2.0
    two = metabolic_cost(a, cfg, moved=False)
    a.size = 4.0
    assert abs(metabolic_cost(a, cfg, moved=False) - 4.0 * two) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_agent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.agent'`

- [ ] **Step 3: Implement Agent**

`evolution/agent.py`:

```python
from dataclasses import dataclass

from evolution.brain import Brain
from evolution.config import Config
from evolution.genome import Genome


@dataclass
class Agent:
    id: int
    genome: Genome
    brain: Brain
    x: int
    y: int
    energy: float
    age: int = 0
    budget: float = 0.0
    depth: int = 0
    alive: bool = True
    moved: bool = False
    brain_links: int = 0
    size: float = 1.0
    diet: float = 0.0
    speed: float = 1.0
    sense_range: float = 1.0

    @classmethod
    def create(
        cls, agent_id: int, genome: Genome, x: int, y: int,
        energy: float, depth: int,
    ) -> "Agent":
        return cls(
            id=agent_id,
            genome=genome,
            brain=Brain(genome),
            x=x,
            y=y,
            energy=energy,
            depth=depth,
            brain_links=genome.enabled_count(),
            size=genome.body["size"],
            diet=genome.body["diet"],
            speed=genome.body["speed"],
            sense_range=genome.body["sense_range"],
        )


def metabolic_cost(a: Agent, cfg: Config, moved: bool) -> float:
    cost = (
        cfg.basal
        + cfg.upkeep * a.size * a.size
        + cfg.brain_cost * a.brain_links
    )
    if moved:
        cost += cfg.move_cost * a.size
    return cost
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_agent.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/agent.py tests/test_agent.py
git commit -m "feat: agent record and metabolic cost"
```

---

### Task 10: Simulation tick loop

**Files:**
- Create: `evolution/sim.py`
- Test: `tests/test_sim.py`

**Interfaces:**
- Consumes: everything from Tasks 1-9.
- Produces: `Simulation(cfg, growth_mask=None)` with
  - `.agents: list[Agent]`, `.world: GridWorld`, `.tick_count: int`
  - `.births: int`, `.deaths: dict[str, int]`
  - `.ledger: dict[str, float]` reset each tick, keys
    `grown`, `metabolism`, `decayed`, `conversion`
  - `.tick() -> None`, `.run(ticks: int, on_stats=None) -> None`
  - `.total_energy() -> float`

Three ordering decisions, all deliberate and all load-bearing:

1. **Sensing happens once per tick for the whole population, before anyone acts.**
   Every agent perceives the same world snapshot. This is what makes batched
   sensing (Task 8) valid, and it removes the unfair advantage that would
   otherwise go to whoever the shuffle put first.
2. **A fast agent acts twice on one perception.** Re-sensing mid-tick would
   destroy the batching. Speed buys extra actions, not extra eyes.
3. **Within one action the order is eat, attack, reproduce, move.** Fixed and
   documented so behaviour is reproducible; the brain still chooses *whether* to
   do each.

- [ ] **Step 1: Write the failing test**

`tests/test_sim.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.sim import Simulation


def test_initial_population_is_placed():
    cfg = Config(width=32, height=32, initial_agents=50)
    sim = Simulation(cfg)
    assert len(sim.agents) == 50
    assert int((sim.world.occ_id != -1).sum()) == 50


def test_tick_advances_and_ages_agents():
    cfg = Config(width=32, height=32, initial_agents=20)
    sim = Simulation(cfg)
    sim.tick()
    assert sim.tick_count == 1
    assert all(a.age == 1 for a in sim.agents)


def test_agents_starve_without_food():
    cfg = Config(
        width=32, height=32, initial_agents=20,
        initial_energy=5.0, plant_growth=0.0, plant_cap=0.0,
    )
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(200)
    assert len(sim.agents) == 0
    assert sim.deaths["starved"] > 0


def test_agents_die_of_old_age():
    cfg = Config(width=32, height=32, initial_agents=20, max_age=10,
                 initial_energy=1e6)
    sim = Simulation(cfg)
    sim.run(30)
    assert sim.deaths["age"] > 0


def test_occupancy_stays_consistent_with_the_agent_list():
    cfg = Config(width=32, height=32, initial_agents=40, seed=3)
    sim = Simulation(cfg)
    sim.run(300)
    live_ids = {a.id for a in sim.agents}
    grid_ids = set(sim.world.occ_id[sim.world.occ_id != -1].tolist())
    assert live_ids == grid_ids, "grid and agent list disagree about who exists"


def test_no_two_agents_share_a_cell():
    cfg = Config(width=32, height=32, initial_agents=60, seed=5)
    sim = Simulation(cfg)
    sim.run(300)
    coords = [(a.x, a.y) for a in sim.agents]
    assert len(coords) == len(set(coords))


def test_dead_agents_leave_meat():
    cfg = Config(width=16, height=16, initial_agents=10,
                 initial_energy=2.0, plant_growth=0.0)
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(50)
    assert sim.world.total_meat() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_sim.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.sim'`

- [ ] **Step 3: Implement Simulation**

`evolution/sim.py`:

```python
from typing import Callable

import numpy as np

from evolution.agent import Agent, metabolic_cost
from evolution.config import Config
from evolution.genome import Genome, mutate, random_genome
from evolution.sensing import sense_batch
from evolution.world import GridWorld

# N, S, E, W, stay -- must match the output layout in spec 6
DELTAS = ((0, -1), (0, 1), (1, 0), (-1, 0), (0, 0))
RESERVED = -2


class Simulation:
    def __init__(self, cfg: Config, growth_mask: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.rng = cfg.rng()
        self.world = GridWorld(cfg, self.rng, growth_mask)
        self.agents: list[Agent] = []
        self.by_id: dict[int, Agent] = {}
        self.tick_count = 0
        self.next_id = 0
        self.births = 0
        self.deaths = {"starved": 0, "killed": 0, "age": 0}
        self.ledger = {"grown": 0.0, "metabolism": 0.0, "decayed": 0.0,
                       "conversion": 0.0}
        self._seed_population()

    def _seed_population(self) -> None:
        placed = 0
        guard = 0
        while placed < self.cfg.initial_agents and guard < 10 ** 6:
            guard += 1
            x = int(self.rng.integers(self.cfg.width))
            y = int(self.rng.integers(self.cfg.height))
            if not self.world.is_free(x, y):
                continue
            self._spawn(random_genome(self.rng, self.cfg), x, y,
                        self.cfg.initial_energy, 0)
            placed += 1

    def _spawn(self, genome: Genome, x: int, y: int,
               energy: float, depth: int) -> Agent:
        a = Agent.create(self.next_id, genome, x, y, energy, depth)
        self.next_id += 1
        self.agents.append(a)
        self.by_id[a.id] = a
        self.world.place(a.id, x, y, a.size, a.diet)
        return a

    def total_energy(self) -> float:
        return (
            self.world.total_plant()
            + self.world.total_meat()
            + sum(a.energy for a in self.agents)
        )

    def tick(self) -> None:
        cfg = self.cfg
        for k in self.ledger:
            self.ledger[k] = 0.0

        before_plant = self.world.total_plant()
        before_meat = self.world.total_meat()
        self.world.update()
        self.ledger["grown"] = self.world.total_plant() - before_plant
        self.ledger["decayed"] = before_meat - self.world.total_meat()

        living = self.agents
        if living:
            senses = sense_batch(
                self.world,
                np.array([a.x for a in living]),
                np.array([a.y for a in living]),
                np.array([a.size for a in living]),
                np.array([a.sense_range for a in living]),
                np.array([a.energy for a in living]),
                np.array([a.age for a in living]),
                cfg,
            )
        births: list[tuple[Genome, int, int, float, int]] = []

        for i in self.rng.permutation(len(living)):
            a = living[i]
            if not a.alive:
                continue
            a.moved = False
            a.budget += a.speed
            actions = 0
            while a.budget >= 1.0 and actions < 2 and a.alive:
                a.budget -= 1.0
                actions += 1
                self._act(a, a.brain.step(senses[i]), births)

            cost = metabolic_cost(a, cfg, a.moved)
            a.energy -= cost
            self.ledger["metabolism"] += cost
            a.age += 1

            if a.energy <= 0.0:
                self._kill(a, "starved")
            elif a.age > cfg.max_age:
                self._kill(a, "age")

        for genome, x, y, energy, depth in births:
            self.world.clear(x, y)
            self._spawn(genome, x, y, energy, depth)
            self.births += 1

        self.agents = [a for a in self.agents if a.alive]
        self.tick_count += 1

    def _act(self, a: Agent, out: np.ndarray,
             births: list[tuple[Genome, int, int, float, int]]) -> None:
        cfg = self.cfg
        dx, dy = DELTAS[int(np.argmax(out[0:5]))]
        if out[5] > 0.5:
            self._eat(a)
        if cfg.allow_attack and out[6] > 0.5:
            self._attack(a, dx, dy)
        if out[7] > 0.5:
            self._reproduce(a, births)
        if (dx or dy) and a.alive:
            nx, ny = self.world.wrap(a.x + dx, a.y + dy)
            if self.world.move(a.id, a.x, a.y, nx, ny, a.size, a.diet):
                a.x, a.y = nx, ny
                a.moved = True

    def _eat(self, a: Agent) -> None:
        cfg = self.cfg
        plant = self.world.take_plant(a.x, a.y, cfg.eat_rate * (1.0 - a.diet))
        meat = self.world.take_meat(a.x, a.y, cfg.eat_rate * a.diet)
        gained = plant * cfg.plant_energy + meat * cfg.meat_energy
        a.energy += gained
        self.ledger["conversion"] += meat * (cfg.meat_energy - 1.0)

    def _attack(self, a: Agent, dx: int, dy: int) -> None:
        cfg = self.cfg
        a.energy -= cfg.attack_cost
        self.ledger["metabolism"] += cfg.attack_cost
        if dx == 0 and dy == 0:
            return
        tx, ty = self.world.wrap(a.x + dx, a.y + dy)
        target = self.by_id.get(int(self.world.occ_id[ty, tx]))
        if target is None or not target.alive:
            return
        if self.rng.random() < a.size / (a.size + target.size):
            self._kill(target, "killed")

    def _reproduce(self, a: Agent,
                   births: list[tuple[Genome, int, int, float, int]]) -> None:
        cfg = self.cfg
        if a.energy < cfg.repro_threshold * a.size:
            return
        spot = self.world.free_adjacent(a.x, a.y, self.rng)
        if spot is None:
            return
        child_energy = a.energy / 2.0
        a.energy -= child_energy
        # reserve so two parents cannot target the same cell this tick
        self.world.place(RESERVED, spot[0], spot[1], 0.0, 0.0)
        births.append((mutate(a.genome, self.rng, cfg),
                       spot[0], spot[1], child_energy, a.depth + 1))

    def _kill(self, a: Agent, cause: str) -> None:
        if not a.alive:
            return
        a.alive = False
        self.deaths[cause] += 1
        self.world.clear(a.x, a.y)
        self.world.deposit_meat(a.x, a.y, self.cfg.meat_per_size * a.size)
        self.by_id.pop(a.id, None)

    def run(self, ticks: int,
            on_stats: Callable[["Simulation"], None] | None = None) -> None:
        for _ in range(ticks):
            self.tick()
            if on_stats and self.tick_count % self.cfg.stats_interval == 0:
                on_stats(self)
            if not self.agents:
                break
```

Note the eating rule: intake is budgeted by diet
(`eat_rate * (1 - diet)` of plant, `eat_rate * diet` of meat) rather than taking
a full ration of each and then discounting the gain. Both encode the same
tradeoff, but the budgeted form prevents a pure carnivore from stripping a cell
of plants it cannot digest. Spec §7 is updated to match.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add evolution/sim.py tests/test_sim.py
git commit -m "feat: simulation tick loop with births, deaths, and ledger"
```

---

### Task 11: Determinism and energy accounting

**Files:**
- Test: `tests/test_invariants.py`

**Interfaces:**
- Consumes: `Simulation` from Task 10. Produces no new code — this task adds the
  two invariant tests from §11 and fixes whatever they catch.

- [ ] **Step 1: Write the failing test**

`tests/test_invariants.py`:

```python
import numpy as np
from evolution.config import Config
from evolution.sim import Simulation


def fingerprint(sim: Simulation) -> tuple:
    return (
        len(sim.agents),
        sim.births,
        tuple(sorted(sim.deaths.items())),
        round(sim.world.total_plant(), 6),
        round(sum(a.energy for a in sim.agents), 6),
        tuple(sorted((a.x, a.y) for a in sim.agents)),
    )


def test_same_seed_gives_identical_runs():
    a = Simulation(Config(width=48, height=48, initial_agents=60, seed=11))
    b = Simulation(Config(width=48, height=48, initial_agents=60, seed=11))
    a.run(400)
    b.run(400)
    assert fingerprint(a) == fingerprint(b)


def test_different_seeds_diverge():
    a = Simulation(Config(width=48, height=48, initial_agents=60, seed=11))
    b = Simulation(Config(width=48, height=48, initial_agents=60, seed=12))
    a.run(400)
    b.run(400)
    assert fingerprint(a) != fingerprint(b)


def test_energy_is_conserved_against_the_ledger():
    """Nothing may be created from nothing. With births and deaths switched off,
    every unit of energy in the system must be explained by the ledger."""
    cfg = Config(
        width=48, height=48, initial_agents=60, seed=2,
        initial_energy=1e6, repro_threshold=1e12, max_age=10 ** 9,
        allow_attack=False,  # pinned: this test asserts nobody dies
    )
    sim = Simulation(cfg)
    for _ in range(300):
        before = sim.total_energy()
        sim.tick()
        after = sim.total_energy()
        expected = (
            before
            + sim.ledger["grown"]
            - sim.ledger["metabolism"]
            - sim.ledger["decayed"]
            + sim.ledger["conversion"]
        )
        assert abs(after - expected) < 1e-6, (
            f"unexplained energy at tick {sim.tick_count}: "
            f"{after - expected:+.9f}"
        )
        assert len(sim.agents) == 60, "no agent should have died in this setup"
```

- [ ] **Step 2: Run test to verify it fails or passes**

Run: `python3 -m pytest tests/test_invariants.py -v`
Expected: all three pass if Tasks 1-10 are correct. If
`test_energy_is_conserved_against_the_ledger` fails, the ledger is missing a
flow — the usual culprits are the plant cap silently discarding growth (count
only the growth that actually landed, which `ledger["grown"]` already does by
measuring the total before and after) and `attack_cost` not being recorded.

- [ ] **Step 3: Fix any failure, then re-run**

Run: `python3 -m pytest tests/ -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_invariants.py
git commit -m "test: determinism and energy conservation invariants"
```

---

### Task 12: Statistics and the CLI

**Files:**
- Create: `evolution/stats.py`, `run.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `Simulation` from Task 10.
- Produces:
  - `collect(sim) -> dict[str, float]`
  - `StatsWriter(path)` with `.write(row: dict) -> None` and `.close() -> None`
  - `bimodality(values: np.ndarray) -> float` (Sarle's coefficient)
  - `run.py` CLI: `--ticks --seed --out --attack --no-mutation --width --height`

Sarle's bimodality coefficient is `(skew^2 + 1) / kurtosis`. Values above about
`0.555` indicate a bimodal distribution; the uniform distribution sits at `1.0`
and a normal distribution at about `0.33`. This single number is the primary
evidence for the speciation claim in §1, so it must be computed, not eyeballed.

- [ ] **Step 1: Write the failing test**

`tests/test_stats.py`:

```python
import csv

import numpy as np
from evolution.config import Config
from evolution.sim import Simulation
from evolution.stats import StatsWriter, bimodality, collect


def test_bimodality_separates_unimodal_from_bimodal():
    rng = np.random.default_rng(0)
    unimodal = rng.normal(0.5, 0.1, 5000)
    bimodal = np.concatenate([rng.normal(0.1, 0.03, 2500),
                              rng.normal(0.9, 0.03, 2500)])
    assert bimodality(unimodal) < 0.555
    assert bimodality(bimodal) > 0.555


def test_collect_returns_the_documented_keys():
    sim = Simulation(Config(width=32, height=32, initial_agents=30))
    sim.run(20)
    row = collect(sim)
    for key in (
        "tick", "population", "births", "deaths_starved", "deaths_killed",
        "deaths_age", "mean_links", "max_links", "mean_hidden", "mean_diet",
        "std_diet", "diet_bimodality", "mean_size", "mean_sense_range",
        "mean_speed", "mean_mutation_rate", "diversity", "total_plant",
        "total_meat", "total_agent_energy", "mean_depth",
    ):
        assert key in row, f"missing metric {key}"


def test_collect_on_an_empty_population_does_not_crash():
    cfg = Config(width=16, height=16, initial_agents=4,
                 initial_energy=1.0, plant_growth=0.0)
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(100)
    assert collect(sim)["population"] == 0


def test_writer_round_trips(tmp_path):
    path = tmp_path / "stats.csv"
    w = StatsWriter(path)
    w.write({"tick": 1, "population": 5})
    w.write({"tick": 2, "population": 7})
    w.close()
    rows = list(csv.DictReader(path.open()))
    assert [r["population"] for r in rows] == ["5", "7"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.stats'`

- [ ] **Step 3: Implement stats**

`evolution/stats.py`:

```python
import csv
from pathlib import Path

import numpy as np

from evolution.sim import Simulation


def bimodality(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient. Above ~0.555 suggests two modes."""
    n = len(values)
    if n < 4:
        return 0.0
    sd = values.std()
    if sd < 1e-12:
        return 0.0
    z = (values - values.mean()) / sd
    skew = float((z ** 3).mean())
    kurt = float((z ** 4).mean())
    if kurt <= 0.0:
        return 0.0
    return (skew * skew + 1.0) / kurt


def _diversity(sim: Simulation, rng: np.random.Generator, k: int = 100) -> float:
    """Mean pairwise distance in body-gene space over a random sample."""
    n = len(sim.agents)
    if n < 2:
        return 0.0
    idx = rng.choice(n, size=min(k, n), replace=False)
    m = np.array([
        [sim.agents[i].diet, sim.agents[i].size / 3.0,
         sim.agents[i].sense_range / 6.0, sim.agents[i].speed / 2.0]
        for i in idx
    ])
    diffs = m[:, None, :] - m[None, :, :]
    d = np.sqrt((diffs ** 2).sum(axis=2))
    iu = np.triu_indices(len(m), k=1)
    return float(d[iu].mean()) if len(iu[0]) else 0.0


def collect(sim: Simulation) -> dict[str, float]:
    agents = sim.agents
    n = len(agents)
    diet = np.array([a.diet for a in agents]) if n else np.zeros(0)
    links = np.array([a.brain_links for a in agents]) if n else np.zeros(0)
    row = {
        "tick": sim.tick_count,
        "population": n,
        "births": sim.births,
        "deaths_starved": sim.deaths["starved"],
        "deaths_killed": sim.deaths["killed"],
        "deaths_age": sim.deaths["age"],
        "mean_links": float(links.mean()) if n else 0.0,
        "max_links": int(links.max()) if n else 0,
        "mean_hidden": float(np.mean([a.genome.hidden_count() for a in agents]))
        if n else 0.0,
        "mean_diet": float(diet.mean()) if n else 0.0,
        "std_diet": float(diet.std()) if n else 0.0,
        "diet_bimodality": bimodality(diet) if n else 0.0,
        "mean_size": float(np.mean([a.size for a in agents])) if n else 0.0,
        "mean_sense_range": float(np.mean([a.sense_range for a in agents]))
        if n else 0.0,
        "mean_speed": float(np.mean([a.speed for a in agents])) if n else 0.0,
        "mean_mutation_rate": float(
            np.mean([a.genome.mutation_rate for a in agents])
        ) if n else 0.0,
        "diversity": _diversity(sim, np.random.default_rng(sim.tick_count)),
        "total_plant": sim.world.total_plant(),
        "total_meat": sim.world.total_meat(),
        "total_agent_energy": float(sum(a.energy for a in agents)),
        "mean_depth": float(np.mean([a.depth for a in agents])) if n else 0.0,
        "mean_x": float(np.mean([a.x for a in agents])) if n else 0.0,
    }
    return row


class StatsWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
```

`_diversity` seeds its sampler from `sim.tick_count` rather than drawing from
`sim.rng`. Drawing from the simulation's own generator would make the recorded
statistics change the trajectory of the run being measured.

- [ ] **Step 4: Implement the CLI**

`run.py`:

```python
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
```

- [ ] **Step 5: Run tests and a short smoke run**

```bash
python3 -m pytest tests/test_stats.py -v
python3 run.py --ticks 2000 --width 64 --height 64 --out runs/smoke.csv
```

Expected: 4 passed, and the smoke run prints rows with a non-zero population.

- [ ] **Step 6: Commit**

```bash
git add evolution/stats.py run.py tests/test_stats.py
git commit -m "feat: statistics collection and CLI entry point"
```

---

### Task 13: Population viability tuning

**Files:**
- Modify: `evolution/config.py` (constants only)
- Create: `tests/test_viability.py`

**Interfaces:** No new code. This task tunes the constants in `Config` until the
world can sustain a population, and locks that in with a test.

Nothing downstream works if the population dies at tick 300 or saturates the
grid. This is expected to need iteration; the spec's numbers (§9) are a starting
point, not a solution.

- [ ] **Step 1: Write the failing test**

`tests/test_viability.py`:

```python
from evolution.config import Config
from evolution.sim import Simulation


def test_population_survives_and_does_not_saturate():
    """The world must be able to carry life without carrying all possible life.
    Plant energy is the only brake in the system (spec 4); if this fails, the
    constants are wrong, not the code."""
    cfg = Config(width=64, height=64, seed=0)
    sim = Simulation(cfg)
    sim.run(20_000)
    cells = cfg.width * cfg.height
    assert len(sim.agents) > 0, f"extinct at tick {sim.tick_count}"
    assert len(sim.agents) < 0.5 * cells, "population saturated the grid"


def test_reproduction_actually_happens():
    sim = Simulation(Config(width=64, height=64, seed=0))
    sim.run(5_000)
    assert sim.births > 0, "no agent ever reproduced; check the energy economy"


def test_survives_across_several_seeds():
    for seed in (1, 2, 3):
        sim = Simulation(Config(width=64, height=64, seed=seed))
        sim.run(10_000)
        assert len(sim.agents) > 0, f"extinct on seed {seed}"
```

- [ ] **Step 2: Run it and read the failure**

Run: `python3 -m pytest tests/test_viability.py -v -x`

- [ ] **Step 3: Diagnose before tuning**

Run a short instrumented run and look at which way it is broken:

```bash
python3 run.py --ticks 5000 --width 64 --height 64 --out runs/tune.csv
```

Read the printed rows. Match the symptom to the fix:

| Symptom | Meaning | Change |
|---|---|---|
| Population falls to 0 in the first ~500 ticks, `deaths_starved` dominates | Agents cannot find food before their starting energy runs out | Raise `initial_energy`; lower `basal` |
| Population declines slowly, `births` near 0 | Nobody ever banks enough energy to reproduce | Lower `repro_threshold`; raise `plant_growth` or `eat_rate` |
| Population grows to fill the grid | Food is too abundant relative to cost | Lower `plant_growth` or `plant_cap`; raise `basal` |
| Population oscillates violently between near-zero and saturation | Classic predator-free boom-bust; usually `plant_growth` too high with `repro_threshold` too low | Raise `repro_threshold` first |
| `mean_links` climbs while population is flat or falling | Brain bloat | Raise `brain_cost` |

Change **one constant at a time** and re-run. Record each attempt and its
outcome in the commit message. Do not change more than one at a time — with five
interacting constants, a batch change teaches you nothing about which one
mattered.

- [ ] **Step 4: Do NOT reach for the respawn crutch**

`Config` has no `min_population` field and must not gain one. Per §13, a respawn
floor masks an unbalanced economy and would invalidate every result. If the
population cannot survive, the constants are wrong.

- [ ] **Step 5: Re-run until the tests pass**

Run: `python3 -m pytest tests/test_viability.py -v`
Expected: 3 passed

- [ ] **Step 6: Commit**

```bash
git add evolution/config.py tests/test_viability.py
git commit -m "tune: constants that sustain a viable population

<record each constant changed and the symptom it fixed>"
```

---

### Task 14: The validation gates

**Files:**
- Create: `tests/test_validation.py`

**Interfaces:** No new code. This is §11 tests 7 and 8, and it is the gate that
decides whether the engine actually evolves anything.

**Nothing past this task may be started until both tests pass.** Everything after
this is presentation. If the engine cannot climb a gradient, a beautiful chart of
it failing is worse than no chart, because it looks like a result.

- [ ] **Step 1: Write the failing test**

`tests/test_validation.py`:

```python
import numpy as np
import pytest
from evolution.config import Config
from evolution.sim import Simulation


def mean_x(sim: Simulation) -> float:
    return float(np.mean([a.x for a in sim.agents])) if sim.agents else float("nan")


def eastern_mask(cfg: Config) -> np.ndarray:
    mask = np.zeros((cfg.height, cfg.width), dtype=bool)
    mask[:, cfg.width // 2:] = True
    return mask


def test_null_run_shows_no_directional_drift():
    """Control. With mutation off, reproduction copies genomes exactly, so no
    lineage can get better at anything. If THIS run trends, the trend in the
    real run is an artefact of the harness and proves nothing."""
    cfg = Config(width=64, height=64, wrap=False, seed=21,
                 mutation_enabled=False)
    sim = Simulation(cfg, growth_mask=eastern_mask(cfg))
    sim.run(10_000)
    if not sim.agents:
        pytest.skip("control population went extinct; retune before trusting the gate")
    assert mean_x(sim) < 0.60 * cfg.width, (
        f"control drifted east to {mean_x(sim):.1f} without mutation; "
        "something other than evolution is moving the population"
    )


def test_rigged_world_gate_population_evolves_eastward():
    """Gate. Plants grow only in the eastern half and the world does not wrap,
    so the optimal strategy is knowable in advance: go east. If the engine
    cannot solve this, it cannot solve anything (spec 11)."""
    cfg = Config(width=64, height=64, wrap=False, seed=21)
    sim = Simulation(cfg, growth_mask=eastern_mask(cfg))
    sim.run(10_000)
    assert sim.agents, f"population went extinct at tick {sim.tick_count}"
    assert mean_x(sim) > 0.65 * cfg.width, (
        f"mean x reached only {mean_x(sim):.1f} of {cfg.width}; "
        "the population did not learn to exploit the food gradient"
    )
```

- [ ] **Step 2: Run the gate**

Run: `python3 -m pytest tests/test_validation.py -v`
Expected: both pass.

- [ ] **Step 3: If the gate fails, work the list in order**

Do not proceed and do not weaken the threshold. Check, in this order:

1. **Is the control also drifting?** If `test_null_run` fails too, the bug is in
   the harness, not in evolution. The usual cause is a directional bias in
   `free_adjacent` (children being placed preferentially east) — verify the
   neighbour order is genuinely shuffled by the seeded RNG.
2. **Do brains ever grow?** Add `print(row["mean_links"])` to the run. If it is
   flat, `add_node` is not surviving; re-check Task 5's function-preservation
   test and `brain_cost`.
3. **Can agents perceive the gradient at all?** Sensor channels 0-3 must be
   non-zero for an agent standing near the food boundary. Assert it directly.
4. **Is 10,000 ticks enough?** Try 30,000. If it passes at 30,000, that is a real
   result — raise the tick count in the test rather than lowering the threshold.
5. **Is mutation reaching the population?** Check `mean_mutation_rate` has not
   collapsed to the 0.01 floor.

- [ ] **Step 4: Commit**

```bash
git add tests/test_validation.py
git commit -m "test: null-run control and rigged-world evolution gate"
```

---

### Task 15: Report figure

**Files:**
- Create: `evolution/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: a `stats.csv` written by Task 12.
- Produces: `build_report(csv_path, png_path) -> None`. Importable and callable
  as `python3 -m evolution.report runs/stats.csv runs/report.png`.

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:

```python
from evolution.config import Config
from evolution.report import build_report
from evolution.sim import Simulation
from evolution.stats import StatsWriter, collect


def test_report_renders_from_a_real_run(tmp_path):
    csv_path = tmp_path / "stats.csv"
    png_path = tmp_path / "report.png"
    sim = Simulation(Config(width=32, height=32, initial_agents=40, seed=1))
    writer = StatsWriter(csv_path)
    sim.run(600, on_stats=lambda s: writer.write(collect(s)))
    writer.close()
    build_report(csv_path, png_path)
    assert png_path.exists() and png_path.stat().st_size > 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'evolution.report'`

- [ ] **Step 3: Implement the report**

`evolution/report.py`:

```python
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
    if "bimodality" in [p[0] for p in PANELS]:
        axes.flat[4].axhline(0.555, ls="--", c="crimson", lw=1)
    for ax in axes.flat[len(PANELS):]:
        ax.axis("off")
    fig.suptitle(f"Emergent evolution: {Path(csv_path).name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    build_report(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_report.py -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add evolution/report.py tests/test_report.py
git commit -m "feat: matplotlib report figure"
```

---

### Task 16: Live ASCII viewer

**Files:**
- Create: `evolution/viewers/ascii.py`
- Modify: `run.py` (add `--watch`)
- Test: `tests/test_ascii_viewer.py`

**Interfaces:**
- Consumes: `Simulation` from Task 10.
- Produces: `render(sim, max_w=100, max_h=40) -> str`, and `watch(sim)` which
  prints it with a cursor-home escape.

`render` returns a string rather than printing, so it is testable without
capturing stdout.

Glyphs: `.` empty, `,` sparse plant, `:` medium plant, `#` dense plant,
`o` herbivore (`diet < 0.33`), `x` omnivore, `@` carnivore (`diet > 0.66`).
Agents always draw over plants.

- [ ] **Step 1: Write the failing test**

`tests/test_ascii_viewer.py`:

```python
from evolution.config import Config
from evolution.sim import Simulation
from evolution.viewers.ascii import render


def test_render_shape_and_content():
    sim = Simulation(Config(width=40, height=20, initial_agents=30, seed=1))
    text = render(sim, max_w=40, max_h=20)
    lines = text.splitlines()
    grid = [ln for ln in lines if set(ln) <= set(".,:#ox@")]
    assert len(grid) == 20
    assert all(len(ln) == 40 for ln in grid)


def test_agents_are_visible():
    sim = Simulation(Config(width=40, height=20, initial_agents=200, seed=1))
    assert any(c in render(sim, 40, 20) for c in "ox@")


def test_render_downsamples_a_large_world():
    sim = Simulation(Config(width=128, height=128, initial_agents=50, seed=1))
    grid = [ln for ln in render(sim, max_w=64, max_h=32).splitlines()
            if set(ln) <= set(".,:#ox@")]
    assert len(grid) == 32 and all(len(ln) == 64 for ln in grid)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_ascii_viewer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement the viewer**

`evolution/viewers/ascii.py`:

```python
from evolution.sim import Simulation

PLANT_GLYPHS = ".,:#"


def _agent_glyph(diet: float) -> str:
    if diet < 0.33:
        return "o"
    if diet > 0.66:
        return "@"
    return "x"


def render(sim: Simulation, max_w: int = 100, max_h: int = 40) -> str:
    cfg = sim.cfg
    step_x = max(1, cfg.width // max_w)
    step_y = max(1, cfg.height // max_h)
    cols = cfg.width // step_x
    rows = cfg.height // step_y

    grid = [[PLANT_GLYPHS[0]] * cols for _ in range(rows)]
    for ry in range(rows):
        for rx in range(cols):
            block = sim.world.plant[
                ry * step_y:(ry + 1) * step_y, rx * step_x:(rx + 1) * step_x
            ]
            frac = float(block.mean()) / max(cfg.plant_cap, 1e-9)
            level = min(len(PLANT_GLYPHS) - 1, int(frac * len(PLANT_GLYPHS)))
            grid[ry][rx] = PLANT_GLYPHS[level]

    for a in sim.agents:
        rx, ry = a.x // step_x, a.y // step_y
        if 0 <= ry < rows and 0 <= rx < cols:
            grid[ry][rx] = _agent_glyph(a.diet)

    n = len(sim.agents)
    mean_links = sum(a.brain_links for a in sim.agents) / n if n else 0.0
    mean_diet = sum(a.diet for a in sim.agents) / n if n else 0.0
    header = (
        f"tick {sim.tick_count:>7}  pop {n:>5}  links {mean_links:>6.2f}  "
        f"diet {mean_diet:.3f}  births {sim.births}  "
        f"deaths s/k/a {sim.deaths['starved']}/{sim.deaths['killed']}/"
        f"{sim.deaths['age']}"
    )
    body = "\n".join("".join(r) for r in grid)
    return f"{header}\n{body}\n  o herbivore   x omnivore   @ carnivore"


def watch(sim: Simulation) -> None:
    print("\033[H\033[J" + render(sim), flush=True)
```

- [ ] **Step 4: Wire `--watch` into run.py**

In `run.py`, add the argument and use it inside `record`:

```python
    p.add_argument("--watch", action="store_true", help="live ASCII view")
```

and in `record`, replace the `print(...)` call with:

```python
        if args.watch:
            from evolution.viewers.ascii import watch
            watch(s)
        else:
            print(
                f"tick {row['tick']:>7}  pop {row['population']:>5}  "
                f"links {row['mean_links']:>6.2f}  diet {row['mean_diet']:.3f}  "
                f"bimod {row['diet_bimodality']:.3f}",
                flush=True,
            )
```

The import stays inside the branch so the core never pulls in a viewer.

- [ ] **Step 5: Run tests and watch it live**

```bash
python3 -m pytest tests/test_ascii_viewer.py -v
python3 run.py --ticks 3000 --width 96 --height 48 --watch
```

Expected: 3 passed, and a live view where plant patches visibly drift.

- [ ] **Step 6: Commit**

```bash
git add evolution/viewers/ascii.py run.py tests/test_ascii_viewer.py
git commit -m "feat: live ASCII viewer"
```

---

### Task 17: Enable carnivory and run the real experiment

**Files:**
- Create: `docs/results/2026-XX-XX-run-notes.md`
- Test: `tests/test_predation.py`

**Do NOT change the `allow_attack` default in `Config`.** Leaving it `False` and
opting in per run (`--attack`, or `Config(allow_attack=True)` in a test) keeps
every earlier test meaning exactly what it meant when it was written. Flipping
the default would silently change the conditions of the viability tuning in
Task 13 and the energy-conservation test in Task 11.

**Interfaces:** No new code paths — `_attack` was written in Task 10 and gated
behind `allow_attack`. This task turns it on and runs the experiment the whole
project exists to run.

- [ ] **Step 1: Write the failing test**

`tests/test_predation.py`:

```python
from evolution.config import Config
from evolution.sim import Simulation


def test_attack_is_off_by_default_until_enabled():
    sim = Simulation(Config(width=32, height=32, seed=1))
    sim.run(2000)
    assert sim.deaths["killed"] == 0 or sim.cfg.allow_attack


def test_killing_happens_when_enabled():
    cfg = Config(width=48, height=48, initial_agents=200,
                 allow_attack=True, seed=4)
    sim = Simulation(cfg)
    sim.run(5000)
    assert sim.deaths["killed"] > 0, "no agent ever killed another"


def test_a_kill_leaves_meat_that_can_be_eaten():
    """The killer gains nothing directly; the corpse is a public good (spec 7)."""
    cfg = Config(width=48, height=48, initial_agents=200,
                 allow_attack=True, seed=4)
    sim = Simulation(cfg)
    sim.run(2000)
    assert sim.world.total_meat() > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_predation.py -v`
Expected: `test_killing_happens_when_enabled` fails or passes depending on
whether attacking is reachable. If it fails, that is informative, not a bug in
this task: it means no lineage has discovered attacking. Raise the tick count
before touching anything else.

- [ ] **Step 3: Run the full experiment**

```bash
mkdir -p runs docs/results
python3 run.py --ticks 50000 --seed 1 --attack --out runs/main-s1.csv
python3 run.py --ticks 50000 --seed 2 --attack --out runs/main-s2.csv
python3 run.py --ticks 50000 --seed 3 --attack --out runs/main-s3.csv
python3 run.py --ticks 50000 --seed 1 --attack --no-mutation --out runs/control-s1.csv
```

Three seeds because a single run cannot distinguish evolution from luck, and one
mutation-off control on the same seed as the baseline for comparison.

- [ ] **Step 4: Build the reports**

```bash
for f in runs/*.csv; do
  python3 -m evolution.report "$f" "${f%.csv}.png"
done
```

- [ ] **Step 5: Write up what actually happened**

Create `docs/results/2026-XX-XX-run-notes.md` answering the three claims from
§1 **with numbers from `stats.csv`, and stating plainly when a claim failed**:

```markdown
# Run notes: <date>

Command: <exact command>   Seeds: 1, 2, 3   Ticks: 50,000

## Claim 1: brain complexity grew
mean_links at tick 0 / 10k / 50k: ... / ... / ...
Control (mutation off) at 50k: ...
Verdict: <supported | not supported>

## Claim 2: the diet gene separated into modes
diet_bimodality peak: ...  at tick ...  (threshold 0.555)
Verdict: <supported | not supported>

## Claim 3: a behaviour appeared that was never coded
What was observed, and how it was observed (ASCII view, sensor traces):
Verdict: <supported | not supported>

## Anything surprising
## Anything that looks like a bug rather than a result
```

A "not supported" verdict is a valid and useful outcome. Record it honestly; the
null-run control exists precisely so a negative result can be trusted.

- [ ] **Step 6: Commit**

```bash
git add tests/test_predation.py docs/results
git commit -m "feat: carnivory runs and first full-run results"
```

---

### Task 18: Invariant guard and HTML replay viewer

**Files:**
- Create: `evolution/recorder.py`, `evolution/viewers/html.py`
- Create: `tests/test_no_fitness_function.py`, `tests/test_recorder.py`
- Modify: `run.py` (add `--record`)

**Interfaces:**
- Produces:
  - `Recorder(path, every=10)` with `.capture(sim) -> None`, `.close() -> None`
  - `build_html(jsonl_path, html_path) -> None`

- [ ] **Step 1: Write the guard test**

`tests/test_no_fitness_function.py`:

```python
import ast
from pathlib import Path

CORE = Path("evolution")
BANNED_CALLS = {"sorted", "max", "min"}
ALLOWED = {"stats.py", "report.py"}


def core_modules():
    return [
        p for p in CORE.rglob("*.py")
        if p.name not in ALLOWED and "viewers" not in p.parts
    ]


def test_core_never_ranks_agents():
    """Spec: no fitness function. Ranking agents anywhere in the core would be
    artificial selection wearing a disguise."""
    offenders = []
    for path in core_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name in BANNED_CALLS and node.args:
                src = ast.unparse(node.args[0])
                if "agent" in src.lower():
                    offenders.append(f"{path}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, "ranking of agents found in core:\n" + "\n".join(offenders)


def test_core_does_not_import_viewers_or_matplotlib():
    offenders = []
    for path in core_modules():
        text = path.read_text()
        for banned in ("matplotlib", "evolution.viewers", "import curses"):
            if banned in text:
                offenders.append(f"{path}: imports {banned}")
    assert not offenders, "\n".join(offenders)


def test_core_uses_only_the_seeded_generator():
    offenders = []
    for path in CORE.rglob("*.py"):
        text = path.read_text()
        if "import random" in text or "np.random.seed" in text:
            offenders.append(f"{path}: unseeded randomness")
    assert not offenders, "\n".join(offenders)
```

- [ ] **Step 2: Run it and fix any violation**

Run: `python3 -m pytest tests/test_no_fitness_function.py -v`
Expected: 3 passed. A failure here is serious — it means selection has leaked
into the core and the results so far are suspect.

- [ ] **Step 3: Write the recorder test**

`tests/test_recorder.py`:

```python
import json

from evolution.config import Config
from evolution.recorder import Recorder
from evolution.sim import Simulation


def test_recorder_writes_one_frame_per_interval(tmp_path):
    path = tmp_path / "frames.jsonl"
    sim = Simulation(Config(width=32, height=32, initial_agents=20, seed=1))
    rec = Recorder(path, every=10)
    for _ in range(100):
        sim.tick()
        rec.capture(sim)
    rec.close()
    frames = [json.loads(l) for l in path.read_text().splitlines()]
    assert len(frames) == 10
    assert {"tick", "agents", "plant", "width", "height"} <= set(frames[0])
```

- [ ] **Step 4: Implement the recorder**

`evolution/recorder.py`:

```python
import json
from pathlib import Path

import numpy as np

from evolution.sim import Simulation


class Recorder:
    """Writes downsampled frames as JSONL for the HTML replay viewer."""

    def __init__(self, path: str | Path, every: int = 10, plant_bins: int = 64):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w")
        self.every = every
        self.plant_bins = plant_bins

    def capture(self, sim: Simulation) -> None:
        if sim.tick_count % self.every:
            return
        cfg = sim.cfg
        step = max(1, cfg.width // self.plant_bins)
        h, w = cfg.height // step, cfg.width // step
        coarse = sim.world.plant[: h * step, : w * step]
        coarse = coarse.reshape(h, step, w, step).mean(axis=(1, 3))
        frame = {
            "tick": sim.tick_count,
            "width": w,
            "height": h,
            "plant": np.round(coarse / max(cfg.plant_cap, 1e-9), 3).tolist(),
            "agents": [
                [a.x // step, a.y // step, round(a.diet, 2), round(a.size, 2)]
                for a in sim.agents
            ],
        }
        self._fh.write(json.dumps(frame, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._fh.close()
```

- [ ] **Step 5: Implement the HTML viewer**

`evolution/viewers/html.py` reads the JSONL, embeds it directly into a
self-contained page with a canvas and a scrub slider, and writes one HTML file
with no external requests:

```python
import json
import sys
from pathlib import Path

TEMPLATE = """<!doctype html><meta charset=utf-8>
<title>Evolution replay</title>
<style>
 body{background:#111;color:#ddd;font:14px system-ui;margin:0;padding:16px}
 canvas{image-rendering:pixelated;width:100%;max-width:900px;
        border:1px solid #333;background:#000}
 input{width:100%;max-width:900px}
</style>
<h1 style="font-size:16px">Evolution replay</h1>
<canvas id=c></canvas>
<input id=s type=range min=0 value=0>
<div id=info></div>
<script>
const FRAMES = __DATA__;
const c = document.getElementById('c'), x = c.getContext('2d');
const s = document.getElementById('s'), info = document.getElementById('info');
s.max = FRAMES.length - 1;
function draw(i){
  const f = FRAMES[i];
  c.width = f.width; c.height = f.height;
  const img = x.createImageData(f.width, f.height);
  for (let y = 0; y < f.height; y++)
    for (let X = 0; X < f.width; X++){
      const v = f.plant[y][X], p = 4 * (y * f.width + X);
      img.data[p] = 20; img.data[p+1] = 40 + 180 * v;
      img.data[p+2] = 30; img.data[p+3] = 255;
    }
  x.putImageData(img, 0, 0);
  for (const [ax, ay, diet] of f.agents){
    x.fillStyle = `rgb(${Math.round(60+195*diet)},${Math.round(200-160*diet)},240)`;
    x.fillRect(ax, ay, 1, 1);
  }
  info.textContent = `tick ${f.tick} — ${f.agents.length} agents`;
}
s.oninput = () => draw(+s.value);
draw(0);
let i = 0;
setInterval(() => { if (document.hasFocus()){ i = (i+1) % FRAMES.length;
  s.value = i; draw(i);} }, 80);
</script>
"""


def build_html(jsonl_path: str | Path, html_path: str | Path) -> None:
    frames = [
        json.loads(line)
        for line in Path(jsonl_path).read_text().splitlines()
        if line
    ]
    if not frames:
        raise ValueError(f"{jsonl_path} contains no frames")
    Path(html_path).write_text(
        TEMPLATE.replace("__DATA__", json.dumps(frames, separators=(",", ":")))
    )


if __name__ == "__main__":
    build_html(sys.argv[1], sys.argv[2])
```

- [ ] **Step 6: Wire `--record` into run.py**

```python
    p.add_argument("--record", default=None,
                   help="write frames.jsonl for the HTML replay viewer")
```

Create a `Recorder` when set, call `rec.capture(sim)` from inside `record`, and
close it in the `finally` block alongside the stats writer.

- [ ] **Step 7: Run everything**

```bash
python3 -m pytest tests/ -v
python3 run.py --ticks 20000 --attack --out runs/replay.csv --record runs/frames.jsonl
python3 -m evolution.viewers.html runs/frames.jsonl runs/replay.html
open runs/replay.html
```

Expected: full suite passes; the replay page opens and animates with no network
requests.

- [ ] **Step 8: Commit**

```bash
git add evolution/recorder.py evolution/viewers/html.py run.py tests/
git commit -m "feat: invariant guard tests, frame recorder, HTML replay viewer"
```

---

## Appendix: what "done" means

The project is complete when all of the following are true, verified by running
the commands, not by inspection:

- [ ] `python3 -m pytest tests/ -v` passes with no skips other than the
      documented extinction skip in `test_validation.py`
- [ ] `tests/test_validation.py` passes — the rigged-world gate is the proof the
      engine evolves anything at all
- [ ] `tests/test_no_fitness_function.py` passes — no selection has leaked into
      the core
- [ ] `docs/results/` contains run notes with real numbers for all three claims
      in §1, including any that came out unsupported
- [ ] A `report.png` exists for at least three seeds plus one mutation-off control

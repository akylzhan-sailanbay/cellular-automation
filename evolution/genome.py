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
    nodes += [NodeGene(N_INPUTS + j, "output", "tanh") for j in range(N_OUTPUTS)]
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
            assert n.activation == "identity", "input activation mutated"
        if n.kind == "output":
            assert n.activation == "tanh", "output activation mutated"
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

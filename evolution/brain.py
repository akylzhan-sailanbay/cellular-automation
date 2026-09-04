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

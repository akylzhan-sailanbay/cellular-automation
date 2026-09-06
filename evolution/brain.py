from typing import Callable

import numpy as np

from evolution.genome import Genome

# Recurrent connections are permitted and `identity`/`relu` are unbounded, so a
# cycle with loop gain > 1 amplifies its own state every tick and diverges
# exponentially. Measured: |state| reached 1.1e308 by tick 16055 with a max
# weight of only 3.5. Because outputs are tanh the divergence is INVISIBLE from
# the outside until state overflows to inf, sums become nan, and argmax starts
# returning arbitrary actions. STATE_LIMIT bounds the recurrent state far above
# any meaningful signal (inputs are normalised, weights are ~N(0,1)) while making
# divergence impossible.
STATE_LIMIT = 1e6

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
        self.hidden_idx = np.array(
            [index[n.id] for n in genome.nodes if n.kind == "hidden"], dtype=np.int64
        )
        # Lesion switch for the ablation experiment. Silences hidden nodes
        # WITHOUT touching the genome, so enabled_count and therefore the
        # per-tick brain rent stay exactly the same. Deleting the nodes would
        # refund 13-20% of a reproduction budget and lesioned agents might then
        # survive BETTER for purely energetic reasons, inverting the result.
        self.lesioned = False

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
        np.clip(new, -STATE_LIMIT, STATE_LIMIT, out=new)
        if self.lesioned and self.hidden_idx.size:
            new[self.hidden_idx] = 0.0
        new[self.input_idx] = inputs

        self.state = new
        return new[self.output_idx]

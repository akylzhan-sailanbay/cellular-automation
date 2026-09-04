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

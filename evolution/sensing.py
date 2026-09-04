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

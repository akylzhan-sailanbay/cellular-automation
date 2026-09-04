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


_PATCH_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _patch_masks(r: int) -> tuple[np.ndarray, np.ndarray]:
    """For radius r return (dir_mask of shape (4, s, s), dist of shape (s, s))."""
    if r in _PATCH_CACHE:
        return _PATCH_CACHE[r]
    size = 2 * r + 1
    dir_mask = np.zeros((4, size, size), dtype=bool)
    dist = np.zeros((size, size))
    for iy, dy in enumerate(range(-r, r + 1)):
        for ix, dx in enumerate(range(-r, r + 1)):
            dist[iy, ix] = max(abs(dx), abs(dy))
            if dx == 0 and dy == 0:
                continue
            dir_mask[direction_of(dx, dy), iy, ix] = True
    _PATCH_CACHE[r] = (dir_mask, dist)
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
    """Vectorised sensing for the whole population.

    Groups agents by integer sense radius (at most six groups) and does one
    gather per group instead of one per agent. Must agree with
    sense_reference to within floating point; test_sensing_batched enforces it.
    """
    n = len(xs)
    out = np.zeros((n, N_INPUTS))
    if n == 0:
        return out

    xs = np.asarray(xs, dtype=np.int64)
    ys = np.asarray(ys, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.float64)
    radii = np.clip(np.asarray(ranges).astype(np.int64), 1, 6)

    for radius in np.unique(radii):
        r = int(radius)
        sel = np.nonzero(radii == r)[0]
        k = len(sel)
        dir_mask, dist = _patch_masks(r)
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
        osize = world.occ_size[rows, cols].reshape(k, -1)
        odiet = world.occ_diet[rows, cols].reshape(k, -1)

        counts = dir_mask.sum(axis=(1, 2)).astype(np.float64)
        plant_sum = np.tensordot(plant, dir_mask, axes=([1, 2], [1, 2]))
        out[np.ix_(sel, np.arange(4))] = plant_sum / (counts * cfg.plant_cap)

        flat_dist = dist.reshape(-1)
        rowidx = np.arange(k)
        for d in range(4):
            visible = occ & dir_mask[d]
            masked = np.where(visible, dist[None, :, :], np.inf).reshape(k, -1)
            best = np.argmin(masked, axis=1)
            # argmin over an all-inf row returns 0, a real cell. Without this
            # mask every agent would believe a neighbour is always due north.
            found = np.isfinite(masked[rowidx, best])
            out[sel, 4 + d] = np.where(found, 1.0 - flat_dist[best] / (r + 1.0), 0.0)
            out[sel, 8 + d] = np.where(
                found, osize[rowidx, best] / np.maximum(sizes[sel], 1e-9), 0.0
            )
            out[sel, 12 + d] = np.where(found, odiet[rowidx, best], 0.0)

    out[:, 16] = np.asarray(energies) / cfg.repro_threshold
    out[:, 17] = np.asarray(ages) / cfg.max_age
    out[:, 18] = sizes / MAX_SIZE
    out[:, 19] = 1.0
    return out

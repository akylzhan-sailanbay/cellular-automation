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
        self,
        agent_id: int,
        x: int,
        y: int,
        nx: int,
        ny: int,
        size: float,
        diet: float,
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

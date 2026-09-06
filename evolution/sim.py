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
    """The tick loop. The only module that knows the order of events.

    Three deliberate ordering decisions:
      1. Sensing happens ONCE per tick for the whole population, before anyone
         acts, so every agent perceives the same snapshot and the shuffle
         confers no advantage.
      2. A fast agent acts twice on ONE perception. Speed buys extra actions,
         not extra eyes.
      3. Within an action the order is eat, attack, reproduce, move.
    """

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
        self.ledger = {
            "grown": 0.0, "metabolism": 0.0, "decayed": 0.0, "conversion": 0.0
        }
        # Cumulative energy absorbed by feeding. A behavioural readout for the
        # ablation experiment: population size is chaotic over thousands of
        # ticks, but "are they still managing to eat" responds immediately.
        self.intake = 0.0
        # Experimental ablation switch. When True, every brain born into this
        # simulation has its hidden nodes silenced. Children must inherit it or
        # the lesion would wash out within a generation as fresh brains appear.
        self.lesion = False
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
            self._spawn(
                random_genome(self.rng, self.cfg), x, y, self.cfg.initial_energy, 0
            )
            placed += 1

    def _spawn(
        self, genome: Genome, x: int, y: int, energy: float, depth: int
    ) -> Agent:
        a = Agent.create(self.next_id, genome, x, y, energy, depth)
        a.brain.lesioned = self.lesion
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
        for key in self.ledger:
            self.ledger[key] = 0.0

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

    def _act(
        self,
        a: Agent,
        out: np.ndarray,
        births: list[tuple[Genome, int, int, float, int]],
    ) -> None:
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
        self.intake += gained
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

    def _reproduce(
        self, a: Agent, births: list[tuple[Genome, int, int, float, int]]
    ) -> None:
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
        births.append(
            (mutate(a.genome, self.rng, cfg), spot[0], spot[1], child_energy,
             a.depth + 1)
        )

    def _kill(self, a: Agent, cause: str) -> None:
        if not a.alive:
            return
        a.alive = False
        self.deaths[cause] += 1
        self.world.clear(a.x, a.y)
        self.world.deposit_meat(a.x, a.y, self.cfg.meat_per_size * a.size)
        self.by_id.pop(a.id, None)

    def run(
        self, ticks: int, on_stats: Callable[["Simulation"], None] | None = None
    ) -> None:
        for _ in range(ticks):
            self.tick()
            if on_stats and self.tick_count % self.cfg.stats_interval == 0:
                on_stats(self)
            if not self.agents:
                break

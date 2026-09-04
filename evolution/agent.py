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
        cls,
        agent_id: int,
        genome: Genome,
        x: int,
        y: int,
        energy: float,
        depth: int,
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
    cost = cfg.basal + cfg.upkeep * a.size * a.size + cfg.brain_cost * a.brain_links
    if moved:
        cost += cfg.move_cost * a.size
    return cost

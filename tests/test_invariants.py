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
            f"unexplained energy at tick {sim.tick_count}: {after - expected:+.9f}"
        )
        assert len(sim.agents) == 60, "no agent should have died in this setup"

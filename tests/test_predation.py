from evolution.config import Config
from evolution.sim import Simulation


def test_attack_is_off_by_default():
    """allow_attack stays False in Config on purpose. Every earlier test was
    tuned and measured with carnivory disabled; flipping the default would
    silently change what those tests mean. Carnivory is opted into per run."""
    assert Config().allow_attack is False


def test_no_kills_happen_while_attack_is_disabled():
    sim = Simulation(Config(width=32, height=32, seed=1))
    sim.run(2000)
    assert sim.deaths["killed"] == 0


def test_killing_happens_when_enabled():
    cfg = Config(width=48, height=48, initial_agents=200, allow_attack=True, seed=4)
    sim = Simulation(cfg)
    sim.run(5000)
    assert sim.deaths["killed"] > 0, "no agent ever killed another"


def test_a_kill_leaves_meat_that_someone_else_could_eat():
    """The killer gains NO energy directly; the corpse is a public good, so
    killing and feeding are two separately-evolved behaviours (spec 7)."""
    cfg = Config(width=48, height=48, initial_agents=200, allow_attack=True, seed=4)
    sim = Simulation(cfg)
    sim.run(2000)
    assert sim.world.total_meat() > 0.0

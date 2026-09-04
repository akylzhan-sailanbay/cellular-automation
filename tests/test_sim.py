from evolution.config import Config
from evolution.sim import Simulation


def test_initial_population_is_placed():
    cfg = Config(width=32, height=32, initial_agents=50)
    sim = Simulation(cfg)
    assert len(sim.agents) == 50
    assert int((sim.world.occ_id != -1).sum()) == 50


def test_tick_advances_and_ages_agents():
    cfg = Config(width=32, height=32, initial_agents=20)
    sim = Simulation(cfg)
    sim.tick()
    assert sim.tick_count == 1
    assert all(a.age == 1 for a in sim.agents)


def test_agents_starve_without_food():
    cfg = Config(
        width=32, height=32, initial_agents=20,
        initial_energy=5.0, plant_growth=0.0, plant_cap=0.0,
    )
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(200)
    assert len(sim.agents) == 0
    assert sim.deaths["starved"] > 0


def test_agents_die_of_old_age():
    cfg = Config(width=32, height=32, initial_agents=20, max_age=10,
                 initial_energy=1e6)
    sim = Simulation(cfg)
    sim.run(30)
    assert sim.deaths["age"] > 0


def test_occupancy_stays_consistent_with_the_agent_list():
    cfg = Config(width=32, height=32, initial_agents=40, seed=3)
    sim = Simulation(cfg)
    sim.run(300)
    live_ids = {a.id for a in sim.agents}
    grid_ids = set(sim.world.occ_id[sim.world.occ_id != -1].tolist())
    assert live_ids == grid_ids, "grid and agent list disagree about who exists"


def test_no_two_agents_share_a_cell():
    cfg = Config(width=32, height=32, initial_agents=60, seed=5)
    sim = Simulation(cfg)
    sim.run(300)
    coords = [(a.x, a.y) for a in sim.agents]
    assert len(coords) == len(set(coords))


def test_dead_agents_leave_meat():
    cfg = Config(width=16, height=16, initial_agents=10,
                 initial_energy=2.0, plant_growth=0.0)
    sim = Simulation(cfg)
    sim.world.plant[:] = 0.0
    sim.run(50)
    assert sim.world.total_meat() > 0.0

from evolution.config import Config
from evolution.sim import Simulation
from evolution.viewers.ascii import render

GLYPHS = set(".,:#ox@")


def grid_lines(text):
    return [ln for ln in text.splitlines() if ln and set(ln) <= GLYPHS]


def test_render_shape_and_content():
    sim = Simulation(Config(width=40, height=20, initial_agents=30, seed=1))
    lines = grid_lines(render(sim, max_w=40, max_h=20))
    assert len(lines) == 20
    assert all(len(ln) == 40 for ln in lines)


def test_agents_are_visible():
    sim = Simulation(Config(width=40, height=20, initial_agents=200, seed=1))
    assert any(c in render(sim, 40, 20) for c in "ox@")


def test_render_downsamples_a_large_world():
    sim = Simulation(Config(width=128, height=128, initial_agents=50, seed=1))
    lines = grid_lines(render(sim, max_w=64, max_h=32))
    assert len(lines) == 32 and all(len(ln) == 64 for ln in lines)

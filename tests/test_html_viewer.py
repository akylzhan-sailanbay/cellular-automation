from evolution.config import Config
from evolution.recorder import Recorder
from evolution.sim import Simulation
from evolution.viewers.html import build_html


def test_html_is_self_contained(tmp_path):
    jsonl, html = tmp_path / "f.jsonl", tmp_path / "f.html"
    sim = Simulation(Config(width=32, height=32, initial_agents=20, seed=1))
    rec = Recorder(jsonl, every=10)
    for _ in range(50):
        sim.tick()
        rec.capture(sim)
    rec.close()
    build_html(jsonl, html)
    text = html.read_text()
    assert "<canvas" in text and "FRAMES" in text
    for external in ("http://", "https://", "src=", "@import"):
        assert external not in text, f"page reaches out to {external}"

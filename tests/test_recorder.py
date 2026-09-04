import json

from evolution.config import Config
from evolution.recorder import Recorder
from evolution.sim import Simulation


def test_recorder_writes_one_frame_per_interval(tmp_path):
    path = tmp_path / "frames.jsonl"
    sim = Simulation(Config(width=32, height=32, initial_agents=20, seed=1))
    rec = Recorder(path, every=10)
    for _ in range(100):
        sim.tick()
        rec.capture(sim)
    rec.close()
    frames = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(frames) == 10
    assert {"tick", "agents", "plant", "width", "height"} <= set(frames[0])

from evolution.config import Config
from evolution.report import build_report
from evolution.sim import Simulation
from evolution.stats import StatsWriter, collect


def test_report_renders_from_a_real_run(tmp_path):
    csv_path = tmp_path / "stats.csv"
    png_path = tmp_path / "report.png"
    sim = Simulation(Config(width=32, height=32, initial_agents=40, seed=1))
    writer = StatsWriter(csv_path)
    sim.run(600, on_stats=lambda s: writer.write(collect(s)))
    writer.close()
    build_report(csv_path, png_path)
    assert png_path.exists() and png_path.stat().st_size > 5000

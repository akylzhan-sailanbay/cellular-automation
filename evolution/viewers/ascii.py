from evolution.sim import Simulation

PLANT_GLYPHS = ".,:#"


def _agent_glyph(diet: float) -> str:
    if diet < 0.33:
        return "o"
    if diet > 0.66:
        return "@"
    return "x"


def render(sim: Simulation, max_w: int = 100, max_h: int = 40) -> str:
    cfg = sim.cfg
    step_x = max(1, cfg.width // max_w)
    step_y = max(1, cfg.height // max_h)
    cols = cfg.width // step_x
    rows = cfg.height // step_y

    grid = [[PLANT_GLYPHS[0]] * cols for _ in range(rows)]
    for ry in range(rows):
        for rx in range(cols):
            block = sim.world.plant[
                ry * step_y:(ry + 1) * step_y, rx * step_x:(rx + 1) * step_x
            ]
            frac = float(block.mean()) / max(cfg.plant_cap, 1e-9)
            level = min(len(PLANT_GLYPHS) - 1, int(frac * len(PLANT_GLYPHS)))
            grid[ry][rx] = PLANT_GLYPHS[level]

    for a in sim.agents:
        rx, ry = a.x // step_x, a.y // step_y
        if 0 <= ry < rows and 0 <= rx < cols:
            grid[ry][rx] = _agent_glyph(a.diet)

    n = len(sim.agents)
    mean_links = sum(a.brain_links for a in sim.agents) / n if n else 0.0
    mean_diet = sum(a.diet for a in sim.agents) / n if n else 0.0
    header = (
        f"tick {sim.tick_count:>7}  pop {n:>5}  links {mean_links:>6.2f}  "
        f"diet {mean_diet:.3f}  births {sim.births}  "
        f"deaths s/k/a {sim.deaths['starved']}/{sim.deaths['killed']}/"
        f"{sim.deaths['age']}"
    )
    body = "\n".join("".join(r) for r in grid)
    return f"{header}\n{body}\n  o herbivore   x omnivore   @ carnivore"


def watch(sim: Simulation) -> None:
    print("\033[H\033[J" + render(sim), flush=True)

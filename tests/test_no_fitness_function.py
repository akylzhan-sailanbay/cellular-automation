import ast
from pathlib import Path

CORE = Path(__file__).resolve().parent.parent / "evolution"
BANNED_CALLS = {"sorted", "max", "min"}
# Exporters and analysis modules may read agent state for measurement and
# display. They are not part of the simulation core and cannot influence
# who reproduces.
ALLOWED = {"stats.py", "report.py", "showcase.py"}


def core_modules():
    return [
        p for p in CORE.rglob("*.py")
        if p.name not in ALLOWED and "viewers" not in p.parts
    ]


def test_core_never_ranks_agents():
    """Spec: no fitness function. Ranking agents anywhere in the core would be
    artificial selection wearing a disguise."""
    offenders = []
    for path in core_modules():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", "")
            if name in BANNED_CALLS and node.args:
                src = ast.unparse(node.args[0])
                if "agent" in src.lower():
                    offenders.append(f"{path.name}:{node.lineno}: {ast.unparse(node)}")
    assert not offenders, "ranking of agents found in core:\n" + "\n".join(offenders)


def test_core_does_not_import_viewers_or_matplotlib():
    offenders = []
    for path in core_modules():
        text = path.read_text()
        for banned in ("matplotlib", "evolution.viewers", "import curses"):
            if banned in text:
                offenders.append(f"{path.name}: imports {banned}")
    assert not offenders, "\n".join(offenders)


def test_core_uses_only_the_seeded_generator():
    offenders = []
    for path in CORE.rglob("*.py"):
        text = path.read_text()
        if "import random" in text or "np.random.seed" in text:
            offenders.append(f"{path.name}: unseeded randomness")
    assert not offenders, "\n".join(offenders)

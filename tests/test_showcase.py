import json

import numpy as np
import pytest

from evolution.genome import validate
from evolution.showcase import (
    ALPHABET,
    PLANT_BINS,
    _links_bucket,
    build_bundle,
)
from evolution.genome import ConnGene, Genome, NodeGene

TICKS, EVERY = 4000, 100


@pytest.fixture(scope="module")
def bundle():
    return build_bundle(ticks=TICKS, every=EVERY)


def test_alphabet_covers_the_grid_exactly():
    """Positions are packed one character each, so the alphabet must be at
    least as large as the grid is wide or coordinates would wrap silently."""
    assert len(ALPHABET) == 64
    assert len(set(ALPHABET)) == 64


def test_both_worlds_export_the_same_frame_count(bundle):
    """The page drives both canvases from one clock; unequal lengths would
    desynchronise the comparison."""
    a, b = bundle["worlds"]
    if a["extinct"] is None and b["extinct"] is None:
        assert len(a["frames"]) == len(b["frames"])


def test_frames_are_well_formed(bundle):
    for w in bundle["worlds"]:
        for f in w["frames"]:
            assert len(f["p"]) == PLANT_BINS * PLANT_BINS
            assert len(f["a"]) % 4 == 0
            assert len(f["a"]) // 4 == f["s"][0], "agent chars disagree with population"
            assert set(f["p"]) <= set(ALPHABET)
            assert set(f["a"]) <= set(ALPHABET)


def test_positions_round_trip_exactly(bundle):
    """Diet and size are lossy by design; positions must not be."""
    grid = bundle["meta"]["grid"]
    for w in bundle["worlds"]:
        for f in w["frames"][:20]:
            for i in range(0, len(f["a"]), 4):
                x = ALPHABET.index(f["a"][i])
                y = ALPHABET.index(f["a"][i + 1])
                assert 0 <= x < grid and 0 <= y < grid


def test_links_bucket_is_monotone_and_bounded():
    buckets = [_links_bucket(n) for n in range(10, 40)]
    assert buckets[0] == 0 and buckets[-1] == 9
    assert all(b <= c for b, c in zip(buckets, buckets[1:]))
    assert _links_bucket(0) == 0, "must clamp, not go negative"


def test_milestones_are_ordered_and_in_range(bundle):
    for w in bundle["worlds"]:
        if not w["frames"]:
            continue
        lo, hi = w["frames"][0]["t"], w["frames"][-1]["t"]
        for name, tick in w["milestones"].items():
            assert lo <= tick <= hi, f"{name} at {tick} outside [{lo}, {hi}]"
        if "first_structure" in w["milestones"] and "peak_complexity" in w["milestones"]:
            assert w["milestones"]["first_structure"] <= w["milestones"]["peak_complexity"]


def test_crash_milestone_skips_the_founding_transient(bundle):
    """Every run dies back hard in its first few hundred ticks as random
    genomes are culled. Reporting that as the crash points at the wrong moment."""
    for w in bundle["worlds"]:
        if "crash" in w["milestones"]:
            assert w["milestones"]["crash"] > TICKS // 5


def test_champion_genomes_are_valid(bundle):
    for w in bundle["worlds"]:
        for champ in w["champions"]:
            kinds = {"i": "input", "h": "hidden", "o": "output"}
            g = Genome(
                nodes=[NodeGene(i, kinds[k], a) for i, k, a in champ["nodes"]],
                conns=[ConnGene(s, d, wt) for s, d, wt in champ["conns"]],
                body={"diet": 0.5, "size": 1.0, "sense_range": 3.0, "speed": 1.0},
                mutation_rate=0.1,
                next_node_id=max(n[0] for n in champ["nodes"]) + 1,
            )
            validate(g)


def test_bundle_stays_under_the_size_cap(bundle):
    kb = len(json.dumps(bundle, separators=(",", ":"))) / 1024
    projected = kb * (40_000 / TICKS)
    assert projected < 2048, f"projected full bundle {projected:.0f} KB exceeds 2 MB"


def test_export_is_deterministic():
    a = json.dumps(build_bundle(ticks=1500, every=100), separators=(",", ":"))
    b = json.dumps(build_bundle(ticks=1500, every=100), separators=(",", ":"))
    assert a == b

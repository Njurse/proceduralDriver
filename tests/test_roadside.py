"""worldgen.roads.roadside: per-biome strips along the roads, deterministic,
absent where a biome lays none, and kept clear by the structures."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.world import World  # noqa: E402
from worldgen.biomes import parse_biome  # noqa: E402
from worldgen.query import WorldGround  # noqa: E402
from worldgen.roads.roadside import reach  # noqa: E402
from worldgen.spec import WorldSpec  # noqa: E402
from worldgen.structures.base import poly_distance  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World(seed=42)


def _keys(world, biome, r=2):
    x, z = world.regions.site_of(biome)
    size = world.spec.chunk_size_m
    ci, ck = math.floor(x / size), math.floor(z / size)
    return [(i, k) for i in range(ci - r, ci + r + 1) for k in range(ck - r, ck + r + 1)]


@pytest.mark.parametrize("biome", ["metropolis", "british_town"])
def test_roadside_is_deterministic_and_sane(world, biome):
    other = World(seed=42)
    found = 0
    for key in _keys(world, biome):
        a, b = world.roadside_mesh(*key), other.roadside_mesh(*key)
        assert (a is None) == (b is None)
        if a is None:
            continue
        found += 1
        for f in ("pos", "normal", "color", "uv", "tris"):
            assert np.array_equal(getattr(a, f), getattr(b, f))
        assert np.isfinite(a.pos).all() and np.allclose(np.linalg.norm(a.normal, axis=1), 1.0)
        assert a.tris.max() < len(a.pos)
    assert found > 0


def test_sidewalks_stand_on_the_road_and_walls_rise(world):
    m = next(m for m in map(lambda k: world.roadside_mesh(*k), _keys(world, "metropolis"))
             if m is not None)
    h = world.query(m.pos[:, 0], m.pos[:, 2])[0]
    assert (m.pos[:, 1] - h).max() == pytest.approx(world.spec.roads.lift_m + 0.15, abs=1e-6)
    m = next(m for m in map(lambda k: world.roadside_mesh(*k), _keys(world, "british_town"))
             if m is not None)
    h = world.query(m.pos[:, 0], m.pos[:, 2])[0]
    assert (m.pos[:, 1] - h).max() > 0.75


def test_a_biome_without_a_profile_lays_nothing():
    jungle = parse_biome({"name": "jungle", "roads": {"grid_spacing_m": 150.0}})
    assert jungle.roads.roadside.side_m == 0.0 and jungle.roads.roadside.reach_m == 0.0
    g = WorldGround(WorldSpec(seed=7, biomes=(jungle,)))
    assert all(g.roadside_mesh(i, k) is None for i in range(-3, 3) for k in range(-3, 3))
    assert any(g.road_mesh(i, k) is not None for i in range(-3, 3) for k in range(-3, 3))


def test_bad_profiles_are_errors():
    with pytest.raises(ValueError, match="curb_m"):
        parse_biome({"roads": {"roadside": {"side_m": 1.0, "curb_m": 2.0}}})
    with pytest.raises(ValueError, match="unknown"):
        parse_biome({"roads": {"roadside": {"width_m": 1.0}}})


@pytest.mark.parametrize("biome", ["metropolis", "british_town", "desert"])
def test_no_structure_stands_on_a_roadside(world, biome):
    n = 0
    for key in _keys(world, biome):
        for s in world.structures.chunk(*key):
            lo, hi = s.footprint.min(axis=0), s.footprint.max(axis=0)
            gx, gz = np.meshgrid(np.arange(lo[0], hi[0], 0.5), np.arange(lo[1], hi[1], 0.5))
            gx, gz = gx.ravel(), gz.ravel()
            inside = poly_distance(s.footprint, gx, gz) == 0.0
            gx, gz = gx[inside], gz[inside]
            for r in world.roads.roads_near(lo[0] - 20, lo[1] - 20, hi[0] + 20, hi[1] + 20):
                t = r.terms(gx, gz)
                assert (t[0] >= reach(r)).all(), (s.kind, s.pos, float(t[0].min()), reach(r))
            n += 1
    assert n > 0

"""worldgen.structures: deterministic, off the roads, in their biome, sane counts."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.world import World  # noqa: E402
from worldgen.biomes import parse_biome  # noqa: E402
from worldgen.query import WorldGround  # noqa: E402
from worldgen.spec import WorldSpec  # noqa: E402
from worldgen.structures import structures_mesh  # noqa: E402
from worldgen.structures.base import poly_distance  # noqa: E402

#              biome           kind counts over 5 x 5 chunks round a region site
COUNTS = {"metropolis": {"tower": (25, 250)},
          "desert": {"rock": (3, 80), "cactus": (1, 40), "pole": (1, 60)},
          "british_town": {"cottage": (3, 60), "tree": (1, 60), "shrub": (1, 60)}}


@pytest.fixture(scope="module")
def world():
    return World(seed=42)


def _block(world, biome, r=2):
    x, z = world.regions.site_of(biome)
    size = world.spec.chunk_size_m
    ci, ck = math.floor(x / size), math.floor(z / size)
    return {(i, k): world.structures.chunk(i, k)
            for i in range(ci - r, ci + r + 1) for k in range(ck - r, ck + r + 1)}


@pytest.mark.parametrize("biome", sorted(COUNTS))
def test_counts_kinds_and_biome(world, biome):
    found = [s for chunk in _block(world, biome).values() for s in chunk]
    kinds = {}
    for s in found:
        kinds[s.kind] = kinds.get(s.kind, 0) + 1
    for kind, (lo, hi) in COUNTS[biome].items():
        assert lo <= kinds.get(kind, 0) <= hi, (kind, kinds)
    b = world.biomes.index(next(bi for bi in world.biomes if bi.name == biome))
    for s in found:
        fp = s.footprint
        assert (world.biome_index(fp[:, 0], fp[:, 1]) == b).all()


@pytest.mark.parametrize("biome", sorted(COUNTS))
def test_nothing_stands_on_a_road(world, biome):
    n = 0
    for s in (s for chunk in _block(world, biome).values() for s in chunk):
        lo, hi = s.footprint.min(axis=0), s.footprint.max(axis=0)
        gx, gz = np.meshgrid(np.arange(lo[0], hi[0], 0.75), np.arange(lo[1], hi[1], 0.75))
        gx, gz = gx.ravel(), gz.ravel()
        inside = poly_distance(s.footprint, gx, gz) == 0.0
        if inside.any():
            assert not world.query(gx[inside], gz[inside])[3].any(), (s.kind, s.pos)
            n += 1
    assert n > 0


def test_structures_are_deterministic(world):
    other = World(seed=42)
    for key, chunk in _block(world, "metropolis", r=1).items():
        again = other.structures.chunk(*key)
        assert [(s.kind, s.pos, s.height, s.color, s.trim, s.base_y) for s in chunk] == \
            [(s.kind, s.pos, s.height, s.color, s.trim, s.base_y) for s in again]
        assert all(np.array_equal(a.footprint, b.footprint)
                   for a, b in zip(chunk, again, strict=True))


def test_towers_differ_by_seed_and_light_some_neon(world):
    a = [s for c in _block(world, "metropolis").values() for s in c]
    other = World(seed=43)
    b = [s for c in _block(other, "metropolis").values() for s in c]
    assert {s.pos for s in a}.isdisjoint({s.pos for s in b})
    assert 0 < sum(s.emissive for s in a) < len(a)
    assert max(s.height for s in a) > 3.0 * min(s.height for s in a)


def test_mesh_is_finite_and_closed_over_the_footprints(world):
    chunks = _block(world, "metropolis", r=1)
    inst = [s for c in chunks.values() for s in c]
    m = structures_mesh(inst)
    assert m.tris.shape[1] == 3 and len(m.pos) == m.tris.size
    assert np.isfinite(m.pos).all() and np.allclose(np.linalg.norm(m.normal, axis=1), 1.0)
    assert m.pos[:, 1].max() > 20.0 and (m.color[:, :3] <= 1.0).all()


def test_unknown_generator_is_an_error():
    b = parse_biome({"name": "x", "structures": [{"gen": "nope"}]})
    with pytest.raises(ValueError, match="unknown gen"):
        WorldGround(WorldSpec(seed=1, biomes=(b,)))
    with pytest.raises(ValueError, match="structures"):
        parse_biome({"name": "x", "structures": [{"kinds": []}]})

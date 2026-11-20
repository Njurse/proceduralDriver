"""Road junctions: the strips and their caps leave no hole where roads meet,
and the ground query calls the whole junction road."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World(seed=42)


def _side(p, q, rx, rz):
    return (q[:, 0] - p[:, 0]) * (rz - p[:, 1]) - (q[:, 1] - p[:, 1]) * (rx - p[:, 0])


def _covered(meshes, px, pz):
    """True where (px, pz) lies under a triangle of one of the meshes (seen from above)."""
    hit = np.zeros(len(px), bool)
    rx, rz = px[:, None], pz[:, None]
    for m in meshes:
        a, b, c = (m.pos[m.tris[:, i]][:, [0, 2]] for i in range(3))
        s1, s2, s3 = _side(a, b, rx, rz), _side(b, c, rx, rz), _side(c, a, rx, rz)
        inside = (((s1 >= -1e-9) & (s2 >= -1e-9) & (s3 >= -1e-9))
                  | ((s1 <= 1e-9) & (s2 <= 1e-9) & (s3 <= 1e-9)))
        hit |= inside.any(axis=1)
    return hit


def _junctions(world, biome, r=450.0, near=150.0):
    x, z = world.regions.site_of(biome)
    nodes = {}
    for road in world.roads.roads_near(x - r, z - r, x + r, z + r):
        for p in (road.pts[0], road.pts[-1]):
            key = (round(p[0] * 8), round(p[1] * 8))
            n, hw, _ = nodes.get(key, (0, 0.0, p))
            nodes[key] = (n + 1, max(hw, road.half_width), p)
    out = [(hw, p) for n, hw, p in nodes.values() if n >= 3]
    out.sort(key=lambda t: math.hypot(t[1][0] - x, t[1][1] - z))
    return out[:4]


@pytest.mark.parametrize("biome", ["metropolis", "british_town", "village", "plains"])
def test_junctions_have_no_holes_and_are_road(world, biome):
    found = _junctions(world, biome)
    assert found
    size = world.spec.chunk_size_m
    ang = np.tile(np.linspace(0.0, 2.0 * math.pi, 48, endpoint=False), 3)
    for hw, p in found:
        rr = np.repeat([0.4, 0.75, 0.97], 48) * hw
        px, pz = p[0] + rr * np.cos(ang), p[1] + rr * np.sin(ang)
        ci, ck = math.floor(p[0] / size), math.floor(p[1] / size)
        meshes = [m for m in (world.road_mesh(ci + i, ck + k)
                              for i in (-1, 0, 1) for k in (-1, 0, 1)) if m is not None]
        hit = _covered(meshes, px, pz)
        assert hit.all(), (biome, p, int((~hit).sum()))
        assert world.query(px, pz)[3].all()

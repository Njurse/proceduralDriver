"""The procedural terrain (worldgen/height.py) and its chunk meshes (worldgen/chunk.py)."""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.car_profile import load_cosmetics  # noqa: E402
from core.redriver2_physics import PhysicsWorld, Vehicle  # noqa: E402
from game.world_view import ChunkStreamer  # noqa: E402
from world.spec import world_spec  # noqa: E402
from worldgen.chunk import chunk_mesh, chunks_around  # noqa: E402
from worldgen.ground import Ground  # noqa: E402
from worldgen.height import Terrain  # noqa: E402

DT = 1.0 / 60.0
RNG = np.random.default_rng(5)
PTS = RNG.uniform(-5000.0, 5000.0, (200, 2))

SPEC = world_spec(biomes=())  # chunk size and res


@pytest.fixture(scope="module")
def terrain():
    return Terrain(seed=config.TERRAIN_SEED)


def _inp(steer=0.0, throttle=0.0, handbrake=False):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=0.0, handbrake=handbrake,
                           fast_steer=False, wheelspin=False)


def test_terrain_is_a_ground(terrain):
    assert isinstance(terrain, Ground)
    h, n, surface = terrain(12.5, -40.0)
    assert h == terrain.height(12.5, -40.0)
    assert np.allclose(n, terrain.normal(12.5, -40.0))
    assert surface == terrain.surface(12.5, -40.0)


def test_same_seed_same_world_other_seed_differs():
    x, z = PTS[:, 0], PTS[:, 1]
    a, b, c = Terrain(seed=11), Terrain(seed=11), Terrain(seed=12)
    assert np.array_equal(a.height(x, z), b.height(x, z))
    assert not np.allclose(a.height(x, z), c.height(x, z))


def test_scalar_and_array_paths_agree(terrain):
    arr = terrain.sample(PTS[:, 0], PTS[:, 1])
    for k, (x, z) in enumerate(PTS[:20]):
        one = terrain.sample(x, z)
        assert one == pytest.approx([arr[0][k], arr[1][k], arr[2][k]], abs=1e-12)


def test_normals_match_the_height_field(terrain):
    eps = 1e-3
    for x, z in PTS[:50]:
        dhx = (terrain.height(x + eps, z) - terrain.height(x - eps, z)) / (2 * eps)
        dhz = (terrain.height(x, z + eps) - terrain.height(x, z - eps)) / (2 * eps)
        expect = np.array([-dhx, 1.0, -dhz]) / math.sqrt(dhx * dhx + 1.0 + dhz * dhz)
        n = terrain.normal(x, z)
        assert np.linalg.norm(n) == pytest.approx(1.0)
        assert n == pytest.approx(expect, abs=1e-6)


def test_hills_are_gentle_and_a_few_metres_high(terrain):
    xs = np.arange(-1500.0, 1500.0, 3.0)
    gx, gz = np.meshgrid(xs, xs)
    h, dhx, dhz = terrain.sample(gx, gz)
    slope = np.degrees(np.arctan(np.hypot(dhx, dhz)))
    assert 2.0 < h.max() < terrain.amplitude
    assert -terrain.amplitude < h.min() < -2.0
    assert slope.max() < 15.0


@pytest.mark.parametrize("a,b,axis", [((0, 0), (1, 0), "x"), ((0, 0), (0, 1), "z"),
                                      ((-1, -3), (0, -3), "x"), ((5, -1), (5, 0), "z")])
def test_no_seams_between_chunks(terrain, a, b, axis):
    res = SPEC.chunk_res
    ma, mb = chunk_mesh(terrain, *a, SPEC), chunk_mesh(terrain, *b, SPEC)
    grid = (res + 1, res + 1)  # rows run along z, columns along x
    pa, pb = ma.pos.reshape(grid + (3,)), mb.pos.reshape(grid + (3,))
    na, nb = ma.normal.reshape(grid + (3,)), mb.normal.reshape(grid + (3,))
    if axis == "x":
        ea, eb, fa, fb = pa[:, -1], pb[:, 0], na[:, -1], nb[:, 0]
    else:
        ea, eb, fa, fb = pa[-1], pb[0], na[-1], nb[0]
    assert np.array_equal(ea, eb)
    assert np.array_equal(fa, fb)


def test_chunk_mesh_lies_on_the_terrain(terrain):
    m = chunk_mesh(terrain, 3, -2, SPEC)
    x, z = m.pos[:, 0], m.pos[:, 2]
    assert np.allclose(m.pos[:, 1], terrain.height(x, z))
    assert np.allclose(m.normal, terrain.normal(x, z))
    size = SPEC.chunk_size_m
    assert x.min() == 3 * size and x.max() == 4 * size
    assert z.min() == -2 * size and z.max() == -1 * size
    res = SPEC.chunk_res
    assert len(m.tris) == 2 * res * res and m.tris.max() == len(m.pos) - 1


def test_chunks_stream_round_the_car(terrain):
    from panda3d.core import NodePath

    root = NodePath("root")
    s = ChunkStreamer(terrain, root, radius=2, builds_per_frame=3, build_ms=math.inf,
                      spec=SPEC)
    assert s.update(10.0, 10.0) == 3
    assert (0, 0) in s.loaded  # the chunk under the car comes first
    s.update(10.0, 10.0, budget=100)
    assert set(s.loaded) == set(chunks_around(10.0, 10.0, 2, SPEC.chunk_size_m)) and len(s.loaded) == 25
    far = 10 * SPEC.chunk_size_m
    s.update(far, 0.0, budget=100)
    assert (0, 0) not in s.loaded and len(s.loaded) == 25
    assert s.root.getNumChildren() == 25


def test_car_settles_across_a_slope(terrain):
    xs = np.arange(-400.0, 400.0, 2.0)
    gx, gz = np.meshgrid(xs, xs)
    _, dhx, dhz = terrain.sample(gx, gz)
    k = int(np.argmax(np.hypot(dhx, dhz)))
    x0, z0 = gx.flat[k], gz.flat[k]
    slope = math.atan(math.hypot(dhx.flat[k], dhz.flat[k]))
    assert math.degrees(slope) > 6.0
    across = math.atan2(-dhx.flat[k], -dhz.flat[k]) + math.pi / 2
    v = Vehicle(load_cosmetics())
    v.reset(x0, z0, across, ground=terrain)
    world = PhysicsWorld()
    world.add(v)
    v.process_input(_inp(handbrake=True))
    for _ in range(180):
        world.step(DT)
    assert np.linalg.norm(v.vel) < 0.05
    assert math.hypot(v.pos[0] - x0, v.pos[2] - z0) < 1.0
    assert all(c > 0.0 for c in v.compression)
    assert v.rot[1, 1] == pytest.approx(math.cos(slope), abs=0.02)  # leans with the hill
    clearance = v.pos[1] - terrain.height(v.pos[0], v.pos[2])
    assert clearance == pytest.approx(v.rest_height(), abs=0.06)


@pytest.mark.parametrize("heading", [0.0, 2.4, 4.0])
def test_car_drives_over_the_hills_without_falling_through(terrain, heading):
    v = Vehicle(load_cosmetics())
    v.reset(0.0, 0.0, heading, ground=terrain)
    world = PhysicsWorld()
    world.add(v)
    start = v.pos.copy()
    for f in range(round(25.0 / DT)):
        # full throttle, a half-second steering flick each way every 4 s
        phase = (f // 30) % 8
        steer = 1.0 if phase == 0 else (-1.0 if phase == 4 else 0.0)
        v.process_input(_inp(steer=steer, throttle=1.0))
        world.step(DT)
        for corner in v.geo.c_points[:4]:  # the body's bottom corners
            x, y, z = v.pos + v.rot @ corner
            depth = terrain.height(x, z) - y
            assert depth < 0.15, f"body corner {depth:.2f} m under the ground at tick {f}"
        assert v.rot[1, 1] > 0.5, f"car on its side at tick {f}"
    assert np.linalg.norm((v.pos - start)[[0, 2]]) > 300.0

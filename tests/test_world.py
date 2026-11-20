"""Biomes (world/biomes.py), regions (world/regions.py) and roads
(world/roads.py) as one drivable World (world/world.py)."""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.car_forces import SURFACE_OFFROAD, SURFACE_ROAD  # noqa: E402
from core.car_profile import load_cosmetics  # noqa: E402
from core.redriver2_physics import PhysicsWorld, Vehicle  # noqa: E402
from world.biomes import load_biome, load_biomes, parse_biome  # noqa: E402
from world.chunks import chunk_mesh, terrain_colors  # noqa: E402
from world.ground import Ground  # noqa: E402
from world.roads import MARKING_LIFT  # noqa: E402
from world.world import World  # noqa: E402

DT = 1.0 / 60.0
SIZE = config.TERRAIN_CHUNK_SIZE


@pytest.fixture(scope="module")
def world():
    return World()


def _inp(steer=0.0, throttle=0.0):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=0.0, handbrake=False,
                           fast_steer=False, analog=True)


def _border_point(world, x=0.0, z=0.0, reach=4000.0):
    """A point on a border between two biomes (owner weight about 0.5),
    along +x from (x, z)."""
    xs = np.arange(x, x + reach, 2.0)
    w = world.border_weight(xs, np.full_like(xs, z))
    k = int(np.argmin(w))
    assert w[k] < 0.55, "no biome border along the probe line"
    return xs[k], z


def _borders(world, n=12, reach=12000.0):
    """Points on biome borders, at most one per pair of biomes, found along
    a fan of probe lines from the origin."""
    found = {}
    for a in np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False):
        s = np.arange(0.0, reach, 4.0)
        x, z = s * np.cos(a), s * np.sin(a)
        w = world.regions.weights(x, z)[0]
        top = np.argsort(w, axis=1)[:, -2:]
        on = np.nonzero(w.max(axis=1) < 0.55)[0]
        for k in on:
            pair = tuple(sorted(top[k].tolist()))
            found.setdefault(pair, (x[k], z[k], a))
    return found


def _road_points(world, n=40):
    """Centreline points of the roads round the origin, away from nodes."""
    out = []
    for r in world.roads.roads_near(-600.0, -600.0, 600.0, 600.0):
        reach = r.half_width + r.shoulder + 30.0
        s = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
        for k in range(len(r.pts) - 1):
            if reach < s[k] < s[-1] - reach:
                out.append((r, k))
    rng = np.random.default_rng(3)
    return [out[i] for i in rng.choice(len(out), min(n, len(out)), replace=False)]


# ------------------------------------------------------------------ biomes
def test_the_world_biomes_load_with_their_rules():
    biomes = load_biomes()
    assert tuple(b.name for b in biomes) == config.WORLD_BIOMES
    plains, desert = biomes[:2]
    kinds = {b.name: b.terrain.noise_type for b in biomes}
    assert kinds["canyon"] == "terraced" and kinds["treacherous"] == "ridged"
    assert {b.name for b in biomes if b.roads.markings} >= {"village", "treacherous"}
    assert desert.terrain.noise_scale_m > plains.terrain.noise_scale_m
    assert desert.roads.grid_spacing_m == 2 * plains.roads.grid_spacing_m
    assert desert.roads.surface == "packed_sand" and plains.roads.surface == "asphalt"
    assert "description" in plains.extra  # fields of its own are kept


def test_missing_rules_take_defaults_and_bad_ones_are_named():
    b = parse_biome({"name": "x", "terrain": {"amplitude_m": 3}, "notes": [1, 2]})
    assert b.terrain.amplitude_m == 3.0 and b.terrain.octaves == 4
    assert b.roads.width_m == 16.0 and b.extra == {"notes": [1, 2]}
    assert b.weight == 1.0 and b.terrain.noise_type == "fbm" and b.roads.markings is None
    for bad, where in [({"terrain": {"amplitude": 3}}, "unknown key"),
                       ({"terrain": {"octaves": 2.5}}, "octaves"),
                       ({"terrain": {"persistence": 1.5}}, "persistence"),
                       ({"terrain": {"colors": {"low": [1, 2, 3]}}}, "colors.low"),
                       ({"roads": {"surface": "lava"}}, "surface"),
                       ({"roads": {"flatten": 1}}, "flatten"),
                       ({"roads": {"jitter_m": 90.0}}, "jitter_m"),
                       ({"terrain": {"noise_type": "voronoi"}}, "noise_type"),
                       ({"terrain": {"terrace_steps": 1}}, "terrace_steps"),
                       ({"roads": {"markings": [1, 1]}}, "markings"),
                       ({"weight": -1}, "weight")]:
        with pytest.raises(ValueError, match=where):
            parse_biome(bad)
    with pytest.raises(ValueError, match="empty definition"):
        load_biome("harbor")  # still a 0-byte stub


# ----------------------------------------------------------------- regions
def test_the_world_is_a_ground_and_paths_agree(world):
    assert isinstance(world, Ground)
    rng = np.random.default_rng(1)
    pts = np.concatenate([rng.uniform(-3000.0, 3000.0, (150, 2)),
                          [p + rng.normal(0.0, 6.0, 2) for p in
                           [world.roads.nearest(x, z)[1] for x, z in
                            rng.uniform(-500.0, 500.0, (150, 2))]]])
    scalar = np.array([world.sample(x, z) for x, z in pts])
    array = np.stack(world.sample(pts[:, 0], pts[:, 1]), axis=1)
    np.testing.assert_allclose(scalar, array, atol=1e-9)
    e = 1e-4  # the analytic gradient is the real slope
    num = np.array([[(world.height(x + e, z) - world.height(x - e, z)) / (2 * e),
                     (world.height(x, z + e) - world.height(x, z - e)) / (2 * e)]
                    for x, z in pts])
    np.testing.assert_allclose(num, scalar[:, 1:], atol=2e-5)


def test_biomes_come_in_regions(world):
    xs = np.arange(-6000.0, 6000.0, 25.0)
    gx, gz = np.meshgrid(xs, xs)
    owner = world.biome_index(gx, gz)
    share = np.bincount(owner.ravel(), minlength=len(world.biomes)) / owner.size
    assert share.min() > 0.04, dict(zip(config.WORLD_BIOMES, share.round(3), strict=True))
    # regions: along a line the owner changes every few hundred metres, not every few
    flips = np.count_nonzero(np.diff(owner, axis=1)) / len(xs)
    assert 4 <= flips <= 30, f"{flips:.1f} biome changes per 12 km"
    w = world.regions.weights(gx.ravel()[:5000], gz.ravel()[:5000])[0]
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_every_biome_border_stays_drivable(world):
    borders = _borders(world)
    assert len(borders) >= 10, f"only {len(borders)} biome pairs met"
    for pair, (x0, z0, a) in borders.items():
        s = np.arange(-300.0, 300.0, 0.5)   # across the border, along the probe
        x, z = x0 + s * np.cos(a), z0 + s * np.sin(a)
        h, dhx, dhz = world.regions.sample(x, z)
        names = [world.biomes[i].name for i in pair]
        assert np.abs(np.diff(h)).max() < 0.2, names   # 0.5 m apart: no step
        assert np.hypot(dhx, dhz).max() < 0.4, names   # under 22 degrees


def test_chunk_colours_follow_the_biome(world):
    for b in world.biomes:
        x, z = world.regions.site_of(b.name)
        key = (math.floor(x / SIZE), math.floor(z / SIZE))
        m = chunk_mesh(world, *key)
        assert (world.border_weight(m.pos[:, 0], m.pos[:, 2]) == 1.0).all()
        t = b.terrain
        want = terrain_colors(m.pos[:, 1] - t.base_height_m, m.normal, t.amplitude_m,
                              t.colors, t.steep_slope_deg)
        np.testing.assert_allclose(m.color, want, atol=1e-9)


# ------------------------------------------------------------------- roads
def test_the_world_is_deterministic():
    a, b = World(), World()
    b.sample(5000.0, -7000.0)  # a different query order must not matter
    b.road_mesh(40, -12)
    for key in [(3, 4), (-2, 0), (0, -5)]:
        ma, mb = chunk_mesh(a, *key), chunk_mesh(b, *key)
        np.testing.assert_array_equal(ma.pos, mb.pos)
        np.testing.assert_array_equal(ma.color, mb.color)
        ra, rb = a.road_mesh(*key), b.road_mesh(*key)
        assert (ra is None) == (rb is None)
        if ra is not None:
            np.testing.assert_array_equal(ra.pos, rb.pos)
    c = World(seed=config.TERRAIN_SEED + 1)
    assert c.spawn() != a.spawn()


def test_chunks_and_road_strips_meet_without_seams(world):
    keys = [(i, k) for i in range(-3, 3) for k in range(-3, 3)]
    meshes = {key: chunk_mesh(world, *key) for key in keys}
    res = config.TERRAIN_CHUNK_RES
    for (i, k), m in meshes.items():
        g = m.pos.reshape(res + 1, res + 1, 3)
        if (i + 1, k) in meshes:
            n = meshes[(i + 1, k)].pos.reshape(res + 1, res + 1, 3)
            np.testing.assert_array_equal(g[:, -1], n[:, 0])
        if (i, k + 1) in meshes:
            n = meshes[(i, k + 1)].pos.reshape(res + 1, res + 1, 3)
            np.testing.assert_array_equal(g[-1], n[0])
    # every road segment in the block is drawn by exactly one chunk, and the
    # strips of neighbouring chunks share their end vertices exactly
    lo, hi = -3 * SIZE, 3 * SIZE
    expected = segments = 0
    for r in world.roads.roads_near(lo, lo, hi, hi):
        mid = 0.5 * (r.pts[:-1] + r.pts[1:])
        inside = np.nonzero((mid >= lo).all(axis=1) & (mid < hi).all(axis=1))[0]
        segments += len(inside)
        dashes = np.count_nonzero(inside % 2 == 0) if r.markings is not None else 0
        expected += 4 * len(inside) + 2 * dashes   # road quads + dashed centre line
    strips = [world.road_mesh(*key) for key in keys]
    strips = [s for s in strips if s is not None]
    assert segments > 50
    assert sum(len(s.tris) for s in strips) == expected
    verts = {}
    for s in strips:
        for p in s.pos:
            verts.setdefault((p[0], p[2]), set()).add(p[1])
    assert all(len(h) == 1 for h in verts.values())


def test_road_strips_sit_on_the_flattened_ground(world):
    for key in [(0, 0), (1, -1), (-2, 2)]:
        s = world.road_mesh(*key)
        if s is None:
            continue
        h = world.height(s.pos[:, 0], s.pos[:, 2])
        lift = s.pos[:, 1] - h - config.ROAD_LIFT_M
        assert np.all((np.abs(lift) < 1e-9) | (np.abs(lift - MARKING_LIFT) < 1e-9))


def test_roads_are_flat_across_and_within_grade_along(world):
    pts = _road_points(world)
    assert len(pts) >= 30
    for r, k in pts:
        x, z = r.pts[k] + 0.5 * np.array([r.dx[k], r.dz[k]])
        h, dhx, dhz, on_road = world.query(x, z)
        assert on_road and world(x, z)[2] == SURFACE_ROAD
        tan = np.array([r.dx[k], r.dz[k]]) * math.sqrt(r.inv_len2[k])
        along = dhx * tan[0] + dhz * tan[1]
        across = -dhx * tan[1] + dhz * tan[0]
        assert abs(along) <= r.max_grade + 1e-9
        assert abs(across) < 1e-6
        # the whole road width is at the profile height
        for f in (-0.9, 0.9):
            side = np.array([-tan[1], tan[0]]) * f * r.half_width
            assert world.height(x + side[0], z + side[1]) == pytest.approx(h, abs=1e-6)
    for r in world.roads.roads_near(-600.0, -600.0, 600.0, 600.0):
        s = np.hypot(r.dx, r.dz)
        assert np.all(np.abs(r.dh) <= r.max_grade * s + 1e-9), r.key


def test_off_road_is_offroad(world):
    x, z, _ = world.spawn()
    for dx, dz in [(60.0, 0.0), (0.0, 60.0), (-45.0, 30.0)]:
        if world.roads.nearest(x + dx, z + dz)[0] > 20.0:
            assert world(x + dx, z + dz)[2] == SURFACE_OFFROAD


def test_sparse_biomes_keep_every_nth_grid_line(world):
    roads = world.roads
    assert roads.every["desert"] == 2 and roads.every["treacherous"] == 3
    seen = dict.fromkeys(roads.every, 0)
    for i in range(-15, 15):
        for j in range(-15, 15):
            for kind, mid, line in [("h", ((i + 0.5) * roads.base, j * roads.base), j),
                                    ("v", (i * roads.base, (j + 0.5) * roads.base), i)]:
                name = world.regions.biome_at(*mid).name
                exists = roads.road(kind, i, j) is not None
                assert exists == (line % roads.every[name] == 0)
                seen[name] += exists
    assert seen["plains"] > 0 and seen["desert"] > 0


# -------------------------------------------------------------------- car
def _drive(world, x, z, heading, steer_fn, throttle_fn, ticks):
    v = Vehicle(load_cosmetics())
    v.reset(x, z, heading, ground=world)
    pw = PhysicsWorld()
    pw.add(v)
    start, on_road = v.pos.copy(), 0
    for f in range(ticks):
        v.process_input(_inp(steer=steer_fn(v, f), throttle=throttle_fn(v)))
        pw.step(DT)
        for corner in v.geo.c_points[:4]:
            cx, cy, cz = v.pos + v.rot @ corner
            depth = world.height(cx, cz) - cy
            assert depth < 0.15, f"body corner {depth:.2f} m under the ground at tick {f}"
        assert v.rot[1, 1] > 0.5, f"car on its side at tick {f}"
        on_road += world(v.pos[0], v.pos[2])[2] == SURFACE_ROAD
    return float(np.linalg.norm((v.pos - start)[[0, 2]])), on_road / ticks


def _start(world, biome):
    """(x, z, heading) on the road nearest the middle of a ``biome`` region."""
    return world.spawn(*world.regions.site_of(biome)) if biome else world.spawn()


@pytest.mark.parametrize("biome", [None, *config.WORLD_BIOMES[2:]])
def test_car_follows_the_roads_for_25_seconds(world, biome):
    x, z, heading = _start(world, biome)
    cruise = 16.0 if biome == "jungle" else 22.0  # narrow twisty lanes: ~58 km/h
    dist, on_road = _drive(
        world, x, z, heading,
        lambda v, f: world.road_steer(v.pos[0], v.pos[2], v.heading),
        lambda v: 1.0 if np.linalg.norm(v.vel) < cruise else 0.0, round(25.0 / DT))
    assert dist > 250.0
    assert on_road > 0.9, f"on the road {on_road:.0%} of the time"


@pytest.mark.parametrize("biome, turn", [(None, math.pi / 2), (None, -math.pi / 2),
                                         (None, 0.8), ("canyon", math.pi / 2),
                                         ("treacherous", -math.pi / 2),
                                         ("jungle", math.pi / 2)])
def test_car_crosses_roads_flat_out_for_25_seconds(world, biome, turn):
    x, z, heading = _start(world, biome)

    def flicks(v, f):  # a half-second steering flick each way every 4 s
        phase = (f // 30) % 8
        return 1.0 if phase == 0 else (-1.0 if phase == 4 else 0.0)

    dist, on_road = _drive(world, x, z, heading + turn, flicks, lambda v: 1.0,
                           round(25.0 / DT))
    assert dist > 300.0
    assert 0.0 < on_road < 0.5  # it crossed road and open ground

"""Save file (world/save.py), session (world/session.py) and the world map
export (world/worldmap.py)."""

import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from world.save import SaveState, encode_chunks, read_save, write_save  # noqa: E402
from world.session import Session  # noqa: E402
from world.spec import world_spec  # noqa: E402
from world.worldmap import TileCache, chunk_tile, export_world, minimap_image  # noqa: E402
from worldgen.chunk import chunk_mesh  # noqa: E402

DT = 1.0 / 60.0
SIZE = world_spec(biomes=()).chunk_size_m


def _inp(steer=0.0, throttle=0.0):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=0.0, handbrake=False,
                           fast_steer=False, analog=True)


def _drive(s, seconds):
    w = s.world
    for _ in range(round(seconds / DT)):
        v = s.vehicle
        steer = w.road_steer(v.pos[0], v.pos[2], v.heading)
        s.step(_inp(steer, 1.0 if np.linalg.norm(v.vel) < 22.0 else 0.0), DT)


@pytest.fixture(scope="module")
def session():
    return Session()


# -------------------------------------------------------------------- save
def test_save_round_trips_and_stays_compact(tmp_path):
    blob = {(x, z) for x in range(-20, 25) for z in range(-7, 9) if (x * x + z * z) < 300}
    state = SaveState(seed=4242, discovered=blob, biomes={"plains", "desert"},
                      car=(12.5, -40.25, 1.2345))
    path = tmp_path / "world.json"
    write_save(path, state)
    back = read_save(path)
    assert back == state
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["seed"] == 4242
    # one run per chunk row: 16 rows of up to 35 chunks
    assert data["discovered"] == encode_chunks(blob)
    assert len(data["discovered"]) == 16 and all(len(r) == 1 for _, r in data["discovered"])
    assert read_save(tmp_path / "missing.json") is None


def test_a_failed_write_leaves_the_old_save(tmp_path, monkeypatch):
    path = tmp_path / "world.json"
    write_save(path, SaveState(seed=1, discovered={(0, 0)}))
    before = path.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(json, "dump", boom)
    with pytest.raises(OSError, match="disk full"):
        write_save(path, SaveState(seed=2))
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["world.json"]  # no temp left


def test_a_session_resumes_its_save(tmp_path):
    path = tmp_path / "world.json"
    a = Session(save_path=path, seed=77)
    _drive(a, 3.0)
    a.save()
    b = Session(save_path=path)
    assert b.seed == 77 and b.save_path == path
    assert a.explorer.discovered <= b.explorer.discovered
    np.testing.assert_allclose(b.vehicle.pos[[0, 2]], a.vehicle.pos[[0, 2]], atol=1e-3)
    assert b.vehicle.heading == pytest.approx(a.vehicle.heading, abs=1e-4)


def test_seed_override_and_new_never_overwrite_another_save(tmp_path):
    path = tmp_path / "world.json"
    write_save(path, SaveState(seed=9, discovered={(500, 500)}))
    before = path.read_bytes()
    s = Session(save_path=path, seed=5)
    assert s.seed == 5 and (500, 500) not in s.explorer.discovered
    assert s.save_path == tmp_path / "world_2.json"
    s.save()
    assert path.read_bytes() == before and read_save(s.save_path).seed == 5
    n = Session(save_path=path, new=True)
    assert n.save_path == tmp_path / "world_3.json" and (500, 500) not in n.explorer.discovered
    same = Session(save_path=path, seed=9)  # the save's own seed resumes it
    assert same.save_path == path and (500, 500) in same.explorer.discovered


def test_offscreen_sessions_write_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Session()
    s.set_paused(True)
    assert s.save() is False and list(tmp_path.iterdir()) == []


# ------------------------------------------------------------- discovery
def test_discovered_chunks_grow_as_the_car_drives():
    s = Session(seed=config.TERRAIN_SEED)
    first = set(s.explorer.discovered)
    r = config.TERRAIN_VIEW_CHUNKS
    assert len(first) == (2 * r + 1) ** 2
    _drive(s, 12.0)
    v = s.vehicle
    here = (math.floor(v.pos[0] / SIZE), math.floor(v.pos[2] / SIZE))
    assert first < s.explorer.discovered and here in s.explorer.discovered
    assert len(s.explorer.discovered) > len(first) + 2 * r
    assert s.explorer.biomes <= {b.name for b in s.world.biomes} and s.explorer.biomes


# ----------------------------------------------------------------- pause
def test_pausing_freezes_the_physics(session):
    s = session
    s.restart()
    for _ in range(60):
        s.step(_inp(0.0, 1.0), DT)
    s.set_paused(True)
    state = [a.copy() for a in (s.vehicle.pos, s.vehicle.vel, s.vehicle.quat, s.vehicle.avel)]
    found = set(s.explorer.discovered)
    assert sum(s.step(_inp(1.0, 1.0), DT) for _ in range(120)) == 0
    for a, b in zip(state, (s.vehicle.pos, s.vehicle.vel, s.vehicle.quat, s.vehicle.avel),
                    strict=True):
        np.testing.assert_array_equal(a, b)
    assert s.explorer.discovered == found
    s.set_paused(False)
    s.step(_inp(0.0, 1.0), DT)
    assert not np.array_equal(state[0], s.vehicle.pos)


def test_restart_respawns_on_the_road(session):
    session.vehicle.reset(900.0, -300.0, 0.0, ground=session.world)
    session.restart()
    x, z, heading = session.world.spawn()
    assert session.vehicle.pos[0] == pytest.approx(x) and session.vehicle.pos[2] == pytest.approx(z)
    assert session.world.query(x, z)[3]


def test_a_seed_change_rebuilds_the_same_world_as_a_fresh_start(tmp_path):
    path = tmp_path / "world.json"
    s = Session(save_path=path, seed=11)
    _drive(s, 2.0)
    s.save()
    old = read_save(path)
    s.change_seed(42)
    fresh = Session(seed=42)
    assert s.seed == 42 and s.world.regions.seed == 42
    assert s.explorer.discovered == fresh.explorer.discovered
    np.testing.assert_array_equal(s.vehicle.pos, fresh.vehicle.pos)
    for key in [(0, 0), (2, -3)]:
        np.testing.assert_array_equal(chunk_mesh(s.world, *key, s.world.spec).pos,
                                      chunk_mesh(fresh.world, *key, fresh.world.spec).pos)
    assert read_save(path).seed == 42
    assert read_save(tmp_path / "world_11.json") == old  # the old world was kept


# ------------------------------------------------------------------- maps
def test_export_is_one_png_of_the_discovered_area(session, tmp_path):
    from PIL import Image

    world = session.world
    found = {(0, 0), (1, 0), (0, 1), (3, 2)}
    path, size = export_world(world, found, tmp_path / "w.png")
    px = round(SIZE / config.EXPORT_M_PER_PX)
    assert size == (4 * px, 3 * px)
    img = np.asarray(Image.open(path).convert("RGB"))
    assert img.shape == (3 * px, 4 * px, 3)
    # north-up: chunk (cx, cz) sits at column cx - 0, row 2 - cz
    np.testing.assert_array_equal(img[2 * px:, :px], chunk_tile(world, 0, 0, px)[0])
    np.testing.assert_array_equal(img[:px, 3 * px:], chunk_tile(world, 3, 2, px)[0])
    assert (img[:px, :px] == img[0, 0]).all()   # (0, 2) undiscovered: one flat colour


def test_minimap_is_north_up_round_the_car(session):
    world = session.world
    cache = TileCache(world, 16)
    keys = [(i, k) for i in range(-3, 3) for k in range(-3, 3)]
    cache.build(keys, len(keys))
    img = minimap_image(cache, (0.0, 0.0), set(keys), set(keys), 16, 64, grid=False)
    assert img.shape == (64, 64, 3)
    # the centre pixel's right/top neighbours are chunk (0, 0)'s south-west corner
    np.testing.assert_array_equal(img[31, 32], cache.get((0, 0))[0][15, 0])
    np.testing.assert_array_equal(img[32, 31], cache.get((-1, -1))[0][0, 15])

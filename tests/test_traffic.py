"""Lane paths (world.roads.lanes) and traffic (world.traffic)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.car_forces import SURFACE_ROAD  # noqa: E402
from world.roads.lanes import LaneMap, connector  # noqa: E402
from world.traffic import Car, Traffic  # noqa: E402
from world.world import World  # noqa: E402


@pytest.fixture(scope="module")
def world():
    return World(seed=7)


@pytest.fixture(scope="module")
def lanes(world):
    x, z, _ = world.spawn(*world.regions.site_of("british_town"))
    return LaneMap(world.roads), x, z


def test_lanes_drive_on_the_right_of_the_road(world, lanes):
    lm, x, z = lanes
    found = lm.lanes_near(x, z, 250.0)
    assert len(found) > 20
    for lane in found:
        for p in lane.pts[::5]:
            assert world(p[0], p[2])[2] == SURFACE_ROAD
        mid = len(lane.pts) // 2
        p, d = lane.at(lane.s[mid] + 0.01)
        _, c, _, r = lm.net.nearest(p[0], p[2])
        off = np.array([p[0] - c[0], p[2] - c[1]])
        right = np.array([d[2], -d[0]])
        if r.key == lane.key[0]:            # not where another road is nearer
            assert off @ right > 0.3 * r.half_width


def test_exits_leave_the_node_the_lane_ends_at(lanes):
    lm, x, z = lanes
    turns = 0
    for lane in lm.lanes_near(x, z, 250.0):
        exits = lm.exits(lane)
        assert exits
        for e in exits:
            gap = math.hypot(e.pts[0, 0] - lane.pts[-1, 0], e.pts[0, 2] - lane.pts[-1, 2])
            assert gap < 2.5 * 16.0
            if e.key[0] == lane.key[0]:
                assert len(exits) == 1         # U-turns only at dead ends
            else:
                turns += 1
            j = connector(lane, e)
            assert np.allclose(j.pts[0], lane.pts[-1]) and np.allclose(j.pts[-1], e.pts[0])
            assert j.length < 3.0 * gap + 10.0
    assert turns > 20


# ------------------------------------------------------------------ traffic
def _traffic(world, lanes, seconds=20.0, seed=3):
    lm, x, z = lanes
    t = Traffic(world, seed=seed)
    dt = 1.0 / 30.0
    track = {}
    for _ in range(round(seconds / dt)):
        t.step(dt, x, z)
        for c in t.cars:
            track.setdefault(c.id, []).append((c.pose()[0], c.lane))
    return t, track, x, z


def test_traffic_spawns_on_the_road_round_the_player_up_to_the_cap(world, lanes):
    t, track, x, z = _traffic(world, lanes)
    assert len(t.cars) == config.TRAFFIC_MAX
    for steps in track.values():
        p = steps[0][0]
        d = math.hypot(p[0] - x, p[2] - z)
        assert config.TRAFFIC_SPAWN_MIN_M <= d <= config.TRAFFIC_SPAWN_M
        for p, _ in steps[::10]:
            assert world(p[0], p[2])[2] == SURFACE_ROAD


def test_traffic_follows_lanes_through_junctions_without_jumps(world, lanes):
    _, track, _, _ = _traffic(world, lanes)
    changed = 0
    for steps in track.values():
        pos = np.array([p for p, _ in steps])
        step = np.hypot(np.diff(pos[:, 0]), np.diff(pos[:, 2]))
        assert step.max() < 1.2 * config.TRAFFIC_SPEED_MS / 30.0 + 0.05
        changed += len({id(lane) for _, lane in steps}) > 2   # lane, junction, lane
    assert changed >= 3


def test_traffic_stops_behind_the_car_ahead(world, lanes):
    lm, x, z = lanes
    lane = max(lm.lanes_near(x, z, 250.0), key=lambda ln: ln.length)
    t = Traffic(world, seed=1, max_cars=0)
    lead = Car(1, lane, 0.8 * lane.length, 0.0, 0.0, (1.0, 1.0, 1.0))
    back = Car(2, lane, 0.6 * lane.length, 12.0, 12.0, (1.0, 1.0, 1.0))
    t.cars = [lead, back]
    px, _, pz = lane.at(0.5 * lane.length)[0]      # the player stands by, so none despawn
    for _ in range(round(30.0 * 30)):
        t.step(1.0 / 30.0, px, pz)
    assert back.speed < 0.05
    assert config.TRAFFIC_GAP_M - 1.0 < lead.s - back.s < config.TRAFFIC_GAP_M + 3.0


def test_traffic_picks_random_exits_not_u_turns(world, lanes):
    lm, x, z = lanes
    lane = next(ln for ln in lm.lanes_near(x, z, 250.0) if len(lm.exits(ln)) >= 2)
    t = Traffic(world, seed=5, max_cars=0)
    picks = {t._pick_next(Car(1, lane, lane.length, 0.0, 0.0, (1, 1, 1))).to.key
             for _ in range(40)}
    assert len(picks) >= 2 and all(k[0] != lane.key[0] for k in picks)

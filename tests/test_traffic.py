"""Lane paths (world.roads.lanes) and traffic (world.traffic)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.car_forces import SURFACE_ROAD  # noqa: E402
from world.roads.lanes import LaneMap, connector  # noqa: E402
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

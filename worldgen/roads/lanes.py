"""Lane paths over the road network, for traffic (world.traffic).

Every Road carries two lanes, one each way, driven on the right: the road's
centre line pushed LANE_SHARE x the half width to the right (mitred through
bends), heights from the road's profile.  A lane runs node to node.  At its
end node it may go on into any lane leaving that node on another road, or
turn back on its own road at a dead end; ``connector`` joins the two with a
short curve through the junction.
"""

import math
from dataclasses import dataclass, field

import numpy as np

LANE_SHARE = 0.5          # lane centre: this share of the half width off the middle
CONNECT_STEP_M = 2.0      # connector curve sampling


@dataclass(eq=False)
class Lane:
    """A drivable polyline: pts (n, 3) as x, height, z; s its arc lengths."""

    key: tuple                 # (road key, forward) or ("join", from key, to key)
    pts: np.ndarray
    end_node: tuple | None = None   # node ref the lane ends at (None: a connector)
    end_xz: tuple | None = None     # that node's position
    to: "Lane | None" = None        # a connector's next lane
    s: np.ndarray = field(init=False)

    def __post_init__(self):
        step = np.hypot(np.diff(self.pts[:, 0]), np.diff(self.pts[:, 2]))
        self.s = np.concatenate([[0.0], np.cumsum(step)])

    @property
    def length(self):
        return float(self.s[-1])

    def at(self, s):
        """(position (3,), unit direction (3,)) at arc length s (clamped)."""
        s = min(max(s, 0.0), self.length)
        k = min(int(np.searchsorted(self.s, s, side="right")) - 1, len(self.pts) - 2)
        seg = self.pts[k + 1] - self.pts[k]
        span = max(self.s[k + 1] - self.s[k], 1e-9)
        d = seg / max(float(np.linalg.norm(seg)), 1e-9)
        return self.pts[k] + seg * ((s - self.s[k]) / span), d


def _offset(xz, h, width):
    """Polyline xz (n, 2) pushed ``width`` to its right (x east, z north)."""
    seg = np.diff(xz, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    tan = np.concatenate([seg[:1], seg[:-1] + seg[1:], seg[-1:]])
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    mitre = 1.0 / np.maximum(np.einsum("ij,ij->i", tan, np.concatenate([seg, seg[-1:]])), 0.5)
    right = np.stack([tan[:, 1], -tan[:, 0]], axis=1) * (width * mitre)[:, None]
    p = xz + right
    return np.stack([p[:, 0], h, p[:, 1]], axis=1)


def road_lane(road, forward):
    """The lane along ``road`` (forward: from its first node to its last)."""
    xz, h = (road.pts, road.h) if forward else (road.pts[::-1], road.h[::-1])
    end = road.key[3:] if forward else road.key[:3]
    return Lane((road.key, forward), _offset(xz, h, LANE_SHARE * road.half_width), end,
                (float(xz[-1, 0]), float(xz[-1, 1])))


def connector(a, b):
    """A lane from the end of lane a to the start of lane b: a quadratic
    curve whose control point is where their end and start lines cross
    (the midpoint when they do not)."""
    p0, p2 = a.pts[-1], b.pts[0]
    d0 = a.pts[-1] - a.pts[-2]
    d2 = b.pts[1] - b.pts[0]
    den = d0[0] * d2[2] - d0[2] * d2[0]
    p1 = 0.5 * (p0 + p2)
    if abs(den) > 1e-6:
        t = ((p2[0] - p0[0]) * d2[2] - (p2[2] - p0[2]) * d2[0]) / den
        u = ((p2[0] - p0[0]) * d0[2] - (p2[2] - p0[2]) * d0[0]) / den
        gap = float(np.hypot(p2[0] - p0[0], p2[2] - p0[2]))
        if t > 0.0 and u < 0.0 and t * np.hypot(d0[0], d0[2]) < 2.0 * gap + 5.0:
            p1 = p0 + t * d0
            p1[1] = 0.5 * (p0[1] + p2[1])
    n = max(2, math.ceil(float(np.linalg.norm(p2 - p0)) / CONNECT_STEP_M))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t * t * p2
    return Lane(("join", a.key, b.key), pts, to=b)


class LaneMap:
    """Lanes of a world.roads.RoadNetwork, made on demand and cached."""

    def __init__(self, network):
        self.net = network
        self._lanes = {}

    def lane(self, road, forward):
        key = (road.key, forward)
        hit = self._lanes.get(key)
        if hit is None:
            if len(self._lanes) > 4000:
                self._lanes.clear()
            hit = self._lanes[key] = road_lane(road, forward)
        return hit

    def lanes_near(self, x, z, r):
        """Both lanes of every road within r of (x, z), in a fixed order."""
        return [self.lane(road, f) for road in self.net.roads_near(x - r, z - r, x + r, z + r)
                for f in (True, False)]

    def exits(self, lane):
        """Lanes leaving the node ``lane`` ends at, on other roads; at a dead
        end, the way back along its own road."""
        node = lane.end_node
        road_key = lane.key[0]
        x, z = lane.end_xz
        near = self.net.roads_near(x - 1.0, z - 1.0, x + 1.0, z + 1.0)
        out = []
        for road in near:
            if road.key == road_key:
                continue
            if road.key[:3] == node:
                out.append(self.lane(road, True))
            elif road.key[3:] == node:
                out.append(self.lane(road, False))
        if not out:
            for road in near:
                if road.key == road_key:
                    out.append(self.lane(road, not lane.key[1]))
        return out

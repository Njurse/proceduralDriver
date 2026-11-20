"""The drivable world: worldgen.query.WorldGround built from config, plus
the game's helpers on it (the road-following autopilot, spawn points) and
the ground texture the chunks draw with.
"""

import math

import numpy as np

import config
from world.spec import world_spec
from worldgen.query import WorldGround


class World(WorldGround):
    def __init__(self, biomes=None, seed=None):
        super().__init__(world_spec(seed, biomes))
        first = self.biomes[0].terrain           # one ground texture for every chunk
        self.detail_texture = (config.ASSETS / first.detail_texture).as_posix()

    def road_steer(self, x, z, heading, lookahead=15.0, gain=2.5):
        """Steering (-1 left .. 1 right) that follows the nearest road the
        way the car already points: aim at a point ``lookahead`` metres on
        along the road's bends (its end node at most, where the next road
        takes over).  An autopilot for screenshots and tests."""
        _, p, tan, road = self.roads.nearest(x, z)
        k = int(road.terms(np.array([float(x)]), np.array([float(z)]))[6][0])
        pts = road.pts
        if tan[0] * math.sin(heading) + tan[1] * math.cos(heading) < 0.0:
            pts, k = pts[::-1], len(pts) - 2 - k
        aim, left = p, lookahead
        for q in pts[k + 1:]:
            step = float(np.hypot(*(q - aim)))
            if step >= left:
                aim = aim + (q - aim) * (left / step)
                break
            aim, left = q, left - step
        tx, tz = aim - np.array([x, z])
        err = (math.atan2(tx, tz) - heading + math.pi) % (2.0 * math.pi) - math.pi
        return max(-1.0, min(1.0, gain * err))

    def spawn(self, x=0.0, z=0.0, along=0.35):
        """(x, z, heading) on the road nearest (x, z): ``along`` of the way
        down it, facing along it (heading as in core.redriver2_physics)."""
        _, _, _, road = self.roads.nearest(x, z)
        k = min(int(along * (len(road.pts) - 1)), len(road.pts) - 2)
        px, pz = road.pts[k]
        return float(px), float(pz), math.atan2(road.dx[k], road.dz[k])

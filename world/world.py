"""The drivable world: biome regions with a road grid flattened into them.

``World`` is the Ground the car and the chunk streamer share.  Its height
field is world.regions.Regions (two biomes blended in blobs) with
world.roads.RoadNetwork cut and filled in; the surface is SURFACE_ROAD on
the tarmac (or sand track) and SURFACE_OFFROAD everywhere else.
"""

import math

import numpy as np

import config
from core.car_forces import SURFACE_OFFROAD, SURFACE_ROAD
from world.chunks import ChunkMesh
from world.ground import SampledGround
from world.regions import Regions
from world.roads import RoadNetwork


class World(SampledGround):
    def __init__(self, biomes=None, seed=None):
        self.regions = Regions(biomes, seed)
        self.roads = RoadNetwork(self.regions)
        self.biomes = self.regions.biomes
        self.detail_texture = self.regions.detail_texture
        self.detail_tile_m = self.regions.detail_tile_m

    def query(self, x, z):
        """(height, dh/dx, dh/dz, on_road) for scalars or arrays."""
        h, hx, hz = self.regions.sample(x, z)
        if np.ndim(x) == 0 and np.ndim(z) == 0:
            x, z = float(x), float(z)
            roads = self.roads.roads_near(x, z, x, z)
            if not roads:
                return h, hx, hz, False
            out = self.roads.blend(roads, np.array([x]), np.array([z]), np.array([h]),
                                   np.array([hx]), np.array([hz]))
            return float(out[0][0]), float(out[1][0]), float(out[2][0]), bool(out[3][0])
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        x, z = x.ravel(), z.ravel()
        h, hx, hz = (np.broadcast_to(a, shape).ravel() for a in (h, hx, hz))
        roads = self.roads.roads_near(x.min(), z.min(), x.max(), z.max())
        out = self.roads.blend(roads, x, z, h, hx, hz)
        return tuple(np.reshape(a, shape) for a in out)

    def sample(self, x, z):
        return self.query(x, z)[:3]

    def surface(self, x, z):
        return SURFACE_ROAD if np.all(self.query(x, z)[3]) else SURFACE_OFFROAD

    def __call__(self, x, z):
        h, dhx, dhz, on_road = self.query(x, z)
        inv = 1.0 / math.sqrt(dhx * dhx + 1.0 + dhz * dhz)
        return (h, np.array([-dhx * inv, inv, -dhz * inv]),
                SURFACE_ROAD if on_road else SURFACE_OFFROAD)

    def colors(self, x, z, h, n):
        return self.regions.colors(x, z, h, n)

    def biome_weight(self, x, z):
        """0 in the first biome, 1 in the second, between in the blend."""
        return self.regions.weight(x, z)[0]

    def road_mesh(self, ci, ck, size=None):
        """Road strips for chunk (ci, ck), or None when it has no road."""
        size = float(config.TERRAIN_CHUNK_SIZE if size is None else size)
        out = self.roads.strips(self, ci * size, ck * size, (ci + 1) * size, (ck + 1) * size,
                                config.ROAD_LIFT_M)
        if out is None:
            return None
        pos, normal, color, uv, tris = out
        return ChunkMesh(pos=pos, normal=normal, color=color,
                         uv=uv / config.ROAD_DETAIL_TILE_M, tris=tris)

    def road_steer(self, x, z, heading, lookahead=15.0, gain=2.5):
        """Steering (-1 left .. 1 right) that follows the nearest road the
        way the car already points: aim at a point ``lookahead`` metres down
        the road.  An autopilot for screenshots and tests."""
        _, p, tan, _ = self.roads.nearest(x, z)
        if tan[0] * math.sin(heading) + tan[1] * math.cos(heading) < 0.0:
            tan = -tan
        tx, tz = p + tan * lookahead - np.array([x, z])
        err = (math.atan2(tx, tz) - heading + math.pi) % (2.0 * math.pi) - math.pi
        return max(-1.0, min(1.0, gain * err))

    def spawn(self, x=0.0, z=0.0, along=0.35):
        """(x, z, heading) on the road nearest (x, z): ``along`` of the way
        down it, facing along it (heading as in core.redriver2_physics)."""
        _, _, _, road = self.roads.nearest(x, z)
        k = min(int(along * (len(road.pts) - 1)), len(road.pts) - 2)
        px, pz = road.pts[k]
        return float(px), float(pz), math.atan2(road.dx[k], road.dz[k])

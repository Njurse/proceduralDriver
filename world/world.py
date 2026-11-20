"""The drivable world: biome regions with a road grid flattened into them.

``World`` is the Ground the car and the chunk streamer share.  Its height
field is worldgen.regions.Regions (the biomes in blended regions) with
worldgen.roads.RoadNetwork cut and filled in; the surface is SURFACE_ROAD on
the tarmac (or sand track) and SURFACE_OFFROAD everywhere else.
"""

import math

import numpy as np

import config
from world.chunks import ChunkMesh
from world.spec import world_spec
from worldgen.ground import SURFACE_OFFROAD, SURFACE_ROAD, SampledGround
from worldgen.regions import Regions
from worldgen.roads import RoadNetwork


class World(SampledGround):
    def __init__(self, biomes=None, seed=None):
        self.spec = world_spec(seed, biomes)
        self.regions = Regions(self.spec)
        self.roads = RoadNetwork(self.regions)
        self.biomes = self.regions.biomes
        first = self.biomes[0].terrain           # one ground texture for every chunk
        self.detail_texture = (config.ASSETS / first.detail_texture).as_posix()
        self.detail_tile_m = first.detail_tile_m

    def query(self, x, z):
        """(height, dh/dx, dh/dz, on_road) for scalars or arrays."""
        h, hx, hz = self.regions.sample(x, z)
        if np.ndim(x) == 0 and np.ndim(z) == 0:
            x, z = float(x), float(z)
            return self.roads.query_point(x, z, h, hx, hz)
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

    def biome_index(self, x, z):
        """Index into ``biomes`` of the biome owning each point (same shape)."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        return self.regions.biome_index(x.ravel(), z.ravel()).reshape(x.shape)

    def border_weight(self, x, z):
        """The owning biome's weight: 1 inside a region, about 0.5 on a
        border between two (same shape as x)."""
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        return self.regions.weights(x.ravel(), z.ravel())[0].max(axis=1).reshape(x.shape)

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

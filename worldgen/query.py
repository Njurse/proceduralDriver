"""The generated world as one Ground: biome regions with the road network
cut and filled in.

``WorldGround(spec)`` answers the physics query (height, slope, surface) and
hands out chunk road strips.  Its height field is worldgen.regions.Regions
(the biomes in blended regions) with worldgen.roads.RoadNetwork flattened
in; the surface is SURFACE_ROAD on the tarmac (or sand track) and
SURFACE_OFFROAD everywhere else.
"""

import math

import numpy as np

from worldgen.chunk import ChunkMesh
from worldgen.ground import SURFACE_OFFROAD, SURFACE_ROAD, SampledGround
from worldgen.regions import Regions
from worldgen.roads import RoadNetwork


class WorldGround(SampledGround):
    def __init__(self, spec):
        self.spec = spec
        self.regions = Regions(spec)
        self.roads = RoadNetwork(spec, self.regions)
        self.biomes = self.regions.biomes
        self.detail_tile_m = self.biomes[0].terrain.detail_tile_m   # uv scale of every chunk

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
        size = float(self.spec.chunk_size_m if size is None else size)
        out = self.roads.strips(self, ci * size, ck * size, (ci + 1) * size, (ck + 1) * size,
                                self.spec.roads.lift_m)
        if out is None:
            return None
        pos, normal, color, uv, tris = out
        return ChunkMesh(pos=pos, normal=normal, color=color,
                         uv=uv / self.spec.roads.detail_tile_m, tris=tris)

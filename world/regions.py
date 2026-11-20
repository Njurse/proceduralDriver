"""Region splotching: which biome owns each patch of ground.

A low-frequency Perlin field f(x, z) (wavelength config.REGION_SCALE_M)
splits the world between two biomes: f < -band is all the first biome,
f > +band all the second, and a smoothstep blends them in between.  Blobs
come out a few hundred metres to a kilometre across.  The blend mixes the two
biomes' height fields, so it has no step, and its gradient includes the
weight's own gradient, so normals stay exact across the band:

    h  = (1 - w) hA + w hB
    dh = (1 - w) dhA + w dhB + (hB - hA) dw

Physics and chunk meshes both read ``sample``, so they cannot disagree.
"""

import numpy as np

import config
from core.car_forces import SURFACE_OFFROAD
from world.biomes import load_biomes
from world.chunks import terrain_colors
from world.ground import SampledGround
from world.terrain import Terrain


def biome_terrain(biome, seed):
    """The fBm height field a biome's terrain rules describe."""
    t = biome.terrain
    return Terrain(seed=seed, scale=t.noise_scale_m, octaves=t.octaves,
                   amplitude=t.amplitude_m, persistence=t.persistence,
                   lacunarity=t.lacunarity, base_height=t.base_height_m)


class Regions(SampledGround):
    """Two biomes in seeded blobs, blended over a band.  No roads."""

    def __init__(self, biomes=None, seed=None, scale_m=None, band=None):
        self.biomes = tuple(load_biomes() if biomes is None else biomes)
        if len(self.biomes) != 2:
            raise ValueError(f"Regions blends two biomes, got {len(self.biomes)}")
        self.seed = config.TERRAIN_SEED if seed is None else int(seed)
        self.scale_m = float(config.REGION_SCALE_M if scale_m is None else scale_m)
        self.band = float(config.REGION_BAND if band is None else band)
        self.terrains = tuple(biome_terrain(b, self.seed + 1 + k)
                              for k, b in enumerate(self.biomes))
        self._field = Terrain(seed=self.seed + 100, scale=self.scale_m, octaves=2,
                              amplitude=1.0, persistence=0.5)
        first = self.biomes[0].terrain
        self.detail_texture = (config.ASSETS / first.detail_texture).as_posix()
        self.detail_tile_m = first.detail_tile_m

    def weight(self, x, z):
        """(w, dw/dx, dw/dz): 0 in the first biome, 1 in the second."""
        f, fx, fz = self._field.sample(x, z)
        inv = 0.5 / self.band
        t = (f + self.band) * inv
        if np.ndim(t) == 0:
            if t <= 0.0:
                return 0.0, 0.0, 0.0
            if t >= 1.0:
                return 1.0, 0.0, 0.0
        else:
            t = np.clip(t, 0.0, 1.0)   # dw is 0 at both clip ends
        dw = 6.0 * t * (1.0 - t) * inv
        return t * t * (3.0 - 2.0 * t), dw * fx, dw * fz

    def biome_at(self, x, z):
        """The biome that owns (x, z): the one weighted at least a half."""
        return self.biomes[1 if self.weight(x, z)[0] >= 0.5 else 0]

    def sample(self, x, z):
        w, wx, wz = self.weight(x, z)
        if np.ndim(w) == 0 and w in (0.0, 1.0):
            return self.terrains[int(w)].sample(x, z)
        ha, ax, az = self.terrains[0].sample(x, z)
        hb, bx, bz = self.terrains[1].sample(x, z)
        dh = hb - ha
        return ha + w * dh, ax + w * (bx - ax) + dh * wx, az + w * (bz - az) + dh * wz

    def surface(self, x, z):
        return SURFACE_OFFROAD

    def colors(self, x, z, h, n):
        """RGBA per vertex: each biome's ramp, blended by the region weight."""
        w = np.asarray(self.weight(x, z)[0])[..., None]
        ramps = [terrain_colors(np.asarray(h) - b.terrain.base_height_m, n,
                                b.terrain.amplitude_m, b.terrain.colors,
                                b.terrain.steep_slope_deg) for b in self.biomes]
        return ramps[0] + (ramps[1] - ramps[0]) * w

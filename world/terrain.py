"""Seeded rolling hills: 2D gradient (Perlin) noise summed in fBm octaves.

``Terrain`` is a world.ground.Ground.  Heights and normals are analytic (the
noise returns its own gradient), so the physics, the chunk meshes and the
normals all agree exactly.  Every function takes scalars (the physics path,
plain Python for speed) or numpy arrays (the mesh path) of x/z in metres.
"""

import math

import numpy as np

import config
from world.ground import SampledGround

_GRADIENTS = 16          # unit gradient directions, evenly spaced
LACUNARITY = 2.0         # frequency step per octave
PERSISTENCE = 0.30       # amplitude step per octave; higher makes small bumps
                         # that throw the car (0.45 rolled it in 4 of 8 test runs)


def _fade(t):
    """Quintic ease 6t^5 - 15t^4 + 10t^3: C2, so normals are continuous."""
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _dfade(t):
    return 30.0 * t * t * (t * (t - 2.0) + 1.0)


class Terrain(SampledGround):
    """height(x, z) = base_height + amplitude * fBm(x / scale, z / scale), in metres.

    ``persistence`` is the amplitude ratio and ``lacunarity`` the frequency
    ratio between octaves; the octave amplitudes are normalised so the field
    stays within about +-amplitude / 2 of base_height.
    """

    def __init__(self, seed=None, scale=None, octaves=None, amplitude=None,
                 persistence=PERSISTENCE, lacunarity=LACUNARITY, base_height=0.0):
        self.seed = config.TERRAIN_SEED if seed is None else int(seed)
        self.scale = float(config.TERRAIN_SCALE if scale is None else scale)
        self.octaves = int(config.TERRAIN_OCTAVES if octaves is None else octaves)
        self.amplitude = float(config.TERRAIN_AMPLITUDE if amplitude is None else amplitude)
        self.persistence = float(persistence)
        self.lacunarity = float(lacunarity)
        self.base_height = float(base_height)

        rng = np.random.default_rng(self.seed)
        perm = rng.permutation(256)
        self._perm_np = np.concatenate([perm, perm]).astype(np.int64)
        self._perm = [int(p) for p in self._perm_np]
        angles = (np.arange(_GRADIENTS) + rng.random()) * (2.0 * math.pi / _GRADIENTS)
        self._gx_np, self._gy_np = np.cos(angles), np.sin(angles)
        self._gx, self._gy = self._gx_np.tolist(), self._gy_np.tolist()
        # per-octave lattice offsets, so octaves don't all vanish at the origin
        self._offsets = (rng.random((self.octaves, 2)) * 256.0).tolist()
        amps = [self.persistence ** k for k in range(self.octaves)]
        norm = self.amplitude / sum(amps)
        self._octaves = [(self.lacunarity ** k / self.scale, a * norm, ox, oy)
                         for k, (a, (ox, oy)) in enumerate(zip(amps, self._offsets,
                                                               strict=True))]

    # ------------------------------------------------------------------ noise
    def _noise(self, x, y):
        """Perlin noise and its gradient at one point: (n, dn/dx, dn/dy)."""
        xf, yf = math.floor(x), math.floor(y)
        fx, fy = x - xf, y - yf
        i, j = int(xf) & 255, int(yf) & 255
        p, gx, gy = self._perm, self._gx, self._gy
        h00 = p[p[i] + j] & 15
        h10 = p[p[i + 1] + j] & 15
        h01 = p[p[i] + j + 1] & 15
        h11 = p[p[i + 1] + j + 1] & 15
        a = gx[h00] * fx + gy[h00] * fy
        b = gx[h10] * (fx - 1.0) + gy[h10] * fy
        c = gx[h01] * fx + gy[h01] * (fy - 1.0)
        d = gx[h11] * (fx - 1.0) + gy[h11] * (fy - 1.0)
        u, v = _fade(fx), _fade(fy)
        du, dv = _dfade(fx), _dfade(fy)
        k = a - b - c + d
        n = a + u * (b - a) + v * (c - a) + u * v * k
        uv = u * v
        dx = (gx[h00] + u * (gx[h10] - gx[h00]) + v * (gx[h01] - gx[h00])
              + uv * (gx[h00] - gx[h10] - gx[h01] + gx[h11]) + du * (b - a + v * k))
        dy = (gy[h00] + u * (gy[h10] - gy[h00]) + v * (gy[h01] - gy[h00])
              + uv * (gy[h00] - gy[h10] - gy[h01] + gy[h11]) + dv * (c - a + u * k))
        return n, dx, dy

    def _noise_np(self, x, y):
        """Vectorised _noise for numpy arrays."""
        xf, yf = np.floor(x), np.floor(y)
        fx, fy = x - xf, y - yf
        i = xf.astype(np.int64) & 255
        j = yf.astype(np.int64) & 255
        p, gx, gy = self._perm_np, self._gx_np, self._gy_np
        h00 = p[p[i] + j] & 15
        h10 = p[p[i + 1] + j] & 15
        h01 = p[p[i] + j + 1] & 15
        h11 = p[p[i + 1] + j + 1] & 15
        a = gx[h00] * fx + gy[h00] * fy
        b = gx[h10] * (fx - 1.0) + gy[h10] * fy
        c = gx[h01] * fx + gy[h01] * (fy - 1.0)
        d = gx[h11] * (fx - 1.0) + gy[h11] * (fy - 1.0)
        u, v = _fade(fx), _fade(fy)
        du, dv = _dfade(fx), _dfade(fy)
        k = a - b - c + d
        n = a + u * (b - a) + v * (c - a) + u * v * k
        uv = u * v
        dx = (gx[h00] + u * (gx[h10] - gx[h00]) + v * (gx[h01] - gx[h00])
              + uv * (gx[h00] - gx[h10] - gx[h01] + gx[h11]) + du * (b - a + v * k))
        dy = (gy[h00] + u * (gy[h10] - gy[h00]) + v * (gy[h01] - gy[h00])
              + uv * (gy[h00] - gy[h10] - gy[h01] + gy[h11]) + dv * (c - a + u * k))
        return n, dx, dy

    # ---------------------------------------------------------------- terrain
    def sample(self, x, z):
        """(height, dh/dx, dh/dz) in metres; scalars or arrays."""
        if np.ndim(x) == 0 and np.ndim(z) == 0:
            noise, x, z = self._noise, float(x), float(z)
        else:
            noise = self._noise_np
            x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        h, dhx, dhz = self.base_height, 0.0, 0.0
        for freq, amp, ox, oy in self._octaves:
            n, dx, dy = noise(x * freq + ox, z * freq + oy)
            h = h + amp * n
            dhx = dhx + amp * freq * dx
            dhz = dhz + amp * freq * dy
        return h, dhx, dhz

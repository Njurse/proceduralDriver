"""Region splotching: which biome owns each patch of ground.

Region sites sit on a jittered square grid (config.REGION_SCALE_M apart), and
each site picks a biome by a seeded hash, weighted by the biomes' ``weight``.
A biome with ``region_size`` n > 1 instead claims whole n x n blocks of sites
(odds: its weight share), so its regions come out n times as wide.
A point belongs to its nearest site; near the bisector between two sites the
biomes blend.  Each site within REGION_BAND_M of the nearest one gets

    w_s = smoothstep(1 - (d_s - d_min) / band)

(1 for the nearest, falling to 0 a band further out, with zero slope at
both ends), and the biome weights are the normalised sums of their sites'.
The blend mixes the biomes' height fields, and the gradient carries the
weights' own gradients, so heights and normals are continuous everywhere:

    h  = sum_b W_b h_b
    dh = sum_b (W_b dh_b + h_b dW_b)

Physics and chunk meshes both read ``sample``, so they cannot disagree.
"""

import math

import numpy as np

import config
from core.car_forces import SURFACE_OFFROAD
from world.biomes import load_biomes
from world.chunks import terrain_colors
from world.ground import SampledGround
from world.terrain import Terrain

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_REACH = 2      # sites searched: (2 * _REACH + 1)^2 cells round the point's


def _mix(h):
    """splitmix64 finaliser on uint64 arrays (wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def _hash01(seed, salt, i, j):
    """Deterministic floats in [0, 1) for integer arrays i, j."""
    with np.errstate(over="ignore"):
        h = _mix(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt))
        h = _mix(h + np.asarray(i).astype(np.int64).astype(np.uint64))
        h = _mix(h + np.asarray(j).astype(np.int64).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def biome_terrain(biome, seed):
    """The height field a biome's terrain rules describe."""
    t = biome.terrain
    return Terrain(seed=seed, scale=t.noise_scale_m, octaves=t.octaves,
                   amplitude=t.amplitude_m, persistence=t.persistence,
                   lacunarity=t.lacunarity, base_height=t.base_height_m,
                   noise_type=t.noise_type, terrace_steps=t.terrace_steps,
                   terrace_ramp=t.terrace_ramp)


class Regions(SampledGround):
    """The world's biomes in seeded regions, blended at their borders.  No roads."""

    def __init__(self, biomes=None, seed=None, scale_m=None, band_m=None, jitter=None):
        self.biomes = tuple(load_biomes() if biomes is None else biomes)
        if not self.biomes:
            raise ValueError("Regions needs at least one biome")
        weights = np.array([b.weight for b in self.biomes], float)
        if weights.sum() <= 0.0:
            raise ValueError("every biome has weight 0")
        big = np.array([b.region_size > 1 for b in self.biomes])
        small = np.where(big, 0.0, weights)
        if small.sum() <= 0.0:
            small, big = weights, np.zeros_like(big)
        self._cum = np.cumsum(small) / small.sum()
        self._big = [(b, self.biomes[b].region_size, weights[b] / weights.sum())
                     for b in np.nonzero(big)[0].tolist()]
        self.seed = config.TERRAIN_SEED if seed is None else int(seed)
        self.scale_m = float(config.REGION_SCALE_M if scale_m is None else scale_m)
        self.band_m = float(config.REGION_BAND_M if band_m is None else band_m)
        self.jitter = float(config.REGION_JITTER if jitter is None else jitter)
        self.terrains = tuple(biome_terrain(b, self.seed + 1 + k)
                              for k, b in enumerate(self.biomes))
        first = self.biomes[0].terrain
        self.detail_texture = (config.ASSETS / first.detail_texture).as_posix()
        self.detail_tile_m = first.detail_tile_m
        self._cells = {}
        self._last_weights = None
        r = np.arange(-_REACH, _REACH + 1)
        self._offsets = np.stack(np.meshgrid(r, r), axis=-1).reshape(-1, 2)

    # ------------------------------------------------------------- sites
    def _sites(self, ci, cj):
        """(positions (k, 2), biome index (k,)) of the sites round cell (ci, cj)."""
        key = (ci, cj)
        hit = self._cells.get(key)
        if hit is None:
            if len(self._cells) > 4096:
                self._cells.clear()
            i = ci + self._offsets[:, 0]
            j = cj + self._offsets[:, 1]
            s, jit = self.scale_m, self.jitter
            x = (i + 0.5 + (2.0 * _hash01(self.seed, 11, i, j) - 1.0) * jit) * s
            z = (j + 0.5 + (2.0 * _hash01(self.seed, 12, i, j) - 1.0) * jit) * s
            pick = np.searchsorted(self._cum, _hash01(self.seed, 13, i, j), side="right")
            for b, n, odds in self._big:
                claim = _hash01(self.seed, 14 + b, np.floor_divide(i, n), np.floor_divide(j, n))
                pick = np.where(claim < odds, b, pick)
            pos = np.stack([x, z], axis=1)
            owner = np.minimum(pick, len(self.biomes) - 1)
            hit = self._cells[key] = (pos, owner,
                                      list(zip(x.tolist(), z.tolist(), owner.tolist(),
                                               strict=True)))
        return hit

    def _weights_scalar(self, x, z):
        """weights() for one point in plain Python (the physics path):
        [(biome, W, dW/dx, dW/dz)] for the biomes with W > 0."""
        sites = self._sites(math.floor(x / self.scale_m), math.floor(z / self.scale_m))[2]
        near = []
        for sx, sz, b in sites:
            px, pz = x - sx, z - sz
            near.append((math.hypot(px, pz), px, pz, b))
        d0, px0, pz0, _ = min(near)
        inv0 = 1.0 / max(d0, 1e-9)
        g0x, g0z = px0 * inv0, pz0 * inv0
        acc = {}
        for d, px, pz, b in near:
            t = (d - d0) / self.band_m
            if t >= 1.0:
                continue
            f = 1.0 - t * t * (3.0 - 2.0 * t)
            df = -6.0 * t * (1.0 - t) / self.band_m
            inv = 1.0 / max(d, 1e-9)
            a = acc.setdefault(b, [0.0, 0.0, 0.0])
            a[0] += f
            a[1] += df * (px * inv - g0x)
            a[2] += df * (pz * inv - g0z)
        total = sum(a[0] for a in acc.values())
        tx = sum(a[1] for a in acc.values())
        tz = sum(a[2] for a in acc.values())
        out = []
        for b, (w, wx, wz) in sorted(acc.items()):
            wn = w / total
            out.append((b, wn, (wx - wn * tx) / total, (wz - wn * tz) / total))
        return out

    def weights(self, x, z):
        """(W, dW/dx, dW/dz), each (n, n_biomes), for 1-D point arrays.  The
        last answer is kept: a chunk asks for its heights, then its colours."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        last = self._last_weights
        if last is not None and np.array_equal(last[0], x) and np.array_equal(last[1], z):
            return last[2]
        n, nb = len(x), len(self.biomes)
        w, wx, wz = np.zeros((n, nb)), np.zeros((n, nb)), np.zeros((n, nb))
        ci = np.floor(x / self.scale_m).astype(np.int64)
        cj = np.floor(z / self.scale_m).astype(np.int64)
        if ci.min() == ci.max() and cj.min() == cj.max():
            groups = [(int(ci[0]), int(cj[0]), slice(None))]       # one cell: the usual case
        else:
            cells, inv = np.unique(np.stack([ci, cj], axis=1), axis=0, return_inverse=True)
            inv = inv.ravel()
            groups = [(int(c[0]), int(c[1]), np.nonzero(inv == g)[0])
                      for g, c in enumerate(cells)]
        for gi, gj, m in groups:
            pos, owner, _ = self._sites(gi, gj)
            px = x[m, None] - pos[None, :, 0]
            pz = z[m, None] - pos[None, :, 1]
            d = np.hypot(px, pz)
            inv_d = 1.0 / np.maximum(d, 1e-9)
            gx, gz = px * inv_d, pz * inv_d
            k = np.argmin(d, axis=1)
            r = np.arange(len(d))
            t = np.clip((d - d[r, k][:, None]) / self.band_m, 0.0, 1.0)
            f = 1.0 - t * t * (3.0 - 2.0 * t)
            df = -6.0 * t * (1.0 - t) / self.band_m
            fx = df * (gx - gx[r, k][:, None])
            fz = df * (gz - gz[r, k][:, None])
            onehot = (owner[:, None] == np.arange(nb)[None, :]).astype(float)   # site -> biome
            w[m], wx[m], wz[m] = f @ onehot, fx @ onehot, fz @ onehot
        total = w.sum(axis=1, keepdims=True)
        tx, tz = wx.sum(axis=1, keepdims=True), wz.sum(axis=1, keepdims=True)
        wn = w / total
        out = wn, (wx - wn * tx) / total, (wz - wn * tz) / total
        self._last_weights = (x.copy(), z.copy(), out)
        return out

    def site_mean(self, x, z, values):
        """Per point, the mean of per-site ``values(pos (k, 2), biome (k,))``
        -> (k, m), weighted like the terrain blend (no gradients): (n, m)."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        ci = np.floor(x / self.scale_m).astype(np.int64)
        cj = np.floor(z / self.scale_m).astype(np.int64)
        out = None
        for cell in np.unique(np.stack([ci, cj], axis=1), axis=0):
            m = np.nonzero((ci == cell[0]) & (cj == cell[1]))[0]
            pos, owner, _ = self._sites(int(cell[0]), int(cell[1]))
            d = np.hypot(x[m, None] - pos[None, :, 0], z[m, None] - pos[None, :, 1])
            t = np.clip((d - d.min(axis=1, keepdims=True)) / self.band_m, 0.0, 1.0)
            f = 1.0 - t * t * (3.0 - 2.0 * t)
            v = values(pos, owner)
            if out is None:
                out = np.zeros((len(x), v.shape[1]))
            out[m] = f @ v / f.sum(axis=1, keepdims=True)
        return out

    def biome_index(self, x, z):
        """Index of the biome weighted most at each point (array)."""
        return np.argmax(self.weights(x, z)[0], axis=1)

    def biome_at(self, x, z):
        """The biome that owns the point (x, z)."""
        return self.biomes[max(self._weights_scalar(float(x), float(z)),
                               key=lambda e: e[1])[0]]

    # ------------------------------------------------------------ ground
    def sample(self, x, z):
        scalar = np.ndim(x) == 0 and np.ndim(z) == 0
        if scalar:
            x, z = float(x), float(z)
            used = self._weights_scalar(x, z)
            if len(used) == 1:
                return self.terrains[used[0][0]].sample(x, z)
            h = hx = hz = 0.0
            for b, w, wx, wz in used:
                hb, bx, bz = self.terrains[b].sample(x, z)
                h += w * hb
                hx += w * bx + hb * wx
                hz += w * bz + hb * wz
            return float(h), float(hx), float(hz)
        x, z = np.broadcast_arrays(np.asarray(x, float), np.asarray(z, float))
        shape = x.shape
        xf, zf = x.ravel(), z.ravel()
        w, wx, wz = self.weights(xf, zf)
        h, hx, hz = np.zeros(len(xf)), np.zeros(len(xf)), np.zeros(len(xf))
        for b in range(len(self.biomes)):
            m = np.nonzero(w[:, b] > 0.0)[0]
            if not len(m):
                continue
            hb, bx, bz = self.terrains[b].sample(xf[m], zf[m])
            h[m] += w[m, b] * hb
            hx[m] += w[m, b] * bx + hb * wx[m, b]
            hz[m] += w[m, b] * bz + hb * wz[m, b]
        return h.reshape(shape), hx.reshape(shape), hz.reshape(shape)

    def surface(self, x, z):
        return SURFACE_OFFROAD

    def colors(self, x, z, h, n):
        """RGBA per vertex: each biome's ramp, blended by the region weights."""
        x, z, h = (np.asarray(a, float) for a in (x, z, h))
        shape = x.shape
        w = self.weights(x.ravel(), z.ravel())[0]
        hf, nf = h.ravel(), np.asarray(n).reshape(-1, 3)
        out = np.zeros((len(hf), 4))
        for b, biome in enumerate(self.biomes):
            m = np.nonzero(w[:, b] > 0.0)[0]
            if not len(m):
                continue
            t = biome.terrain
            out[m] += w[m, b, None] * terrain_colors(hf[m] - t.base_height_m, nf[m],
                                                     t.amplitude_m, t.colors,
                                                     t.steep_slope_deg)
        return out.reshape(shape + (4,))

    def site_of(self, biome_name, near=(0.0, 0.0), radius_cells=12):
        """Centre (x, z) of the nearest region site owned by ``biome_name``."""
        b = [bi.name for bi in self.biomes].index(biome_name)
        ci = math.floor(near[0] / self.scale_m)
        cj = math.floor(near[1] / self.scale_m)
        best = None
        for r in range(0, radius_cells + 1, _REACH * 2 + 1):
            for di in range(-r, r + 1, _REACH * 2 + 1):
                for dj in range(-r, r + 1, _REACH * 2 + 1):
                    pos, owner, _ = self._sites(ci + di, cj + dj)
                    for p in pos[owner == b]:
                        d = math.hypot(p[0] - near[0], p[1] - near[1])
                        if best is None or d < best[0]:
                            best = (d, (float(p[0]), float(p[1])))
            if best is not None:
                return best[1]
        raise ValueError(f"no {biome_name} region within {radius_cells} cells")

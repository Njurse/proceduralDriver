"""Region splotching: which biome owns each patch of ground.

Region sites sit on a jittered square grid (WorldSpec.region_scale_m apart), and
each site picks a biome by a seeded hash, weighted by the biomes' ``weight``.
A biome with ``region_size`` n > 1 instead claims whole n x n blocks of sites
(odds: its weight share), so its regions come out n times as wide.
A point belongs to its nearest site; near the bisector between two sites the
biomes blend.  Each site within region_band_m of the nearest one gets

    w_s = smoothstep(1 - (d_s - d_min) / band)

(1 for the nearest, falling to 0 a band further out, with zero slope at
both ends), and the biome weights are the normalised sums of their sites'.
The blend mixes the biomes' height fields, and the gradient carries the
weights' own gradients, so heights and normals are continuous everywhere:

    h  = sum_b W_b h_b
    dh = sum_b (W_b dh_b + h_b dW_b)

Borders are made organic two ways.  The lookup is domain-warped: a point
(x, z) finds its sites at (u, v) = (x, z) + region_warp_m * (n1, n2), two
low-frequency noise fields, so straight bisectors wander into lobes and
fingers (the weights' gradients carry the warp's Jacobian).  And a biome's
``encroach`` (0..1) shortens every distance to its sites by
encroach * ENCROACH_SHARE * region_scale_m, so its regions push past the
plain bisector into their neighbours.  Weights still sum to 1.

Physics and chunk meshes both read ``sample``, so they cannot disagree.
"""

import math

import numpy as np

from worldgen.ground import SURFACE_OFFROAD, SampledGround
from worldgen.hashing import hash01_arr
from worldgen.height import Terrain
from worldgen.palette import terrain_colors

_REACH = 2      # sites searched: (2 * _REACH + 1)^2 cells round the point's
ENCROACH_SHARE = 0.25   # encroach 1 shortens distances by this share of the site spacing


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

    def __init__(self, spec):
        self.spec = spec
        self.biomes = tuple(spec.biomes)
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
        self.seed = int(spec.seed)
        self.scale_m = float(spec.region_scale_m)
        self.band_m = float(spec.region_band_m)
        self.jitter = float(spec.region_jitter)
        self._bias = np.array([b.encroach for b in self.biomes]) * ENCROACH_SHARE * self.scale_m
        self.warp_m = float(spec.region_warp_m)
        ws = float(spec.region_warp_scale_m)
        self._warp_u = Terrain(seed=self.seed + 7001, scale=ws, octaves=3, amplitude=2.0,
                               persistence=0.5)
        self._warp_v = Terrain(seed=self.seed + 7002, scale=ws, octaves=3, amplitude=2.0,
                               persistence=0.5)
        self.terrains = tuple(biome_terrain(b, self.seed + 1 + k)
                              for k, b in enumerate(self.biomes))
        self._cells = {}
        self._last_weights = None
        r = np.arange(-_REACH, _REACH + 1)
        self._offsets = np.stack(np.meshgrid(r, r), axis=-1).reshape(-1, 2)

    # ------------------------------------------------------------- warp
    def warp(self, x, z):
        """(u, v, du/dx, du/dz, dv/dx, dv/dz): where (x, z) looks up its sites."""
        a = self.warp_m
        if a == 0.0:
            return x, z, 1.0, 0.0, 0.0, 1.0
        n1, n1x, n1z = self._warp_u.sample(x, z)
        n2, n2x, n2z = self._warp_v.sample(x, z)
        return x + a * n1, z + a * n2, 1.0 + a * n1x, a * n1z, a * n2x, 1.0 + a * n2z

    def unwarp(self, u, v, steps=8):
        """A point (x, z) whose warp lands on (u, v) (fixed-point iteration)."""
        x, z = u, v
        for _ in range(steps):
            wu, wv = self.warp(x, z)[:2]
            x, z = x + (u - wu), z + (v - wv)
        return x, z

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
            x = (i + 0.5 + (2.0 * hash01_arr(self.seed, 11, i, j) - 1.0) * jit) * s
            z = (j + 0.5 + (2.0 * hash01_arr(self.seed, 12, i, j) - 1.0) * jit) * s
            pick = np.searchsorted(self._cum, hash01_arr(self.seed, 13, i, j), side="right")
            for b, n, odds in self._big:
                claim = hash01_arr(self.seed, 14 + b, np.floor_divide(i, n), np.floor_divide(j, n))
                pick = np.where(claim < odds, b, pick)
            pos = np.stack([x, z], axis=1)
            owner = np.minimum(pick, len(self.biomes) - 1)
            bias = self._bias[owner]
            hit = self._cells[key] = (pos, owner,
                                      list(zip(x.tolist(), z.tolist(), owner.tolist(),
                                               bias.tolist(), strict=True)), bias)
        return hit

    def _weights_scalar(self, x, z):
        """weights() for one point in plain Python (the physics path):
        [(biome, W, dW/dx, dW/dz)] for the biomes with W > 0."""
        u, v, ux, uz, vx, vz = self.warp(x, z)
        return [(b, w, wu * ux + wv * vx, wu * uz + wv * vz)
                for b, w, wu, wv in self._weights_scalar_uv(float(u), float(v))]

    def _weights_scalar_uv(self, x, z):
        sites = self._sites(math.floor(x / self.scale_m), math.floor(z / self.scale_m))[2]
        near = []
        for sx, sz, b, bias in sites:
            px, pz = x - sx, z - sz
            d = math.hypot(px, pz)
            near.append((d - bias, d, px, pz, b))
        e0, d0, px0, pz0, _ = min(near)
        inv0 = 1.0 / max(d0, 1e-9)
        g0x, g0z = px0 * inv0, pz0 * inv0
        acc = {}
        for e, d, px, pz, b in near:
            t = (e - e0) / self.band_m
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
        u, v, ux, uz, vx, vz = self.warp(x, z)
        wn, wu, wv = self._weights_uv(u, v)
        out = wn, wu * np.reshape(ux, (-1, 1)) + wv * np.reshape(vx, (-1, 1)), \
            wu * np.reshape(uz, (-1, 1)) + wv * np.reshape(vz, (-1, 1))
        self._last_weights = (x.copy(), z.copy(), out)
        return out

    def _weights_uv(self, x, z):
        """weights() in warped coordinates: (W, dW/du, dW/dv)."""
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
            pos, owner, _, bias = self._sites(gi, gj)
            px = x[m, None] - pos[None, :, 0]
            pz = z[m, None] - pos[None, :, 1]
            d = np.hypot(px, pz)
            inv_d = 1.0 / np.maximum(d, 1e-9)
            gx, gz = px * inv_d, pz * inv_d
            e = d - bias[None, :]
            k = np.argmin(e, axis=1)
            r = np.arange(len(d))
            t = np.clip((e - e[r, k][:, None]) / self.band_m, 0.0, 1.0)
            f = 1.0 - t * t * (3.0 - 2.0 * t)
            df = -6.0 * t * (1.0 - t) / self.band_m
            fx = df * (gx - gx[r, k][:, None])
            fz = df * (gz - gz[r, k][:, None])
            onehot = (owner[:, None] == np.arange(nb)[None, :]).astype(float)   # site -> biome
            w[m], wx[m], wz[m] = f @ onehot, fx @ onehot, fz @ onehot
        total = w.sum(axis=1, keepdims=True)
        tx, tz = wx.sum(axis=1, keepdims=True), wz.sum(axis=1, keepdims=True)
        wn = w / total
        return wn, (wx - wn * tx) / total, (wz - wn * tz) / total

    def site_mean(self, x, z, values):
        """Per point, the mean of per-site ``values(pos (k, 2), biome (k,))``
        -> (k, m), weighted like the terrain blend (no gradients): (n, m)."""
        x, z = np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(z, float))
        x, z = self.warp(x, z)[:2]
        ci = np.floor(x / self.scale_m).astype(np.int64)
        cj = np.floor(z / self.scale_m).astype(np.int64)
        out = None
        for cell in np.unique(np.stack([ci, cj], axis=1), axis=0):
            m = np.nonzero((ci == cell[0]) & (cj == cell[1]))[0]
            pos, owner, _, bias = self._sites(int(cell[0]), int(cell[1]))
            d = np.hypot(x[m, None] - pos[None, :, 0], z[m, None] - pos[None, :, 1])
            d = d - bias[None, :]
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
        """Centre (x, z) of the nearest region site owned by ``biome_name``
        (the point that warps onto the site)."""
        b = [bi.name for bi in self.biomes].index(biome_name)
        ci = math.floor(near[0] / self.scale_m)
        cj = math.floor(near[1] / self.scale_m)
        best = None
        for r in range(0, radius_cells + 1, _REACH * 2 + 1):
            for di in range(-r, r + 1, _REACH * 2 + 1):
                for dj in range(-r, r + 1, _REACH * 2 + 1):
                    pos, owner, _, _ = self._sites(ci + di, cj + dj)
                    for p in pos[owner == b]:
                        d = math.hypot(p[0] - near[0], p[1] - near[1])
                        if best is None or d < best[0]:
                            best = (d, (float(p[0]), float(p[1])))
            if best is not None:
                x, z = self.unwarp(*best[1])
                return float(x), float(z)
        raise ValueError(f"no {biome_name} region within {radius_cells} cells")

"""One road: its curve, height profile and how it flattens the ground.

``build`` shapes an edge as a quadratic Bezier bowed by a hash (less on a
grid), cut into ~SEGMENT_M segments; ``_profile`` smooths the terrain under
it and clamps it to the biome's grade.  ``blend`` (point arrays) and
``blend_point`` (one point, plain Python for the physics tick) cut and fill
the terrain: the road height within half the width, a smoothstep back to the
terrain over the shoulder, overlapping roads averaged by weight.
"""

import math

import numpy as np

from world.roads.layout import hash01

SEGMENT_M = 4.0
SMOOTH_WINDOW_M = 48.0
SMOOTH_PASSES = 3


def _smoothstep_weight(d, half_width, shoulder):
    """Road blend weight from lateral distance d."""
    u = np.clip((half_width + shoulder - d) / shoulder, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), -6.0 * u * (1.0 - u) / shoulder


def _profile(h, s, max_grade):
    """Terrain heights along a road, smoothed then grade-clamped.
    End heights (the nodes) are pinned."""
    n = len(h)
    half = max(1, round(0.5 * SMOOTH_WINDOW_M / max(s[-1] / (n - 1), 1e-6)))
    kernel = np.ones(2 * half + 1) / (2 * half + 1)
    out = h.copy()
    for _ in range(SMOOTH_PASSES):
        out = np.convolve(np.pad(out, half, mode="edge"), kernel, mode="valid")
        out[0], out[-1] = h[0], h[-1]
    ds = np.diff(s) * max_grade
    for i in range(1, n):
        out[i] = min(max(out[i], out[i - 1] - ds[i - 1]), out[i - 1] + ds[i - 1])
    for i in range(n - 2, -1, -1):
        out[i] = min(max(out[i], out[i + 1] - ds[i]), out[i + 1] + ds[i])
    out[0] = h[0]
    return out



class Road:
    """One road between two network nodes: polyline, height profile, rules."""

    def __init__(self, key, pts, h, biome):
        rules = biome.roads
        self.key    = key
        self.biome  = biome
        self.pts    = pts          # (n+1, 2)  x, z
        self.h      = h            # (n+1,)    height profile
        self.half_width = 0.5 * rules.width_m
        self.shoulder   = rules.shoulder_m
        self.flatten    = rules.flatten
        self.color      = rules.color
        self.markings   = rules.markings
        self.max_grade  = rules.max_grade_pct / 100.0
        self.ax, self.az = pts[:-1, 0], pts[:-1, 1]
        self.dx, self.dz = np.diff(pts[:, 0]), np.diff(pts[:, 1])
        self.inv_len2    = 1.0 / (self.dx * self.dx + self.dz * self.dz)
        self.dh = np.diff(h)
        # plain lists for the one-point physics query (blend_point)
        self.seg = list(zip(self.ax.tolist(), self.az.tolist(), self.dx.tolist(),
                            self.dz.tolist(), self.inv_len2.tolist(), h[:-1].tolist(),
                            self.dh.tolist(), strict=True))
        r = self.half_width + self.shoulder
        self.bbox = (pts[:, 0].min() - r, pts[:, 1].min() - r,
                     pts[:, 0].max() + r, pts[:, 1].max() + r)

    def overlaps(self, x0, z0, x1, z1):
        b = self.bbox
        return b[0] <= x1 and x0 <= b[2] and b[1] <= z1 and z0 <= b[3]

    def terms(self, x, z):
        """Per-point: distance, road height, height gradient x/z,
        unit distance-gradient x/z, nearest segment index."""
        px = x[:, None] - self.ax
        pz = z[:, None] - self.az
        t  = np.clip((px * self.dx + pz * self.dz) * self.inv_len2, 0.0, 1.0)
        cx = px - t * self.dx
        cz = pz - t * self.dz
        k  = np.argmin(cx * cx + cz * cz, axis=1)
        r  = np.arange(len(x))
        t, cx, cz = t[r, k], cx[r, k], cz[r, k]
        d  = np.hypot(cx, cz)
        hr = self.h[k] + t * self.dh[k]
        g  = np.where((t > 0.0) & (t < 1.0),
                      self.dh[k] * self.inv_len2[k], 0.0)
        inv = np.where(d > 1e-9, 1.0 / np.maximum(d, 1e-9), 0.0)
        return d, hr, g * self.dx[k], g * self.dz[k], cx * inv, cz * inv, k



def build(regions, seed, key, a, b, biome, grid):
    """The Road from node a to b, (x, z, h) each; None if they coincide."""
    p0, p2 = np.array(a[:2]), np.array(b[:2])
    chord = p2 - p0
    length = float(np.hypot(*chord))
    if length < 1e-3:
        return None
    perp = np.array([-chord[1], chord[0]]) / length
    bow = (2.0 * hash01(seed, 3, *key) - 1.0) * min(
        biome.roads.curviness_m * (1.0 - grid), length * 0.3)
    p1 = 0.5 * (p0 + p2) + perp * bow
    n = max(4, math.ceil(1.05 * length / SEGMENT_M))
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t * t * p2
    ht = np.asarray(regions.height(pts[:, 0], pts[:, 1]), float)
    ht[0], ht[-1] = a[2], b[2]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    return Road(key, pts, _profile(ht, s, biome.roads.max_grade_pct / 100.0), biome)


def blend(roads, x, z, h, hx, hz):
    """Flatten terrain samples into ``roads``.
    Returns (height, dh/dx, dh/dz, on_road)."""
    on_road = np.zeros(len(x), bool)
    s0 = s1 = s0x = s0z = s1x = s1z = 0.0
    weights = []
    for r in roads:
        d, hr, hrx, hrz, ddx, ddz, _ = r.terms(x, z)
        on_road |= d <= r.half_width
        if not r.flatten:
            continue
        w, dw = _smoothstep_weight(d, r.half_width, r.shoulder)
        wx, wz = dw * ddx, dw * ddz
        s0,  s0x,  s0z  = s0  + w,      s0x  + wx,           s0z  + wz
        s1,  s1x,  s1z  = s1  + w * hr, s1x  + wx * hr + w * hrx, \
                                         s1z  + wz * hr + w * hrz
        weights.append((w, wx, wz))
    if not weights:
        return h, hx, hz, on_road
    has = s0 > 0.0
    s0  = np.where(has, s0, 1.0)
    rh  = np.where(has, s1 / s0, h)
    rx  = np.where(has, (s1x * s0 - s1 * s0x) / (s0 * s0), 0.0)
    rz  = np.where(has, (s1z * s0 - s1 * s0z) / (s0 * s0), 0.0)
    big_w  = 1.0 - np.prod([1.0 - w for w, _, _ in weights], axis=0)
    big_wx = big_wz = 0.0
    for i, (_, wx, wz) in enumerate(weights):
        others = np.prod(
            [1.0 - w for j, (w, _, _) in enumerate(weights) if j != i]
            + [np.ones_like(x)], axis=0)
        big_wx += wx * others
        big_wz += wz * others
    lift = rh - h
    return (h + big_w * lift,
            hx + big_wx * lift + big_w * (rx - hx),
            hz + big_wz * lift + big_w * (rz - hz),
            on_road)



def blend_point(segs, x, z, h, hx, hz):
    """``blend`` for one point, in plain Python.  ``segs`` is [(road,
    segment indices)] holding every segment within reach of the point."""
    on_road = False
    s0 = s1 = s0x = s0z = s1x = s1z = 0.0
    big_w, keep, ws = 0.0, 1.0, []
    for r, ks in segs:
        best = None
        for k in ks:
            ax, az, dx, dz, il, _, _ = r.seg[k]
            px, pz = x - ax, z - az
            t = min(max((px * dx + pz * dz) * il, 0.0), 1.0)
            cx, cz = px - t * dx, pz - t * dz
            d2 = cx * cx + cz * cz
            if best is None or d2 < best[0]:
                best = (d2, k, t, cx, cz)
        d2, k, t, cx, cz = best
        d = math.sqrt(d2)
        reach = r.half_width + r.shoulder
        if d >= reach:
            continue                      # weight 0 and off the road
        on_road = on_road or d <= r.half_width
        if not r.flatten:
            continue
        _, _, dx, dz, il, h0, dh = r.seg[k]
        hr = h0 + t * dh
        g = dh * il if 0.0 < t < 1.0 else 0.0
        inv = 1.0 / d if d > 1e-9 else 0.0
        u = min((reach - d) / r.shoulder, 1.0)
        w, dw = u * u * (3.0 - 2.0 * u), -6.0 * u * (1.0 - u) / r.shoulder
        wx, wz = dw * cx * inv, dw * cz * inv
        s0, s0x, s0z = s0 + w, s0x + wx, s0z + wz
        s1, s1x, s1z = s1 + w * hr, s1x + wx * hr + w * g * dx, s1z + wz * hr + w * g * dz
        ws.append((w, wx, wz))
    if not ws or s0 <= 0.0:
        return h, hx, hz, on_road
    rh = s1 / s0
    rx, rz = (s1x * s0 - s1 * s0x) / (s0 * s0), (s1z * s0 - s1 * s0z) / (s0 * s0)
    big_wx = big_wz = 0.0
    for i, (w, wx, wz) in enumerate(ws):
        others = math.prod(1.0 - v for j, (v, _, _) in enumerate(ws) if j != i)
        big_wx, big_wz = big_wx + wx * others, big_wz + wz * others
        keep *= 1.0 - w
    big_w = 1.0 - keep
    lift = rh - h
    return (h + big_w * lift, hx + big_wx * lift + big_w * (rx - hx),
            hz + big_wz * lift + big_w * (rz - hz), on_road)

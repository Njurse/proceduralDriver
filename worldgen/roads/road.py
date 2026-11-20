"""One road: its curve, height profile and how it flattens the ground.

``build`` shapes an edge through its layout ``apex``: one quadratic Bezier,
or with the biome's ``corner_radius_m`` straight legs and a bend of that
radius at the apex (0: a sharp corner), cut into ~SEGMENT_M segments; ``_profile`` smooths the terrain under
it and clamps it to the biome's grade.  ``blend`` (point arrays) and
``blend_point`` (one point, plain Python for the physics tick) cut and fill
the terrain: the road height within half the width, a smoothstep back to the
terrain over the shoulder, overlapping roads averaged by weight.
"""

import math

import numpy as np

from worldgen.roads.layout import apex

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

    def terms(self, x, z, reach=None):
        """Per-point: distance, road height, height gradient x/z,
        unit distance-gradient x/z, nearest segment index.  With ``reach``
        only segments within reach of the points' bounding box are searched
        (points farther than ``reach`` then get some farther segment); None
        when there are none."""
        ax, az, dx, dz, il, sel = self.ax, self.az, self.dx, self.dz, self.inv_len2, None
        if reach is not None:
            bx, bz = ax + dx, az + dz
            m = ((np.minimum(ax, bx) <= x.max() + reach) & (np.maximum(ax, bx) >= x.min() - reach)
                 & (np.minimum(az, bz) <= z.max() + reach)
                 & (np.maximum(az, bz) >= z.min() - reach))
            if not m.any():
                return None
            if not m.all():
                sel = np.nonzero(m)[0]
                ax, az, dx, dz, il = ax[sel], az[sel], dx[sel], dz[sel], il[sel]
        px = x[:, None] - ax
        pz = z[:, None] - az
        t  = np.clip((px * dx + pz * dz) * il, 0.0, 1.0)
        cx = px - t * dx
        cz = pz - t * dz
        k  = np.argmin(cx * cx + cz * cz, axis=1)
        r  = np.arange(len(x))
        t, cx, cz = t[r, k], cx[r, k], cz[r, k]
        d  = np.hypot(cx, cz)
        dxk, dzk, ilk = dx[k], dz[k], il[k]
        if sel is not None:
            k = sel[k]
        hr = self.h[k] + t * self.dh[k]
        g  = np.where((t > 0.0) & (t < 1.0), self.dh[k] * ilk, 0.0)
        inv = np.where(d > 1e-9, 1.0 / np.maximum(d, 1e-9), 0.0)
        return d, hr, g * dxk, g * dzk, cx * inv, cz * inv, k



def build(regions, seed, key, a, b, biome):
    """The Road from node a to b, (x, z, h, g, curv) each; None if they coincide."""
    p0, p2 = np.array(a[:2]), np.array(b[:2])
    length = float(np.hypot(*(p2 - p0)))
    if length < 1e-3:
        return None
    pts = _shape(p0, np.array(apex(seed, a, b)), p2, biome.roads.corner_radius_m)
    ht = np.asarray(regions.height(pts[:, 0], pts[:, 1]), float)
    ht[0], ht[-1] = a[2], b[2]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
    return Road(key, pts, _profile(ht, s, biome.roads.max_grade_pct / 100.0), biome)


def _line(p, q, n):
    t = np.linspace(0.0, 1.0, n + 1)[:, None]
    return p + t * (q - p)


def _shape(p0, p1, p2, radius):
    """Polyline from p0 to p2 bending at p1: one quadratic Bezier when
    ``radius`` is None, else straight legs and a Bezier fillet whose ends
    sit radius x tan(turn / 2) from p1 (clipped to the legs)."""
    l1, l2 = float(np.hypot(*(p1 - p0))), float(np.hypot(*(p2 - p1)))
    if radius is None or min(l1, l2) < 1e-6:
        q0, q2 = p0, p2
    else:
        u1, u2 = (p1 - p0) / l1, (p2 - p1) / l2
        turn = math.acos(min(max(float(u1 @ u2), -1.0), 1.0))
        cut = radius * math.tan(0.5 * turn)
        q0, q2 = p1 - u1 * min(cut, l1), p1 + u2 * min(cut, l2)
    bend = float(np.hypot(*(q0 - p1)) + np.hypot(*(q2 - p1)))
    t = np.linspace(0.0, 1.0, max(2, math.ceil(1.05 * bend / SEGMENT_M)) + 1)[:, None]
    parts = [_line(p0, q0, math.ceil(np.hypot(*(q0 - p0)) / SEGMENT_M))[:-1],
             (1 - t) ** 2 * q0 + 2 * t * (1 - t) * p1 + t * t * q2,
             _line(q2, p2, math.ceil(np.hypot(*(p2 - q2)) / SEGMENT_M))[1:]]
    pts = np.concatenate(parts)
    keep = np.concatenate([[True], np.hypot(*np.diff(pts, axis=0).T) > 1e-6])
    return pts[keep]


def blend(roads, x, z, h, hx, hz):
    """Flatten terrain samples into ``roads``.
    Returns (height, dh/dx, dh/dz, on_road)."""
    on_road = np.zeros(len(x), bool)
    s0 = s1 = s0x = s0z = s1x = s1z = 0.0
    weights = []
    for r in roads:
        near = r.terms(x, z, reach=r.half_width + r.shoulder)
        if near is None:        # no segment within reach: weight 0, off the road
            continue
        d, hr, hrx, hrz, ddx, ddz, _ = near
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

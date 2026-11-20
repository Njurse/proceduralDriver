"""Basic roads: a jittered grid, streamed with the terrain.

Nodes sit on a square lattice of pitch ``base`` (the smallest road spacing of
the world's biomes), each nudged by a hash of its lattice index.  The edge
between two neighbouring nodes is a road when its grid line index is a
multiple of spacing / base for the biome at the edge's midpoint, so a sparser
biome keeps only every 2nd (3rd, ...) line and the density changes at biome
borders.  Each road is a quadratic Bezier between its nodes, bowed sideways by
another hash, cut into ~SEGMENT_M segments, with a height profile that is the
terrain under it smoothed and then clamped to the biome's max grade.

Everything comes from integer hashes of (seed, i, j), never from the order of
queries, so any stretch of road can be rebuilt on its own: physics and every
chunk see the same roads.

``RoadNetwork.blend`` flattens the ground into the roads: within half the
road width the height is the road profile, then a smoothstep over the
shoulder takes it back to the terrain.  Where roads overlap (round a node)
their profiles are averaged by weight, so the surface never steps.
"""

import math

import numpy as np

SEGMENT_M = 4.0           # polyline segment length
MARKING_HALF_WIDTH = 0.12  # dashed centre line: 0.24 m wide, every other segment
MARKING_LIFT = 0.02       # above the road strip
SMOOTH_WINDOW_M = 48.0    # moving-average window of the height profile
SMOOTH_PASSES = 3
REACH_FRACTION = 0.45     # roads may stray this fraction of base from their line

_MASK = (1 << 64) - 1
_CACHE_LIMIT = 20000


def _mix(h):
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK
    return h ^ (h >> 31)


def hash01(*keys):
    """Deterministic float in [0, 1) from integer keys (splitmix64)."""
    h = 0x9E3779B97F4A7C15
    for k in keys:
        h = _mix((h + (k & _MASK)) & _MASK)
    return (h >> 11) / float(1 << 53)


def _smoothstep_weight(d, half_width, shoulder):
    """Road weight from distance d: 1 on the road, 0 past the shoulder."""
    u = np.clip((half_width + shoulder - d) / shoulder, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), -6.0 * u * (1.0 - u) / shoulder


class Road:
    """One road between two grid nodes: polyline, height profile, rules."""

    def __init__(self, key, pts, h, biome):
        rules = biome.roads
        self.key = key
        self.biome = biome
        self.pts = pts                              # (n + 1, 2) x, z
        self.h = h                                  # (n + 1,) profile heights
        self.half_width = 0.5 * rules.width_m
        self.shoulder = rules.shoulder_m
        self.flatten = rules.flatten
        self.color = rules.color
        self.markings = rules.markings
        self.max_grade = rules.max_grade_pct / 100.0
        self.ax, self.az = pts[:-1, 0], pts[:-1, 1]
        self.dx, self.dz = np.diff(pts[:, 0]), np.diff(pts[:, 1])
        self.inv_len2 = 1.0 / (self.dx * self.dx + self.dz * self.dz)
        self.dh = np.diff(h)
        r = self.half_width + self.shoulder
        self.bbox = (pts[:, 0].min() - r, pts[:, 1].min() - r,
                     pts[:, 0].max() + r, pts[:, 1].max() + r)

    def overlaps(self, x0, z0, x1, z1):
        b = self.bbox
        return b[0] <= x1 and x0 <= b[2] and b[1] <= z1 and z0 <= b[3]

    def terms(self, x, z):
        """For point arrays: (distance, road height, its gradient x/z,
        unit gradient of the distance x/z, nearest segment index)."""
        px = x[:, None] - self.ax
        pz = z[:, None] - self.az
        t = np.clip((px * self.dx + pz * self.dz) * self.inv_len2, 0.0, 1.0)
        cx = px - t * self.dx
        cz = pz - t * self.dz
        k = np.argmin(cx * cx + cz * cz, axis=1)
        r = np.arange(len(x))
        t, cx, cz = t[r, k], cx[r, k], cz[r, k]
        d = np.hypot(cx, cz)
        hr = self.h[k] + t * self.dh[k]
        # along the segment the height rises at dh / length; past its ends it
        # is the end height
        g = np.where((t > 0.0) & (t < 1.0), self.dh[k] * self.inv_len2[k], 0.0)
        inv = np.where(d > 1e-9, 1.0 / np.maximum(d, 1e-9), 0.0)
        return d, hr, g * self.dx[k], g * self.dz[k], cx * inv, cz * inv, k


def _profile(h, s, max_grade):
    """Terrain heights along a road, smoothed and clamped to max_grade.
    The end heights (the nodes) stay put so roads meet at their nodes."""
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
    out[0] = h[0]   # only moves where the nodes are too far apart in height
    return out


class RoadNetwork:
    """The jittered road grid over a world.regions.Regions."""

    def __init__(self, regions, seed=None):
        self.regions = regions
        self.seed = regions.seed if seed is None else int(seed)
        rules = [b.roads for b in regions.biomes]
        self.base = min(r.grid_spacing_m for r in rules)
        self.every = {}
        for b in regions.biomes:
            k = b.roads.grid_spacing_m / self.base
            if abs(k - round(k)) > 1e-6:
                raise ValueError(f"{b.name}: road grid_spacing_m {b.roads.grid_spacing_m} "
                                 f"is not a multiple of {self.base}")
            reach = (b.roads.jitter_m + 0.5 * b.roads.curviness_m
                     + 0.5 * b.roads.width_m + b.roads.shoulder_m)
            if reach > REACH_FRACTION * self.base:
                raise ValueError(f"{b.name}: roads reach {reach:.0f} m from their grid line "
                                 f"(jitter + curviness / 2 + width / 2 + shoulder); the "
                                 f"limit is {REACH_FRACTION * self.base:.0f} m")
            self.every[b.name] = round(k)
        self._nodes = {}
        self._roads = {}
        self._cells = {}

    # ----------------------------------------------------------- the grid
    def node(self, i, j):
        """(x, z, height) of grid node (i, j)."""
        key = (i, j)
        hit = self._nodes.get(key)
        if hit is None:
            if len(self._nodes) > _CACHE_LIMIT:
                self._nodes.clear()
            b = self.base
            jit = self.regions.biome_at(i * b, j * b).roads.jitter_m
            x = i * b + (2.0 * hash01(self.seed, 1, i, j) - 1.0) * jit
            z = j * b + (2.0 * hash01(self.seed, 2, i, j) - 1.0) * jit
            hit = self._nodes[key] = (x, z, float(self.regions.height(x, z)))
        return hit

    def road(self, kind, i, j):
        """The road from node (i, j) to (i + 1, j) (kind "h") or (i, j + 1)
        (kind "v"), or None where the grid has no road."""
        key = (kind, i, j)
        if key in self._roads:
            return self._roads[key]
        if len(self._roads) > _CACHE_LIMIT:
            self._roads.clear()
        b = self.base
        if kind == "h":
            mid, line, end = ((i + 0.5) * b, j * b), j, (i + 1, j)
        else:
            mid, line, end = (i * b, (j + 0.5) * b), i, (i, j + 1)
        biome = self.regions.biome_at(*mid)
        road = None
        if line % self.every[biome.name] == 0:
            road = self._build(key, self.node(i, j), self.node(*end), biome)
        self._roads[key] = road
        return road

    def _build(self, key, a, b, biome):
        p0, p2 = np.array(a[:2]), np.array(b[:2])
        chord = p2 - p0
        length = float(np.hypot(*chord))
        perp = np.array([-chord[1], chord[0]]) / length
        kind = 3 if key[0] == "h" else 4
        bow = (2.0 * hash01(self.seed, kind, key[1], key[2]) - 1.0) * biome.roads.curviness_m
        p1 = 0.5 * (p0 + p2) + perp * bow
        n = max(4, math.ceil(1.05 * length / SEGMENT_M))
        t = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t * t * p2
        ht = np.asarray(self.regions.height(pts[:, 0], pts[:, 1]), float)
        ht[0], ht[-1] = a[2], b[2]
        s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        return Road(key, pts, _profile(ht, s, biome.roads.max_grade_pct / 100.0), biome)

    def _cell(self, ci, cj):
        """Roads whose reach touches lattice cell (ci, cj)."""
        key = (ci, cj)
        hit = self._cells.get(key)
        if hit is None:
            if len(self._cells) > _CACHE_LIMIT:
                self._cells.clear()
            b = self.base
            box = (ci * b, cj * b, (ci + 1) * b, (cj + 1) * b)
            cand = [self.road("h", i, j) for i in (ci - 1, ci, ci + 1) for j in (cj, cj + 1)]
            cand += [self.road("v", i, j) for i in (ci, ci + 1) for j in (cj - 1, cj, cj + 1)]
            hit = self._cells[key] = tuple(r for r in cand if r is not None and r.overlaps(*box))
        return hit

    def roads_near(self, x0, z0, x1, z1):
        """Roads whose reach overlaps the box, each once, in a fixed order."""
        b = self.base
        seen = {}
        for ci in range(math.floor(x0 / b), math.floor(x1 / b) + 1):
            for cj in range(math.floor(z0 / b), math.floor(z1 / b) + 1):
                for r in self._cell(ci, cj):
                    if r.key not in seen and r.overlaps(x0, z0, x1, z1):
                        seen[r.key] = r
        return [seen[k] for k in sorted(seen)]

    # ------------------------------------------------------------- ground
    def blend(self, roads, x, z, h, hx, hz):
        """Flatten terrain samples (arrays) into ``roads``.
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
            s0, s0x, s0z = s0 + w, s0x + wx, s0z + wz
            s1, s1x, s1z = s1 + w * hr, s1x + wx * hr + w * hrx, s1z + wz * hr + w * hrz
            weights.append((w, wx, wz))
        if not weights:
            return h, hx, hz, on_road
        has = s0 > 0.0
        s0 = np.where(has, s0, 1.0)
        rh = np.where(has, s1 / s0, h)
        rx = np.where(has, (s1x * s0 - s1 * s0x) / (s0 * s0), 0.0)
        rz = np.where(has, (s1z * s0 - s1 * s0z) / (s0 * s0), 0.0)
        # W = 1 - prod(1 - w_i): 1 on any road, 0 clear of them all
        big_w = 1.0 - np.prod([1.0 - w for w, _, _ in weights], axis=0)
        big_wx = big_wz = 0.0
        for i, (_, wx, wz) in enumerate(weights):
            others = np.prod([1.0 - w for j, (w, _, _) in enumerate(weights) if j != i]
                             + [np.ones_like(x)], axis=0)
            big_wx, big_wz = big_wx + wx * others, big_wz + wz * others
        lift = rh - h
        return (h + big_w * lift, hx + big_wx * lift + big_w * (rx - hx),
                hz + big_wz * lift + big_w * (rz - hz), on_road)

    def nearest(self, x, z):
        """(distance, closest point (x, z), unit tangent (x, z), road) for the
        road nearest (x, z), searching about a grid cell round it."""
        b = self.base
        best = None
        xa, za = np.array([float(x)]), np.array([float(z)])
        for r in self.roads_near(x - b, z - b, x + b, z + b):
            d, _, _, _, ddx, ddz, k = r.terms(xa, za)
            if best is None or d[0] < best[0]:
                k = int(k[0])
                tan = np.array([r.dx[k], r.dz[k]]) * math.sqrt(r.inv_len2[k])
                best = (float(d[0]), np.array([x, z]) - d[0] * np.array([ddx[0], ddz[0]]),
                        tan, r)
        return best

    # ------------------------------------------------------------- meshes
    def strips(self, ground, x0, z0, x1, z1, lift):
        """Road strip geometry for segments whose midpoint lies in
        [x0, x1) x [z0, z1): (pos, normal, color, uv, tris) arrays or None.
        Vertices sit ``lift`` above ``ground`` (the flattened surface), and a
        segment shared by two chunks belongs to exactly one of them."""
        pos, col, uv, tris, lifts = [], [], [], [], []
        for r in self.roads_near(x0, z0, x1, z1):
            mid = 0.5 * (r.pts[:-1] + r.pts[1:])
            sel = np.nonzero((mid[:, 0] >= x0) & (mid[:, 0] < x1)
                             & (mid[:, 1] >= z0) & (mid[:, 1] < z1))[0]
            if not len(sel):
                continue
            # mitred cross-section at every point: neighbours share vertices
            tan = np.gradient(r.pts, axis=0)
            tan /= np.linalg.norm(tan, axis=1, keepdims=True)
            side = np.stack([-tan[:, 1], tan[:, 0]], axis=1) * r.half_width
            s = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
            for k in sel:
                base = sum(len(p) for p in pos)
                quad = []
                for p in (k, k + 1):
                    for f in (-1.0, 0.0, 1.0):
                        quad.append((*(r.pts[p] + f * side[p]), f, s[p]))
                quad = np.array(quad)
                pos.append(quad[:, :2])
                lifts.extend([0.0] * 6)
                shade = 1.0 - 0.12 * np.abs(quad[:, 2:3])
                col.append(np.concatenate([np.array(r.color) * shade,
                                           np.ones((6, 1))], axis=1))
                uv.append(np.stack([(quad[:, 2] + 1.0) * r.half_width, quad[:, 3]], axis=1))
                tris.append(base + np.array([[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]]))
                if r.markings is not None and k % 2 == 0:
                    f = MARKING_HALF_WIDTH / r.half_width
                    dash = np.array([(*(r.pts[p] + g * f * side[p]), 0.0, s[p])
                                     for p in (k, k + 1) for g in (-1.0, 1.0)])
                    pos.append(dash[:, :2])
                    lifts.extend([MARKING_LIFT] * 4)
                    col.append(np.concatenate([np.tile(r.markings, (4, 1)), np.ones((4, 1))],
                                              axis=1))
                    uv.append(dash[:, [2, 3]])
                    tris.append(base + 6 + np.array([[0, 1, 2], [1, 3, 2]]))
        if not pos:
            return None
        xz = np.concatenate(pos)
        h, hx, hz = ground.sample(xz[:, 0], xz[:, 1])
        n = np.stack([-hx, np.ones_like(h), -hz], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        p3 = np.stack([xz[:, 0], h + lift + np.array(lifts), xz[:, 1]], axis=-1)
        tris = np.concatenate(tris)
        # counter-clockwise seen from above (x, z) like the terrain, whichever
        # way the road runs
        a, b, c = (xz[tris[:, i]] for i in range(3))
        cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        tris = np.where((cross < 0.0)[:, None], tris[:, [0, 2, 1]], tris)
        return p3, n, np.concatenate(col), np.concatenate(uv), tris

"""Organic road network: a grown spanning tree, streamed with the terrain.

The world is infinite, so the growth below runs per SUPER_M square
*super-cell*, each from its own hashed seed and confined to its square.
Neighbouring super-cells are stitched by linking mutually-nearest border
nodes across their shared edge.  A cell's graph depends only on (seed, cell)
and a stitch only on its two cells, so any query order sees the same roads.

Roads grow from seed nodes using a priority-queue expansion that minimises a
cost combining distance, terrain slope, and a jittered direction field.  The
result is a sparse spanning tree whose branches follow valleys and contours,
never committed to any axis, with occasional cross-links added for loops.

Growth rules
------------
* A fixed number of *seeds* are scattered (Poisson-disk, so they stay at
  least ``min_road_spacing_m`` apart).  Each seed is a node.
* A frontier heap expands greedily: the cheapest candidate edge is added,
  connecting a frontier node to a new node placed ``step_m`` ahead in the
  current growth direction, nudged by terrain gradient and a smooth noise
  field so the direction drifts over distance.
* A new node is rejected if it lands within ``merge_radius_m`` of an
  existing node (the existing node is used instead, forming a junction).
* After the tree is complete, cheap cross-links are inserted between nodes
  that are spatially close but tree-distant, adding loops.
* Every edge becomes a quadratic Bezier road (same as before) cut into
  ~SEGMENT_M segments with a smoothed, grade-clamped height profile.

``RoadNetwork.blend`` and ``.strips`` are unchanged in interface so the rest
of the engine needs no edits.
"""

import heapq
import math

import numpy as np

SEGMENT_M         = 4.0    # polyline segment length
MARKING_HALF_WIDTH = 0.12  # dashed centre line half-width (m)
MARKING_LIFT      = 0.02   # centre-line lift above road strip
SMOOTH_WINDOW_M   = 48.0   # moving-average window for height profile
SMOOTH_PASSES     = 3
SUPER_M           = 4096.0  # side of a growth super-cell

# Growth tuning
_STEP_M           = 60.0   # advance per growth step
_MIN_SPACING_M    = 120.0  # minimum distance between any two nodes
_MERGE_RADIUS_M   = 55.0   # snap to existing node instead of creating new one
_CROSS_LINK_MAX_D = 180.0  # max straight-line dist for a cross-link edge
_MAX_GRADE        = 0.12   # hard cap on allowable slope for growth cost
_TURN_NOISE_SCALE = 0.00018  # spatial frequency of direction-drift noise
_TURN_NOISE_AMP   = 0.55   # amplitude in radians
_BRANCH_PROB      = 0.18   # probability a grown node also spawns a side branch
_SEED_COUNT       = 48     # seeds per super-cell
_STITCH_M         = 600.0  # longest link between neighbouring super-cells
_TILE_M           = 256.0  # spatial index tile
_MARGIN_M         = 700.0  # roads stray at most this far out of their super-cell

_MASK        = (1 << 64) - 1
_CACHE_LIMIT = 32         # super-cells kept (and 1000x as many roads)


# ------------------------------------------------------------------ hashing

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


def _hash_angle(seed, x, z, scale):
    """Smooth, spatially-coherent angle offset (value-noise-like, 1 octave)."""
    ix, iz = int(math.floor(x * scale)), int(math.floor(z * scale))
    fx, fz = x * scale - ix, z * scale - iz
    # smoothstep mix
    ux = fx * fx * (3 - 2 * fx)
    uz = fz * fz * (3 - 2 * fz)
    a00 = hash01(seed, 7, ix,     iz    ) * 2 * math.pi
    a10 = hash01(seed, 7, ix + 1, iz    ) * 2 * math.pi
    a01 = hash01(seed, 7, ix,     iz + 1) * 2 * math.pi
    a11 = hash01(seed, 7, ix + 1, iz + 1) * 2 * math.pi
    return (a00 * (1 - ux) * (1 - uz) + a10 * ux * (1 - uz)
            + a01 * (1 - ux) * uz     + a11 * ux * uz)


def _smoothstep_weight(d, half_width, shoulder):
    """Road blend weight from lateral distance d."""
    u = np.clip((half_width + shoulder - d) / shoulder, 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u), -6.0 * u * (1.0 - u) / shoulder


# ------------------------------------------------------------------ profile

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


# ------------------------------------------------------------------ Road

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


# ------------------------------------------------------------------ growth

class _NodeStore:
    """Flat list of (x, z, height) nodes with O(1) nearest-within-radius."""

    def __init__(self, cell_size):
        self._cs    = cell_size
        self._cells = {}      # (ci, cj) -> [node_index, ...]
        self._nodes = []      # [(x, z, h)]

    def _cell(self, x, z):
        return int(math.floor(x / self._cs)), int(math.floor(z / self._cs))

    def add(self, x, z, h):
        idx = len(self._nodes)
        self._nodes.append((x, z, h))
        ci, cj = self._cell(x, z)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                self._cells.setdefault((ci + di, cj + dj), []).append(idx)
        return idx

    def nearest_within(self, x, z, radius, exclude=-1):
        """Index of the closest node within ``radius`` (not ``exclude``), or -1."""
        best_d2, best_i = radius * radius, -1
        ci, cj = self._cell(x, z)
        seen = set()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for idx in self._cells.get((ci + di, cj + dj), ()):
                    if idx in seen or idx == exclude:
                        continue
                    seen.add(idx)
                    nx, nz, _ = self._nodes[idx]
                    d2 = (x - nx) ** 2 + (z - nz) ** 2
                    if d2 < best_d2:
                        best_d2, best_i = d2, idx
        return best_i

    def all_within(self, x, z, radius):
        """Indices of all nodes within ``radius``."""
        r2 = radius * radius
        ci, cj = self._cell(x, z)
        seen, out = set(), []
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for idx in self._cells.get((ci + di, cj + dj), ()):
                    if idx in seen:
                        continue
                    seen.add(idx)
                    nx, nz, _ = self._nodes[idx]
                    if (x - nx) ** 2 + (z - nz) ** 2 <= r2:
                        out.append(idx)
        return out

    def __getitem__(self, i):
        return self._nodes[i]

    def __len__(self):
        return len(self._nodes)


def _grow_network(regions, seed, ci, cj):
    """(node_store, edge_list) of the road graph grown in super-cell (ci, cj):
    seeds and growth stay inside [ci, ci + 1) x [cj, cj + 1) * SUPER_M."""
    rng_seed = int(hash01(seed, 9, ci, cj) * (1 << 53))
    x0, z0 = ci * SUPER_M, cj * SUPER_M
    step = _STEP_M

    # 1. Poisson-disk seeds ------------------------------------------------
    ns   = _NodeStore(_MERGE_RADIUS_M)
    seeds_placed = 0
    attempts     = 0
    max_attempts = _SEED_COUNT * 60
    while seeds_placed < _SEED_COUNT and attempts < max_attempts:
        attempts += 1
        sx = x0 + hash01(rng_seed, 10, attempts) * SUPER_M
        sz = z0 + hash01(rng_seed, 11, attempts) * SUPER_M
        if ns.nearest_within(sx, sz, _MIN_SPACING_M) >= 0:
            continue
        h = float(regions.height(sx, sz))
        ns.add(sx, sz, h)
        seeds_placed += 1

    # 2. Priority-queue growth (Prim-like) ---------------------------------
    # heap: (cost, node_idx, dir_angle, branch_depth)
    heap   = []
    edges  = []           # [(i, j)]  node index pairs
    edge_set = set()

    def push_frontier(node_i, angle, depth):
        # jitter the angle with the direction-noise field
        nx, nz, _ = ns[node_i]
        noise_a = _hash_angle(rng_seed, nx, nz, _TURN_NOISE_SCALE)
        a = angle + _TURN_NOISE_AMP * math.sin(noise_a - angle)
        cost = hash01(rng_seed, 20, node_i, depth)  # slight randomness
        heapq.heappush(heap, (cost, node_i, a, depth))

    for idx in range(len(ns)):
        angle = hash01(rng_seed, 30, idx) * 2 * math.pi
        push_frontier(idx, angle, 0)

    max_nodes = _SEED_COUNT * 24
    step_count = 0

    while heap and len(ns) < max_nodes:
        cost, src_i, angle, depth = heapq.heappop(heap)
        sx, sz, sh = ns[src_i]

        # Terrain gradient -> steer slightly downhill (roads avoid steep climbs)
        eps = 2.0
        gx = (float(regions.height(sx + eps, sz)) - float(regions.height(sx - eps, sz))) / (2 * eps)
        gz = (float(regions.height(sx, sz + eps)) - float(regions.height(sx, sz - eps))) / (2 * eps)
        grad_angle = math.atan2(-gz, -gx)   # downhill direction
        blend = 0.25                          # 25 % terrain pull
        da    = grad_angle - angle
        da    = (da + math.pi) % (2 * math.pi) - math.pi   # wrap to (-pi, pi)
        angle = angle + blend * da

        # Apply direction noise for this step
        noise_a = _hash_angle(rng_seed, sx + 0.5, sz + 0.5, _TURN_NOISE_SCALE * 1.3)
        angle   = angle + _TURN_NOISE_AMP * 0.4 * math.sin(noise_a)

        nx = sx + step * math.cos(angle)
        nz = sz + step * math.sin(angle)
        if not (x0 <= nx < x0 + SUPER_M and z0 <= nz < z0 + SUPER_M):
            continue   # the neighbour super-cell grows its own roads
        nh = float(regions.height(nx, nz))

        # Grade check
        grade = abs(nh - sh) / step
        if grade > _MAX_GRADE * 2.5:
            # Try to steer around by rotating toward contour
            perp = angle + math.pi / 2
            nx2  = sx + step * math.cos(perp)
            nz2  = sz + step * math.sin(perp)
            nh2  = float(regions.height(nx2, nz2))
            inside = x0 <= nx2 < x0 + SUPER_M and z0 <= nz2 < z0 + SUPER_M
            if inside and abs(nh2 - sh) / step < grade:
                nx, nz, nh, angle = nx2, nz2, nh2, perp
            else:
                continue   # give up this branch; too steep either way

        # Merge or create
        existing = ns.nearest_within(nx, nz, _MERGE_RADIUS_M)
        if existing >= 0 and existing != src_i:
            dst_i = existing
            if (min(src_i, dst_i), max(src_i, dst_i)) not in edge_set:
                edge_set.add((min(src_i, dst_i), max(src_i, dst_i)))
                edges.append((src_i, dst_i))
            # Don't push frontier from a fully explored node
            continue
        elif existing == src_i:
            continue
        else:
            # Reject if too close to any other node (not just merge radius)
            if ns.nearest_within(nx, nz, _MIN_SPACING_M * 0.6, exclude=src_i) >= 0:
                continue
            dst_i = ns.add(nx, nz, nh)

        edge_set.add((src_i, dst_i))
        edges.append((src_i, dst_i))
        step_count += 1

        # Continue growing from new node
        push_frontier(dst_i, angle, depth + 1)

        # Probabilistic side branch
        if hash01(rng_seed, 40, dst_i) < _BRANCH_PROB:
            branch_angle = angle + (hash01(rng_seed, 41, dst_i) - 0.5) * math.pi * 0.9
            push_frontier(dst_i, branch_angle, depth + 1)

    # 3. Cross-links for loops ---------------------------------------------
    # Build a simple tree-distance approximation via BFS from every node
    # (only needed for small graphs; for large ones we just use Euclidean)
    for i in range(len(ns)):
        xi, zi, _ = ns[i]
        for j in ns.all_within(xi, zi, _CROSS_LINK_MAX_D):
            if j <= i:
                continue
            key = (i, j)
            if key in edge_set:
                continue
            xj, zj, _ = ns[j]
            straight = math.hypot(xi - xj, zi - zj)
            if straight < _MIN_SPACING_M * 0.8:
                continue
            # Quick grade check
            grade = abs(ns[i][2] - ns[j][2]) / max(straight, 1)
            if grade > _MAX_GRADE * 1.5:
                continue
            edge_set.add(key)
            edges.append((i, j))

    return ns, edges


# ------------------------------------------------------------------ RoadNetwork

class RoadNetwork:
    """Organic grown road network over a world.regions.Regions."""

    def __init__(self, regions, seed=None):
        self.regions = regions
        self.seed    = regions.seed if seed is None else int(seed)
        self._graphs = {}   # (ci, cj) -> (node_store, edges)
        self._cells  = {}   # (ci, cj) -> {tile: [(key, a, b, bbox)]}
        self._roads  = {}   # key -> Road | None, built on first use
        # how far a road strays from the box round its two nodes
        self._reach  = max(0.5 * b.roads.curviness_m + 0.5 * b.roads.width_m
                           + b.roads.shoulder_m for b in regions.biomes)

    # ---------------------------------------------------------------- graph

    def _graph(self, ci, cj):
        key = (ci, cj)
        if key not in self._graphs:
            if len(self._graphs) > 2 * _CACHE_LIMIT:
                self._graphs.clear()
            self._graphs[key] = _grow_network(self.regions, self.seed, ci, cj)
        return self._graphs[key]

    def _node(self, ref):
        return self._graph(*ref[0])[0][ref[1]]

    def _stitch(self, a, b):
        """Edges ((cell, i), (cell, j)) joining mutually-nearest nodes within
        _STITCH_M of the border between super-cell a and its +x or +z
        neighbour b."""
        axis = 0 if b[0] != a[0] else 1
        line = b[axis] * SUPER_M
        sides = []
        for cell in (a, b):
            ns = self._graph(*cell)[0]
            sides.append([(cell, i) for i in range(len(ns))
                          if abs(ns[i][axis] - line) < _STITCH_M])
        node = self._node

        def closest(ref, others):
            p = node(ref)
            return min(others, key=lambda o: math.hypot(p[0] - node(o)[0], p[1] - node(o)[1]),
                       default=None)

        out = []
        for ra in sides[0]:
            rb = closest(ra, sides[1])
            if rb is None or closest(rb, sides[0]) != ra:
                continue
            p, q = node(ra), node(rb)
            straight = math.hypot(p[0] - q[0], p[1] - q[1])
            if straight <= _STITCH_M and abs(p[2] - q[2]) / straight <= _MAX_GRADE * 1.5:
                out.append((ra, rb))
        return out

    def _cell(self, ci, cj):
        """{tile: [(key, a, b, bbox)]} for the edges of super-cell (ci, cj) and
        its stitches to the +x and +z neighbours (roads are built lazily)."""
        key = (ci, cj)
        if key in self._cells:
            return self._cells[key]
        if len(self._cells) > _CACHE_LIMIT:
            self._cells.clear()
        ns, edges = self._graph(ci, cj)
        pairs = [((key, i), (key, j)) for i, j in edges]
        pairs += self._stitch(key, (ci + 1, cj)) + self._stitch(key, (ci, cj + 1))
        tiles = {}
        for ra, rb in pairs:
            ra, rb = min(ra, rb), max(ra, rb)
            a, b = self._node(ra), self._node(rb)
            r = self._reach
            box = (min(a[0], b[0]) - r, min(a[1], b[1]) - r, max(a[0], b[0]) + r,
                   max(a[1], b[1]) + r)
            ref = ((*ra[0], ra[1], *rb[0], rb[1]), ra, rb, box)
            for ti in range(math.floor(box[0] / _TILE_M), math.floor(box[2] / _TILE_M) + 1):
                for tj in range(math.floor(box[1] / _TILE_M), math.floor(box[3] / _TILE_M) + 1):
                    tiles.setdefault((ti, tj), []).append(ref)
        self._cells[key] = tiles
        return tiles

    def _road(self, key, ra, rb):
        if key not in self._roads:
            if len(self._roads) > 1000 * _CACHE_LIMIT:
                self._roads.clear()
            a, b = self._node(ra), self._node(rb)
            biome = self.regions.biome_at(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            self._roads[key] = self._build(key, a, b, biome)
        return self._roads[key]

    def _build(self, key, a, b, biome):
        p0, p2 = np.array(a[:2]), np.array(b[:2])
        chord  = p2 - p0
        length = float(np.hypot(*chord))
        if length < 1e-3:
            return None
        perp = np.array([-chord[1], chord[0]]) / length
        bow  = (2.0 * hash01(self.seed, 3, *key) - 1.0) * min(
            biome.roads.curviness_m, length * 0.3)
        p1   = 0.5 * (p0 + p2) + perp * bow
        n    = max(4, math.ceil(1.05 * length / SEGMENT_M))
        t    = np.linspace(0.0, 1.0, n + 1)[:, None]
        pts  = (1 - t) ** 2 * p0 + 2 * t * (1 - t) * p1 + t * t * p2
        ht   = np.asarray(self.regions.height(pts[:, 0], pts[:, 1]), float)
        ht[0], ht[-1] = a[2], b[2]
        s    = np.concatenate([[0.0],
                               np.cumsum(np.hypot(*np.diff(pts, axis=0).T))])
        return Road(key, pts,
                    _profile(ht, s, biome.roads.max_grade_pct / 100.0),
                    biome)

    # ---------------------------------------------------------------- spatial

    def roads_near(self, x0, z0, x1, z1):
        """Roads whose reach overlaps the axis-aligned box, each once, in a
        fixed order."""
        seen = {}
        for ci in range(math.floor((x0 - _MARGIN_M) / SUPER_M),
                        math.floor((x1 + _MARGIN_M) / SUPER_M) + 1):
            for cj in range(math.floor((z0 - _MARGIN_M) / SUPER_M),
                            math.floor((z1 + _MARGIN_M) / SUPER_M) + 1):
                tiles = self._cell(ci, cj)
                for ti in range(math.floor(x0 / _TILE_M), math.floor(x1 / _TILE_M) + 1):
                    for tj in range(math.floor(z0 / _TILE_M), math.floor(z1 / _TILE_M) + 1):
                        for key, ra, rb, b in tiles.get((ti, tj), ()):
                            if (key in seen or b[0] > x1 or x0 > b[2] or b[1] > z1
                                    or z0 > b[3]):
                                continue
                            r = self._road(key, ra, rb)
                            if r is not None and r.overlaps(x0, z0, x1, z1):
                                seen[key] = r
        return [seen[k] for k in sorted(seen)]

    # ---------------------------------------------------------------- ground

    def blend(self, roads, x, z, h, hx, hz):
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

    def nearest(self, x, z):
        """(distance, closest point (x, z), unit tangent (x, z), road) for
        the road nearest (x, z), searching ever wider boxes round it."""
        xa, za = np.array([float(x)]), np.array([float(z)])
        r = _TILE_M
        while True:
            best = None
            for road in self.roads_near(x - r, z - r, x + r, z + r):
                d, _, _, _, ddx, ddz, k = road.terms(xa, za)
                if best is None or d[0] < best[0]:
                    k = int(k[0])
                    tan = np.array([road.dx[k], road.dz[k]]) * math.sqrt(road.inv_len2[k])
                    best = (float(d[0]),
                            np.array([x, z]) - d[0] * np.array([ddx[0], ddz[0]]),
                            tan, road)
            if best is not None and best[0] <= r:
                return best
            if r > 2 * SUPER_M:
                raise ValueError(f"no road within {r:.0f} m of ({x:.0f}, {z:.0f})")
            r *= 2.0

    # ---------------------------------------------------------------- meshes

    def strips(self, ground, x0, z0, x1, z1, lift):
        """Road strip geometry for the box [x0,x1) x [z0,z1).
        Returns (pos, normal, color, uv, tris) or None."""
        pos, col, uv, tris, lifts = [], [], [], [], []
        for r in self.roads_near(x0, z0, x1, z1):
            mid = 0.5 * (r.pts[:-1] + r.pts[1:])
            sel = np.nonzero(
                (mid[:, 0] >= x0) & (mid[:, 0] < x1)
                & (mid[:, 1] >= z0) & (mid[:, 1] < z1))[0]
            if not len(sel):
                continue
            tan  = np.gradient(r.pts, axis=0)
            tan /= np.linalg.norm(tan, axis=1, keepdims=True)
            side = np.stack([-tan[:, 1], tan[:, 0]], axis=1) * r.half_width
            s    = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
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
                col.append(np.concatenate(
                    [np.array(r.color) * shade, np.ones((6, 1))], axis=1))
                uv.append(np.stack(
                    [(quad[:, 2] + 1.0) * r.half_width, quad[:, 3]], axis=1))
                tris.append(base + np.array(
                    [[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]]))
                if r.markings is not None and k % 2 == 0:
                    f    = MARKING_HALF_WIDTH / r.half_width
                    dash = np.array([
                        (*(r.pts[p] + g * f * side[p]), 0.0, s[p])
                        for p in (k, k + 1) for g in (-1.0, 1.0)])
                    pos.append(dash[:, :2])
                    lifts.extend([MARKING_LIFT] * 4)
                    col.append(np.concatenate(
                        [np.tile(r.markings, (4, 1)), np.ones((4, 1))], axis=1))
                    uv.append(dash[:, [2, 3]])
                    tris.append(base + 6 + np.array([[0, 1, 2], [1, 3, 2]]))
        if not pos:
            return None
        xz  = np.concatenate(pos)
        h, hx, hz = ground.sample(xz[:, 0], xz[:, 1])
        n   = np.stack([-hx, np.ones_like(h), -hz], axis=-1)
        n  /= np.linalg.norm(n, axis=-1, keepdims=True)
        p3  = np.stack(
            [xz[:, 0], h + lift + np.array(lifts), xz[:, 1]], axis=-1)
        tris = np.concatenate(tris)
        a, b, c = (xz[tris[:, i]] for i in range(3))
        cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
               - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        tris = np.where((cross < 0.0)[:, None], tris[:, [0, 2, 1]], tris)
        return p3, n, np.concatenate(col), np.concatenate(uv), tris

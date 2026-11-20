"""Road network, streamed with the terrain (see layout for the graph).

    layout  nodes and edges per super-cell: organic spanning tree or turned grid
    road    one road: curve, height profile, and flattening the ground
    mesh    chunk strips and centre markings

``RoadNetwork`` caches a few super-cell graphs (LRU), builds roads lazily
behind a tile index, and keeps a 32 m tile hash of segments so the physics
query only measures the segments within reach.  ``prefetch`` builds the
super-cells the car is heading for on a worker thread, so streaming rarely
waits for one.
"""

import logging
import math
import threading
from collections import OrderedDict

import numpy as np

from worldgen.roads import layout, mesh
from worldgen.roads.layout import SUPER_M
from worldgen.roads.mesh import MARKING_HALF_WIDTH, MARKING_LIFT
from worldgen.roads.road import SEGMENT_M, Road, blend, blend_point, build

__all__ = ["MARKING_HALF_WIDTH", "MARKING_LIFT", "SEGMENT_M", "Road", "RoadNetwork"]

log = logging.getLogger(__name__)

_TILE_M = 256.0          # road index tile
_SEG_TILE_M = 32.0       # segment hash tile
_MARGIN_M = 1400.0       # a stitch road strays this far into the next super-cell
_GRAPHS, _CELLS = 16, 9  # super-cells kept: a 3 x 3 block and its stitch neighbours


_AHEAD_M = 600.0         # prefetch: super-cells a query this much farther out needs
_LOCK = threading.Lock()  # guards the LRU caches, not the builds


def _lru(cache, key, make, limit):
    with _LOCK:
        if key in cache:
            cache.move_to_end(key)
            return cache[key]
    value = make()   # unlocked: the worker and the game may both build one (same result)
    with _LOCK:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
    return value


class RoadNetwork:
    """The roads of a WorldSpec over its worldgen.regions.Regions."""

    def __init__(self, spec, regions):
        self.spec, self.regions = spec, regions
        self.seed = spec.seed
        self.min_junction_deg = spec.roads.min_junction_deg
        self._graphs, self._cells = OrderedDict(), OrderedDict()
        self._roads, self._segs = {}, {}
        self._reach = max(0.5 * b.roads.width_m + b.roads.shoulder_m for b in regions.biomes)
        self._worker = None

    def prefetch(self, x, z, r):
        """Start building, on a worker thread, the super-cells a query
        within r + _AHEAD_M of (x, z) will need (at most 3 x 3 of them, so
        the LRU keeps them all); no-op while one is running."""
        if self._worker is not None and self._worker.is_alive():
            return
        r += _AHEAD_M + _MARGIN_M
        want = [(ci, cj)
                for ci in range(math.floor((x - r) / SUPER_M), math.floor((x + r) / SUPER_M) + 1)
                for cj in range(math.floor((z - r) / SUPER_M), math.floor((z + r) / SUPER_M) + 1)]
        missing = [c for c in want if c not in self._cells]
        if missing:
            self._worker = threading.Thread(
                target=lambda: [self._cell(*c) for c in missing], name="road-prefetch",
                daemon=True)
            self._worker.start()

    def graph(self, ci, cj):
        """layout.Graph of super-cell (ci, cj)."""
        def make():
            g = layout.grow(self.regions, self.seed, ci, cj, self.min_junction_deg)
            log.info("super-cell (%d, %d): %d nodes (%d grid), %d edges in %.0f ms",
                     ci, cj, len(g.x), int(g.grid.sum()), len(g.edges), g.ms)
            return g
        return _lru(self._graphs, (ci, cj), make, _GRAPHS)

    def _cell(self, ci, cj):
        """{tile: [(key, a, b, box)]}: the edges of super-cell (ci, cj) and its
        stitches to the +x and +z neighbours, a and b as (ci, cj, node)."""
        def make():
            ga = self.graph(ci, cj)
            pairs = [((ci, cj, a), (ci, cj, b)) for a, b in ga.edges]
            for axis, nb in ((0, (ci + 1, cj)), (1, (ci, cj + 1))):
                pairs += [((ci, cj, a), (*nb, b)) for a, b in
                          layout.stitch(ga, self.graph(*nb), axis, nb[axis] * SUPER_M,
                                       self.seed, self.min_junction_deg)]
            tiles, r = {}, self._reach
            for ra, rb in pairs:
                ra, rb = min(ra, rb), max(ra, rb)
                a, b = self._node(ra), self._node(rb)
                px, pz = layout.apex(self.seed, a, b)   # the road bulges to its apex
                xs, zs = (a[0], b[0], px), (a[1], b[1], pz)
                box = (min(xs) - r, min(zs) - r, max(xs) + r, max(zs) + r)
                ref = (ra + rb, ra, rb, box)
                for ti in range(math.floor(box[0] / _TILE_M), math.floor(box[2] / _TILE_M) + 1):
                    for tj in range(math.floor(box[1] / _TILE_M),
                                    math.floor(box[3] / _TILE_M) + 1):
                        tiles.setdefault((ti, tj), []).append(ref)
            return tiles
        return _lru(self._cells, (ci, cj), make, _CELLS)

    def _node(self, ref):
        return self.graph(ref[0], ref[1]).node(ref[2])

    def _road(self, key, ra, rb):
        if key not in self._roads:
            if len(self._roads) > 20000:
                self._roads.clear()
            a, b = self._node(ra), self._node(rb)
            biome = self.regions.biome_at(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]))
            self._roads[key] = build(self.regions, self.seed, key, a, b, biome)
        return self._roads[key]

    # ------------------------------------------------------------ queries
    def roads_near(self, x0, z0, x1, z1):
        """Roads whose reach overlaps the box, each once, in a fixed order."""
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

    def segments_at(self, x, z):
        """[(road, segment indices)] within reach of the 32 m tile round (x, z)."""
        key = (math.floor(x / _SEG_TILE_M), math.floor(z / _SEG_TILE_M))
        hit = self._segs.get(key)
        if hit is None:
            if len(self._segs) > 50000:
                self._segs.clear()
            x0, z0 = key[0] * _SEG_TILE_M, key[1] * _SEG_TILE_M
            x1, z1 = x0 + _SEG_TILE_M, z0 + _SEG_TILE_M
            hit = []
            for r in self.roads_near(x0, z0, x1, z1):
                e = r.half_width + r.shoulder
                bx, bz = r.ax + r.dx, r.az + r.dz
                m = ((np.minimum(r.ax, bx) - e <= x1) & (np.maximum(r.ax, bx) + e >= x0)
                     & (np.minimum(r.az, bz) - e <= z1) & (np.maximum(r.az, bz) + e >= z0))
                if m.any():
                    hit.append((r, np.nonzero(m)[0].tolist()))
            self._segs[key] = hit
        return hit

    def query_point(self, x, z, h, hx, hz):
        """(height, dh/dx, dh/dz, on_road) of the terrain sample (h, hx, hz)
        at one point with the roads cut in: the physics path."""
        return blend_point(self.segments_at(x, z), x, z, h, hx, hz)

    blend = staticmethod(blend)

    def nearest(self, x, z):
        """(distance, closest point (x, z), unit tangent (x, z), road) for the
        road nearest (x, z), searching ever wider boxes round it."""
        xa, za = np.array([float(x)]), np.array([float(z)])
        r = _TILE_M
        while True:
            best = None
            for road in self.roads_near(x - r, z - r, x + r, z + r):
                d, _, _, _, ddx, ddz, k = road.terms(xa, za)
                if best is None or d[0] < best[0]:
                    k = int(k[0])
                    tan = np.array([road.dx[k], road.dz[k]]) * math.sqrt(road.inv_len2[k])
                    best = (float(d[0]), np.array([x, z]) - d[0] * np.array([ddx[0], ddz[0]]),
                            tan, road)
            if best is not None and best[0] <= r:
                return best
            if r > 2 * SUPER_M:
                raise ValueError(f"no road within {r:.0f} m of ({x:.0f}, {z:.0f})")
            r *= 2.0

    def strips(self, ground, x0, z0, x1, z1, lift):
        """Road strips for the box (see mesh.strips)."""
        return mesh.strips(self.roads_near(x0, z0, x1, z1), ground, x0, z0, x1, z1, lift)

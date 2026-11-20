"""Structures as plain data: buildings and odds and ends, per chunk.

A biome lists its generators in JSON (``"structures": [{"gen": ..., ...}]``);
``Structures(ground).chunk(ci, ck)`` runs each rule of every biome present
in the chunk and keeps the instances whose footprint lies inside that
biome, at least ``road_margin_m`` (default 2.5) off every road surface and
on ground no steeper than ``max_slope_deg`` (default 12).  ``mesh`` turns
instances into one ChunkMesh.  Generators: voronoi_towers, scatter, cottages.
"""

import math

import numpy as np

from worldgen.structures import cottages, scatter, towers  # noqa: F401  (they register)
from worldgen.structures.base import GENERATORS, Context, Instance, poly_distance
from worldgen.structures.mesh import structures_mesh

__all__ = ["GENERATORS", "Instance", "Structures", "structures_mesh"]

_REACH_M = 60.0          # road points gathered this far round the chunk
_FILTER_KEYS = ("gen", "max_slope_deg", "road_margin_m")


class Structures:
    def __init__(self, ground):
        self.ground = ground
        self.rules = []
        for b, biome in enumerate(ground.biomes):
            for r, rule in enumerate(biome.structures):
                gen = GENERATORS.get(rule["gen"])
                if gen is None:
                    raise ValueError(f"{biome.name}.structures[{r}]: unknown gen "
                                     f"{rule['gen']!r}; known: {', '.join(sorted(GENERATORS))}")
                params = {k: v for k, v in rule.items() if k not in _FILTER_KEYS}
                slope = math.tan(math.radians(float(rule.get("max_slope_deg", 12.0))))
                margin = float(rule.get("road_margin_m", 2.5))
                self.rules.append((b, gen, params, 1000 + 40 * b + r, slope, margin))

    def chunk(self, ci, ck):
        """The structures anchored in chunk (ci, ck), in a fixed order."""
        if not self.rules:
            return []
        g, size = self.ground, self.ground.spec.chunk_size_m
        x0, z0 = ci * size, ck * size
        x1, z1 = x0 + size, z0 + size
        f = np.array([0.0, 0.5, 1.0]) * size
        gx, gz = np.meshgrid(x0 + f, z0 + f)
        present = set(g.biome_index(gx.ravel(), gz.ravel()).tolist())
        rules = [r for r in self.rules if r[0] in present]
        if not rules:
            return []
        roads = g.roads.roads_near(x0 - _REACH_M, z0 - _REACH_M, x1 + _REACH_M, z1 + _REACH_M)
        pts = np.concatenate([r.pts for r in roads]) if roads else np.zeros((0, 2))
        hw = (np.concatenate([np.full(len(r.pts), r.half_width) for r in roads])
              if roads else np.zeros(0))
        ctx = Context(g, g.spec.seed, 0, ci, ck, x0, z0, x1, z1, (pts, hw))
        out = []
        for b, gen, params, salt, slope, margin in rules:
            ctx.salt = salt
            out += [s for s in gen(ctx, params) if self._keep(s, b, slope, margin, pts, hw)]
        return out

    def _keep(self, s, b, slope, margin, pts, hw):
        fp = s.footprint
        c = fp.mean(axis=0)
        probe = np.concatenate([fp, c[None], 0.5 * (fp + np.roll(fp, -1, axis=0))])
        x, z = probe[:, 0], probe[:, 1]
        if len(pts):
            lo, hi = fp.min(axis=0) - 40.0, fp.max(axis=0) + 40.0
            m = ((pts >= lo) & (pts <= hi)).all(axis=1)
            # road points are <= 4 m apart: 2 m more covers the surface between them
            if m.any() and (poly_distance(fp, pts[m, 0], pts[m, 1]) - hw[m]
                            < margin + 2.0).any():
                return False
        if (self.ground.biome_index(x, z) != b).any():
            return False
        h, hx, hz = self.ground.sample(x, z)
        if np.hypot(hx, hz).max() > slope:
            return False
        s.base_y = float(h.min()) - 0.2
        return True

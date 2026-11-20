"""voronoi_towers: a cyberpunk downtown grown from Voronoi cells.

Sites sit on a jittered ``cell_m`` lattice; each site's Voronoi cell, pulled
in by half of ``gap_m`` from every neighbour (the alleys), is a tower's
footprint.  Height mixes the site's own hash, a slow "downtown" noise field
(``core_scale_m``) and a coarser Worley F2 - F1 over ``cluster_m`` that is
tall in the middle of each cluster and low along its seams.  Walls take a
loud palette; some towers wear a neon trim band (emissive).
"""

import math

from worldgen.hashing import hash01
from worldgen.height import Terrain
from worldgen.structures.base import Instance, area, ccw, clip, lerp, pick, register

NEON = ((1.0, 0.1, 0.75), (0.05, 0.95, 1.0), (0.25, 0.35, 1.0), (0.6, 1.0, 0.1),
        (1.0, 0.5, 0.05), (0.62, 0.2, 1.0), (1.0, 0.15, 0.25), (1.0, 0.9, 0.1))
DARK = (0.10, 0.10, 0.14)

DEFAULTS = {"cell_m": 28.0, "gap_m": 6.0, "min_area_m2": 120.0, "height_m": (14.0, 150.0),
            "core_scale_m": 900.0, "cluster_m": 170.0, "neon_odds": 0.45,
            "darken": (0.0, 0.55)}

_CORE = {}


def _core(seed, scale):
    key = (seed, scale)
    if key not in _CORE:
        _CORE[key] = Terrain(seed=seed + 303, scale=scale, octaves=2, amplitude=2.0,
                             persistence=0.5)
    return _CORE[key]


def _site(seed, salt, i, j, cell):
    return ((i + 0.1 + 0.8 * hash01(seed, salt, i, j)) * cell,
            (j + 0.1 + 0.8 * hash01(seed, salt + 1, i, j)) * cell)


def _f2_minus_f1(seed, salt, x, z, cell):
    ci, cj = math.floor(x / cell), math.floor(z / cell)
    d = []
    for i in range(ci - 1, ci + 2):
        for j in range(cj - 1, cj + 2):
            sx, sz = _site(seed, salt, i, j, cell)
            d.append(math.hypot(sx - x, sz - z))
    d.sort()
    return d[1] - d[0]


@register("voronoi_towers")
def voronoi_towers(ctx, params):
    p = {**DEFAULTS, **params}
    cell, gap = float(p["cell_m"]), float(p["gap_m"])
    seed, salt = ctx.seed, ctx.salt
    out, sites = [], {}

    def site(a, b):
        if (a, b) not in sites:
            sites[a, b] = _site(seed, salt, a, b, cell)
        return sites[a, b]

    for i in range(math.floor(ctx.x0 / cell) - 1, math.floor(ctx.x1 / cell) + 1):
        for j in range(math.floor(ctx.z0 / cell) - 1, math.floor(ctx.z1 / cell) + 1):
            sx, sz = site(i, j)
            if not ctx.owns(sx, sz):
                continue
            r = 1.5 * cell
            poly = [(sx - r, sz - r), (sx + r, sz - r), (sx + r, sz + r), (sx - r, sz + r)]
            for a in range(i - 2, i + 3):
                for b in range(j - 2, j + 3):
                    if (a, b) == (i, j) or poly is None:
                        continue
                    nx, nz = site(a, b)
                    dx, dz = nx - sx, nz - sz
                    dist = math.hypot(dx, dz)
                    ux, uz = dx / dist, dz / dist
                    poly = clip(poly, ux, uz, ux * 0.5 * (sx + nx) + uz * 0.5 * (sz + nz)
                                - 0.5 * gap)
            if poly is None or area(poly) < p["min_area_m2"]:
                continue
            r = hash01(seed, salt + 3, i, j)
            core = 0.5 + 0.5 * min(1.0, max(-1.0, _core(seed, p["core_scale_m"]).height(sx, sz)))
            q = min(1.0, _f2_minus_f1(seed, salt + 5, sx, sz, p["cluster_m"])
                    / (0.5 * p["cluster_m"]))
            k = (0.12 + 0.88 * core ** 1.5) * (0.35 + 0.65 * q) * (0.45 + 0.55 * r * r)
            body = pick(hash01(seed, salt + 7, i, j), NEON)
            dk = lerp(hash01(seed, salt + 8, i, j), p["darken"])
            color = tuple(c * (1.0 - dk) + d * dk for c, d in zip(body, DARK, strict=True))
            neon = hash01(seed, salt + 9, i, j) < p["neon_odds"]
            out.append(Instance(gen="voronoi_towers", kind="tower", pos=(sx, sz),
                                yaw=0.0, footprint=ccw(poly), height=lerp(k, p["height_m"]),
                                color=color,
                                trim=pick(hash01(seed, salt + 10, i, j), NEON) if neon else None,
                                emissive=neon))
    return out

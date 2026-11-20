"""scatter: sparse odds and ends (rocks, cacti, poles, trees, shrubs).

Candidates sit on a jittered ``spacing_m`` lattice, each kept with ``odds``
and given one of ``kinds``: a low-poly prism (a spire for trees).
"""

import math

from worldgen.hashing import hash01
from worldgen.structures.base import Instance, lerp, pick, prism, register

#        sides, radius m, height m, roof (spire height share), colour
KINDS = {
    "rock": (5, (0.8, 2.2), (0.6, 2.0), 0.0, (0.47, 0.42, 0.38)),
    "cactus": (4, (0.35, 0.5), (2.5, 5.0), 0.0, (0.27, 0.5, 0.22)),
    "pole": (4, (0.15, 0.15), (7.0, 9.0), 0.0, (0.3, 0.22, 0.16)),
    "tree": (6, (1.4, 2.6), (1.0, 2.0), 3.0, (0.18, 0.4, 0.16)),
    "shrub": (6, (0.8, 1.5), (0.8, 1.6), 0.0, (0.36, 0.42, 0.2)),
}
DEFAULTS = {"spacing_m": 22.0, "odds": 0.3, "kinds": ["rock"]}


@register("scatter")
def scatter(ctx, params):
    p = {**DEFAULTS, **params}
    sp, seed, salt = float(p["spacing_m"]), ctx.seed, ctx.salt
    kinds = [k for k in p["kinds"] if k in KINDS]
    if len(kinds) != len(p["kinds"]):
        raise ValueError(f"scatter: unknown kind in {p['kinds']}; known: {', '.join(KINDS)}")
    out = []
    for i in range(math.floor(ctx.x0 / sp), math.floor(ctx.x1 / sp) + 1):
        for j in range(math.floor(ctx.z0 / sp), math.floor(ctx.z1 / sp) + 1):
            if hash01(seed, salt, i, j) >= p["odds"]:
                continue
            x = (i + 0.15 + 0.7 * hash01(seed, salt + 1, i, j)) * sp
            z = (j + 0.15 + 0.7 * hash01(seed, salt + 2, i, j)) * sp
            if not ctx.owns(x, z):
                continue
            kind = pick(hash01(seed, salt + 3, i, j), kinds)
            sides, rad, hgt, spire, color = KINDS[kind]
            r = lerp(hash01(seed, salt + 4, i, j), rad)
            h = lerp(hash01(seed, salt + 5, i, j), hgt)
            yaw = hash01(seed, salt + 6, i, j) * 2.0 * math.pi
            shade = 0.85 + 0.3 * hash01(seed, salt + 7, i, j)
            out.append(Instance(gen="scatter", kind=kind, pos=(x, z), yaw=yaw,
                                footprint=prism(x, z, r, sides, yaw), height=h,
                                color=tuple(min(1.0, c * shade) for c in color),
                                roof="spire" if spire else "flat", roof_h=spire * h))
    return out

"""cottages: small gabled houses lining the streets.

Candidates on a jittered ``spacing_m`` lattice are kept with ``odds`` when
the nearest road edge is within ``setback_m``; each house turns its front
(and the ridge of its roof) to that road.
"""

import math

import numpy as np

from worldgen.hashing import hash01
from worldgen.structures.base import Instance, box, lerp, pick, register

WALLS = ((0.86, 0.82, 0.70), (0.62, 0.30, 0.22), (0.92, 0.91, 0.88), (0.58, 0.56, 0.52),
         (0.74, 0.62, 0.45))
ROOFS = ((0.28, 0.30, 0.34), (0.55, 0.26, 0.18), (0.30, 0.22, 0.18))
DEFAULTS = {"spacing_m": 30.0, "odds": 0.6, "setback_m": 16.0, "width_m": (6.0, 9.0),
            "depth_m": (7.0, 10.0), "wall_m": (3.0, 4.5), "roof_m": (2.0, 3.2)}


@register("cottages")
def cottages(ctx, params):
    p = {**DEFAULTS, **params}
    sp, seed, salt = float(p["spacing_m"]), ctx.seed, ctx.salt
    roads = ctx.roads
    out = []
    for i in range(math.floor(ctx.x0 / sp), math.floor(ctx.x1 / sp) + 1):
        for j in range(math.floor(ctx.z0 / sp), math.floor(ctx.z1 / sp) + 1):
            if hash01(seed, salt, i, j) >= p["odds"] or not len(roads[0]):
                continue
            x = (i + 0.1 + 0.8 * hash01(seed, salt + 1, i, j)) * sp
            z = (j + 0.1 + 0.8 * hash01(seed, salt + 2, i, j)) * sp
            if not ctx.owns(x, z):
                continue
            pts, hw = roads
            d = np.hypot(pts[:, 0] - x, pts[:, 1] - z) - hw
            k = int(np.argmin(d))
            w, dep = lerp(hash01(seed, salt + 3, i, j), p["width_m"]), \
                lerp(hash01(seed, salt + 4, i, j), p["depth_m"])
            if not 0.5 * dep < d[k] <= p["setback_m"] + 0.5 * dep:
                continue
            yaw = math.atan2(pts[k, 0] - x, pts[k, 1] - z)      # front to the road
            out.append(Instance(gen="cottages", kind="cottage", pos=(x, z), yaw=yaw,
                                footprint=box(x, z, w, dep, yaw),
                                height=lerp(hash01(seed, salt + 5, i, j), p["wall_m"]),
                                color=pick(hash01(seed, salt + 6, i, j), WALLS), roof="gable",
                                roof_h=lerp(hash01(seed, salt + 7, i, j), p["roof_m"]),
                                roof_color=pick(hash01(seed, salt + 8, i, j), ROOFS)))
    return out

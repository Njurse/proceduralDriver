"""frontage: low shopfront blocks lining both sides of every road, so a
street reads as a continuous wall.

Each road side is cut into ``lot_m`` lots by arc length from the road's start
(so neighbouring chunks agree), every lot kept with ``odds``; a block fills
its lot less a small gap, ``setback_m`` behind the roadside (sidewalk), its
depth and height hashed per lot.  Some carry a neon sign band at awning
height.  Lots that run into a crossing road fail the road filter, which
leaves the corners open.
"""

import math
import zlib

import numpy as np

from worldgen.hashing import hash01
from worldgen.roads.roadside import reach
from worldgen.structures.base import Instance, box, lerp, pick, register
from worldgen.structures.towers import NEON

WALLS = ((0.32, 0.30, 0.36), (0.42, 0.26, 0.30), (0.22, 0.30, 0.38), (0.46, 0.42, 0.36),
         (0.28, 0.36, 0.30), (0.52, 0.34, 0.22), (0.36, 0.36, 0.40))
DEFAULTS = {"lot_m": 14.0, "odds": 0.85, "gap_m": (0.4, 2.0), "depth_m": (6.0, 10.0),
            "height_m": (4.5, 13.0), "setback_m": 0.3, "neon_odds": 0.5, "awning_m": 3.2}


@register("frontage")
def frontage(ctx, params):
    p = {**DEFAULTS, **params}
    lot, seed, salt = float(p["lot_m"]), ctx.seed, ctx.salt
    out = []
    for r in ctx.road_list:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
        n = int(s[-1] // lot)
        if n < 1:
            continue
        mid = (np.arange(n) + 0.5) * lot
        k = np.clip(np.searchsorted(s, mid) - 1, 0, len(r.dx) - 1)
        t = (mid - s[k]) / np.maximum(s[k + 1] - s[k], 1e-9)
        cx = r.ax[k] + t * r.dx[k]
        cz = r.az[k] + t * r.dz[k]
        ln = np.hypot(r.dx[k], r.dz[k])
        nx, nz = -r.dz[k] / ln, r.dx[k] / ln            # left normal
        far = reach(r) + p["setback_m"] + p["depth_m"][1]
        near = ((cx + far >= ctx.x0) & (cx - far < ctx.x1)
                & (cz + far >= ctx.z0) & (cz - far < ctx.z1))
        rid = zlib.crc32(repr(r.key).encode())
        front = reach(r) + p["setback_m"]
        for j in np.nonzero(near)[0].tolist():
            for side, f in ((0, -1.0), (1, 1.0)):
                key = (rid, 2 * j + side)
                if hash01(seed, salt, *key) >= p["odds"]:
                    continue
                dep = lerp(hash01(seed, salt + 1, *key), p["depth_m"])
                x = cx[j] + f * nx[j] * (front + 0.5 * dep)
                z = cz[j] + f * nz[j] * (front + 0.5 * dep)
                if not ctx.owns(x, z):
                    continue
                w = lot - lerp(hash01(seed, salt + 2, *key), p["gap_m"])
                yaw = math.atan2(f * nx[j], f * nz[j])        # depth away from the road
                h = lerp(hash01(seed, salt + 3, *key), p["height_m"])
                neon = hash01(seed, salt + 4, *key) < p["neon_odds"]
                out.append(Instance(
                    gen="frontage", kind="storefront", pos=(x, z), yaw=yaw,
                    footprint=box(x, z, w, dep, yaw), height=h,
                    color=pick(hash01(seed, salt + 5, *key), WALLS),
                    trim=pick(hash01(seed, salt + 6, *key), NEON) if neon else None,
                    emissive=neon, trim_y=min(p["awning_m"], 0.6 * h) if neon else None))
    return out

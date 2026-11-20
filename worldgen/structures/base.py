"""Structure records, the generator registry and the geometry they share.

A generator is a function ``gen(ctx, params) -> [Instance]`` registered under
a name a biome's JSON can use::

    "structures": [{"gen": "voronoi_towers", "cell_m": 34}]

It proposes instances anchored in ctx's chunk; worldgen.structures keeps
the ones inside the biome, clear of the roads and their roadsides (sidewalks,
verges, walls) and on gentle ground.
Everything is a pure function of (seed, chunk, params): no RNG state.
"""

import math
from dataclasses import dataclass

import numpy as np

GENERATORS = {}
ROOFS = ("flat", "gable", "spire")


def register(name):
    """Decorator: make ``gen`` available to biome JSON as ``name``."""
    def wrap(gen):
        GENERATORS[name] = gen
        return gen
    return wrap


@dataclass(eq=False)
class Instance:
    """One structure: a convex footprint extruded up from the ground.

    ``footprint`` (k, 2) is world (x, z), counter-clockwise seen from above
    (+y); a gable roof's ridge runs parallel to its first edge.  Walls rise
    ``height`` from ``base_y``; the roof adds ``roof_h`` (gable ridge or
    spire apex).  ``trim`` is an optional band near the top of the walls,
    lit by itself when ``emissive`` (neon)."""

    gen: str
    kind: str
    pos: tuple                 # (x, z) anchor
    yaw: float                 # radians, heading as in the physics (atan2(dx, dz))
    footprint: np.ndarray      # (k, 2)
    height: float
    color: tuple               # walls RGB
    roof: str = "flat"         # one of ROOFS
    roof_h: float = 0.0
    roof_color: tuple | None = None
    trim: tuple | None = None
    emissive: bool = False
    trim_y: float | None = None  # the trim as an awning this high; none: near the top
    base_y: float = 0.0        # set by the filter: the lowest ground under the footprint


@dataclass
class Context:
    """What a generator sees: the chunk box and the world, seeded."""

    ground: object             # worldgen.query.WorldGround
    seed: int
    salt: int                  # per (biome, rule): two rules never share hashes
    ci: int
    ck: int
    x0: float
    z0: float
    x1: float
    z1: float
    # road centre points (n, 2) and how far each road's roadside reaches from its centre
    roads: tuple = (np.zeros((0, 2)), np.zeros(0))
    road_list: tuple = ()      # the worldgen.roads.road.Road objects near the chunk

    def owns(self, x, z):
        """True where (x, z) anchors in this chunk."""
        return (self.x0 <= x) & (x < self.x1) & (self.z0 <= z) & (z < self.z1)


def ccw(poly):
    """``poly`` ordered counter-clockwise seen from above (+y), as an array."""
    poly = np.asarray(poly, float)
    x, z = poly[:, 0], poly[:, 1]
    area = np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z)
    # from +y looking down, x right and z down the screen: positive shoelace is clockwise
    return poly[::-1].copy() if area > 0 else poly


def area(poly):
    poly = np.asarray(poly, float)
    x, z = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))


def clip(poly, nx, nz, c):
    """Convex ``poly`` [(x, z)] cut to the half-plane nx * x + nz * z <= c
    (plain floats: a tower cell takes two dozen cuts)."""
    out = []
    n = len(poly)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = nx * p[0] + nz * p[1] - c, nx * q[0] + nz * q[1] - c
        if fp <= 0.0:
            out.append(p)
        if (fp < 0.0) != (fq < 0.0) and fp != fq:
            t = fp / (fp - fq)
            out.append((p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t))
    return out if len(out) >= 3 else None


def box(cx, cz, w, d, yaw):
    """Rectangle w (across, local x) by d (along, local z) turned to ``yaw``;
    its first edge runs along local x (a gable ridge parallel to the front)."""
    s, c = math.sin(yaw), math.cos(yaw)
    loc = np.array([[-w / 2, -d / 2], [w / 2, -d / 2], [w / 2, d / 2], [-w / 2, d / 2]])
    x = cx + loc[:, 0] * c + loc[:, 1] * s
    z = cz - loc[:, 0] * s + loc[:, 1] * c
    return ccw(np.stack([x, z], axis=1))


def prism(cx, cz, r, sides, yaw):
    a = yaw + np.arange(sides) * (2.0 * math.pi / sides)
    return ccw(np.stack([cx + r * np.sin(a), cz + r * np.cos(a)], axis=1))


def poly_distance(poly, px, pz):
    """Distance from each point to convex ``poly`` (0 inside)."""
    a = poly
    e = np.concatenate([poly[1:], poly[:1]]) - a
    ex, ez = e[:, 0], e[:, 1]
    rx, rz = px[:, None] - a[None, :, 0], pz[:, None] - a[None, :, 1]
    cross = ex[None, :] * rz - ez[None, :] * rx
    inside = (cross <= 0.0).all(axis=1) | (cross >= 0.0).all(axis=1)
    t = np.clip((rx * ex + rz * ez) / np.maximum(ex * ex + ez * ez, 1e-12), 0.0, 1.0)
    d = np.hypot(rx - t * ex, rz - t * ez).min(axis=1)
    return np.where(inside, 0.0, d)


def seg_distance(poly, a, b):
    """Distance from convex ``poly`` to each segment a[i] -> b[i] (0 where they meet)."""
    d = b - a
    ln2 = np.sum(d * d, axis=1)
    rx = poly[:, 0][:, None] - a[None, :, 0]
    rz = poly[:, 1][:, None] - a[None, :, 1]
    t = np.clip((rx * d[:, 0] + rz * d[:, 1]) / np.maximum(ln2, 1e-12), 0.0, 1.0)
    out = np.hypot(rx - t * d[:, 0], rz - t * d[:, 1]).min(axis=0)
    out = np.minimum(out, poly_distance(poly, a[:, 0], a[:, 1]))
    out = np.minimum(out, poly_distance(poly, b[:, 0], b[:, 1]))
    # a segment through the polygon with both ends outside crosses an edge; its
    # ends are then nearer the polygon than its length, so only those are tested
    sel = np.nonzero(out * out < ln2)[0]
    if len(sel):
        p, e = poly, np.concatenate([poly[1:], poly[:1]]) - poly
        a, d = a[sel], d[sel]
        b = a + d

        def side(ox, oz, vx, vz, x, z):     # sign of v x (point - o)
            return np.sign(vx * (z - oz) - vz * (x - ox))

        px, pz, ex, ez = (v[:, None] for v in (p[:, 0], p[:, 1], e[:, 0], e[:, 1]))
        s1 = side(px, pz, ex, ez, a[:, 0], a[:, 1])
        s2 = side(px, pz, ex, ez, b[:, 0], b[:, 1])
        s3 = side(a[:, 0], a[:, 1], d[:, 0], d[:, 1], px, pz)
        s4 = side(a[:, 0], a[:, 1], d[:, 0], d[:, 1], px + ex, pz + ez)
        hit = ((s1 * s2 < 0) & (s3 * s4 < 0)).any(axis=0)
        out[sel[hit]] = 0.0
    return out


def pick(h, options):
    """options[floor(h * len)] for a hash h in [0, 1)."""
    return options[min(int(h * len(options)), len(options) - 1)]


def lerp(h, lo_hi):
    lo, hi = lo_hi
    return lo + (hi - lo) * h

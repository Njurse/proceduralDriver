"""Structure instances as one ChunkMesh: extruded footprints, flat, gable or
spire roofs and trim bands, flat-shaded into the vertex colours (the game
draws them unlit; emissive trims at full brightness)."""

import math

import numpy as np

from worldgen.chunk import ChunkMesh

_LIGHT = (-1.0 / 1.99, 1.4 / 1.99, 1.0 / 1.99)    # unit (|(-1, 1.4, 1)| = 1.99)


class _Builder:
    """Triangles as flat lists (plain floats: a chunk has a few thousand)."""

    def __init__(self):
        self.pos, self.normal, self.color = [], [], []

    def tri(self, a, b, c, inside, rgb, lit=True):
        """One triangle facing away from ``inside``, wound like worldgen.chunk's
        (its physics-frame cross product points into the solid)."""
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln < 1e-9:
            return
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        ox = (a[0] + b[0] + c[0]) / 3.0 - inside[0]
        oy = (a[1] + b[1] + c[1]) / 3.0 - inside[1]
        oz = (a[2] + b[2] + c[2]) / 3.0 - inside[2]
        if nx * ox + ny * oy + nz * oz > 0.0:
            b, c = c, b
        else:
            nx, ny, nz = -nx, -ny, -nz
        k = 0.5 + 0.5 * max(0.0, nx * _LIGHT[0] + ny * _LIGHT[1] + nz * _LIGHT[2]) \
            if lit else 1.0
        col = (min(1.0, rgb[0] * k), min(1.0, rgb[1] * k), min(1.0, rgb[2] * k), 1.0)
        n = (nx, ny, nz)
        self.pos += [a, b, c]
        self.normal += [n, n, n]
        self.color += [col, col, col]

    def quad(self, a, b, c, d, inside, rgb, lit=True):
        self.tri(a, b, c, inside, rgb, lit)
        self.tri(a, c, d, inside, rgb, lit)


def _p(xz, y):
    return (float(xz[0]), y, float(xz[1]))


def _instance(bld, s):
    arr, y0 = np.asarray(s.footprint, float), s.base_y
    fp = [(float(x), float(z)) for x, z in arr]
    top = y0 + (s.height + 0.2)
    c = tuple(float(v) for v in arr.mean(axis=0))
    inside = (c[0], y0 + 0.5 * (top - y0), c[1])
    k = len(fp)
    for e in range(k):
        a, b = fp[e], fp[(e + 1) % k]
        bld.quad(_p(a, y0), _p(b, y0), _p(b, top), _p(a, top), inside, s.color)
    roof = s.roof_color or s.color
    if s.roof == "gable" and k == 4 and s.roof_h > 0.0:
        ridge = top + s.roof_h
        m30 = (0.5 * (fp[3][0] + fp[0][0]), 0.5 * (fp[3][1] + fp[0][1]))
        m12 = (0.5 * (fp[1][0] + fp[2][0]), 0.5 * (fp[1][1] + fp[2][1]))
        mid = (c[0], top + 0.3 * s.roof_h, c[1])
        bld.quad(_p(fp[0], top), _p(fp[1], top), _p(m12, ridge), _p(m30, ridge), mid, roof)
        bld.quad(_p(fp[2], top), _p(fp[3], top), _p(m30, ridge), _p(m12, ridge), mid, roof)
        bld.tri(_p(fp[3], top), _p(fp[0], top), _p(m30, ridge), mid, s.color)
        bld.tri(_p(fp[1], top), _p(fp[2], top), _p(m12, ridge), mid, s.color)
    elif s.roof == "spire" and s.roof_h > 0.0:
        apex = (c[0], top + s.roof_h, c[1])
        mid = (c[0], top + 0.3 * s.roof_h, c[1])
        for e in range(k):
            bld.tri(_p(fp[e], top), _p(fp[(e + 1) % k], top), apex, mid, roof)
    else:
        below = (c[0], top - 1.0, c[1])
        for e in range(k):
            bld.tri(_p(c, top), _p(fp[e], top), _p(fp[(e + 1) % k], top), below, roof)
    if s.trim is not None and s.height >= 8.0:
        ya = y0 + 0.86 * s.height
        yb = ya + max(0.8, 0.03 * s.height)
        out = arr - np.array(c)
        grow = arr + 0.15 * out / np.maximum(np.linalg.norm(out, axis=1)[:, None], 1e-9)
        for e in range(k):
            a, b = grow[e], grow[(e + 1) % k]
            bld.quad(_p(a, ya), _p(b, ya), _p(b, yb), _p(a, yb), inside, s.trim,
                     lit=not s.emissive)


def structures_mesh(instances):
    """One ChunkMesh for ``instances`` (each with base_y set), or None."""
    if not instances:
        return None
    bld = _Builder()
    for s in instances:
        _instance(bld, s)
    n = len(bld.pos)
    return ChunkMesh(pos=np.array(bld.pos), normal=np.array(bld.normal),
                     color=np.array(bld.color), uv=np.zeros((n, 2)),
                     tris=np.arange(n).reshape(-1, 3))

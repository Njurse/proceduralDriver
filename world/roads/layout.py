"""The road graph of one super-cell: nodes and edges (numpy, no Panda3D).

Nodes: one candidate per PITCH_M lattice cell, which reads the road field at
its own spot (``field``: the biomes' rules blended by the terrain's region
weights).  With odds ``grid`` it is a grid node: snapped to the ``spacing``
lattice turned to the region heading, kept only by the cell that holds the
snapped point.  Otherwise it is an organic seed, kept with odds
(PITCH_M / spacing)^2 and jittered round its cell.  Every node depends only
on the field where it stands, so nodes are deterministic and seamless.

Edges: a minimum spanning tree over neighbour pairs (cost = length x (1 +
GRADE_COST x grade), so roads take the gentle way and every node is
reached), plus the one-block axis pairs between grid nodes, plus a few short
organic cross-links for loops.  A non-tree edge leaving a node within
MIN_ANGLE_DEG of another edge is dropped: no near-parallel roads.

Super-cells are SUPER_M squares; each graph depends only on (seed, cell).
Neighbours are joined by the cheapest crossing pair per BORDER_RUN_M of
shared border (and all aligned grid pairs), from the two graphs alone.
"""

import math
import time
from dataclasses import dataclass

import numpy as np

from world.regions import _hash01 as _hash_arr

SUPER_M = 4096.0
PITCH_M = 128.0          # candidate lattice: also the closest two junctions get
LINK_RANGE = 1.6         # candidate edges reach this many local spacings
CROSS_ODDS = 0.12        # short organic non-tree candidates kept as loops
MIN_ANGLE_DEG = 25.0
GRADE_COST = 8.0
BORDER_RUN_M = 1024.0
_N = round(SUPER_M / PITCH_M)
_MASK = (1 << 64) - 1


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


def field(regions, x, z):
    """(grid, heading, spacing) arrays at points: each region's biome rules,
    blended by the terrain's region weights.  Each region adds its grid
    strength times the unit vector of 4 x its heading (a square grid repeats
    every quarter turn); the sum's length is the grid strength and its angle
    / 4 the heading, so where regions disagree the grid fades."""
    def per_site(pos, owner):
        rules = [regions.biomes[b].roads for b in owner]
        turn = _hash_arr(regions.seed, 21, np.round(pos[:, 0]), np.round(pos[:, 1])) * 2 * np.pi
        a4 = np.array([turn[k] if r.grid_heading_deg is None
                       else math.radians(4.0 * r.grid_heading_deg) for k, r in enumerate(rules)])
        g = np.array([r.grid for r in rules])
        return np.stack([g * np.cos(a4), g * np.sin(a4),
                         [r.grid_spacing_m for r in rules]], axis=1)

    v = regions.site_mean(x, z, per_site)
    return np.hypot(v[:, 0], v[:, 1]), 0.25 * np.arctan2(v[:, 1], v[:, 0]), v[:, 2]


@dataclass
class Graph:
    x: np.ndarray
    z: np.ndarray
    h: np.ndarray
    grid: np.ndarray       # bool: a grid node
    g: np.ndarray          # grid strength where the node stands
    heading: np.ndarray
    spacing: np.ndarray
    edges: list            # [(i, j)]
    ms: float = 0.0


def _rot(x, z, a):
    c, s = np.cos(a), np.sin(a)
    return c * x - s * z, s * x + c * z


def grow(regions, seed, ci, cj):
    """The Graph of super-cell (ci, cj)."""
    t0 = time.perf_counter()
    ii, jj = np.meshgrid(np.arange(_N) + ci * _N, np.arange(_N) + cj * _N, indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    qx, qz = (ii + 0.5) * PITCH_M, (jj + 0.5) * PITCH_M
    g, th, sp = field(regions, qx, qz)
    is_grid = _hash_arr(seed, 31, ii, jj) < g
    # grid: snap to the turned lattice; keep when this cell holds the point
    u, w = _rot(qx, qz, -th)
    gx, gz = _rot(np.round(u / sp) * sp, np.round(w / sp) * sp, th)
    own = (np.floor(gx / PITCH_M) == ii) & (np.floor(gz / PITCH_M) == jj)
    # organic: thinned to the spacing, jittered round the cell
    keep = _hash_arr(seed, 32, ii, jj) < (PITCH_M / sp) ** 2
    ox = qx + (_hash_arr(seed, 33, ii, jj) - 0.5) * 0.9 * PITCH_M
    oz = qz + (_hash_arr(seed, 34, ii, jj) - 0.5) * 0.9 * PITCH_M
    sel = np.where(is_grid, own, keep)
    x = np.where(is_grid, gx, ox)[sel]
    z = np.where(is_grid, gz, oz)[sel]
    graph = Graph(x, z, np.asarray(regions.height(x, z), float), is_grid[sel], g[sel],
                  th[sel], sp[sel], [])
    graph.edges = _edges(graph, seed, ci, cj)
    graph.ms = (time.perf_counter() - t0) * 1000.0
    return graph


def _candidates(ax, az, ah, asp, bx, bz, bh, bsp):
    """Pairs (i, j, length, cost) between point sets a and b within range."""
    dx, dz = bx[None, :] - ax[:, None], bz[None, :] - az[:, None]
    d = np.hypot(dx, dz)
    i, j = np.nonzero((d < LINK_RANGE * np.maximum(asp[:, None], bsp[None, :])) & (d > 1.0))
    d = d[i, j]
    return i, j, d, d * (1.0 + GRADE_COST * np.abs(bh[j] - ah[i]) / d)


def _aligned(gr, i, j, d, gr_b=None):
    """Grid pairs one block apart along a lattice axis."""
    b = gr if gr_b is None else gr_b
    phi = np.arctan2(b.z[j] - gr.z[i], b.x[j] - gr.x[i]) - gr.heading[i]
    off = np.abs((phi + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    return gr.grid[i] & b.grid[j] & (off < 0.05) & (np.abs(d / gr.spacing[i] - 1.0) < 0.2)


def _edges(gr, seed, ci, cj):
    n = len(gr.x)
    i, j, d, cost = _candidates(gr.x, gr.z, gr.h, gr.spacing, gr.x, gr.z, gr.h, gr.spacing)
    m = i < j
    i, j, d, cost = i[m], j[m], d[m], cost[m]
    parent = list(range(n))

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    tree, extra = [], []
    aligned = _aligned(gr, i, j, d)
    for k in np.argsort(cost, kind="stable"):
        a, b = root(i[k]), root(j[k])
        if a != b:
            parent[a] = b
            tree.append((int(i[k]), int(j[k])))
        elif aligned[k] or (d[k] < 1.2 * gr.spacing[i[k]] and hash01(
                seed, 35, ci, cj, int(i[k]), int(j[k])) < CROSS_ODDS * (1.0 - gr.g[i[k]])):
            extra.append((int(i[k]), int(j[k])))
    out = tree[:]
    dirs = {}
    for a, b in tree:
        _add_dir(dirs, gr, a, b)
    for a, b in extra:
        if _clear(dirs, gr, a, b):
            _add_dir(dirs, gr, a, b)
            out.append((a, b))
    return out


def _angle(gr, a, b):
    return math.atan2(gr.z[b] - gr.z[a], gr.x[b] - gr.x[a])


def _add_dir(dirs, gr, a, b):
    dirs.setdefault(a, []).append(_angle(gr, a, b))
    dirs.setdefault(b, []).append(_angle(gr, b, a))


def _clear(dirs, gr, a, b):
    """No edge leaves a or b within MIN_ANGLE_DEG of this one."""
    lim = math.radians(MIN_ANGLE_DEG)
    for p, q in ((a, b), (b, a)):
        t = _angle(gr, p, q)
        for u in dirs.get(p, ()):
            if abs((t - u + math.pi) % (2 * math.pi) - math.pi) < lim:
                return False
    return True


def stitch(ga, gb, axis, line):
    """Edges (i in ga, j in gb) across the border ``line`` on ``axis``
    (0: x, 1: z) between two neighbouring super-cells."""
    pa, pb = (ga.x, ga.z)[axis], (gb.x, gb.z)[axis]
    reach = LINK_RANGE * max(ga.spacing.max(initial=0.0), gb.spacing.max(initial=0.0))
    sa, sb = np.nonzero(np.abs(pa - line) < reach)[0], np.nonzero(np.abs(pb - line) < reach)[0]
    if not len(sa) or not len(sb):
        return []
    i, j, d, cost = _candidates(ga.x[sa], ga.z[sa], ga.h[sa], ga.spacing[sa],
                                gb.x[sb], gb.z[sb], gb.h[sb], gb.spacing[sb])
    i, j = sa[i], sb[j]
    m = _aligned(ga, i, j, d, gb)
    out = set(zip(i[m].tolist(), j[m].tolist(), strict=True))
    along = 0.5 * ((ga.z, ga.x)[axis][i] + (gb.z, gb.x)[axis][j])
    best = {}
    for k in np.argsort(cost, kind="stable"):
        best.setdefault(math.floor(along[k] / BORDER_RUN_M), (int(i[k]), int(j[k])))
    return sorted(out | set(best.values()))

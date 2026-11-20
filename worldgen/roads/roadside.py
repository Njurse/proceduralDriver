"""Roadside strips for a chunk: what each road's biome lays along both sides
(``roads.roadside`` in its JSON), as numpy arrays like mesh.strips.

Across a side, inner (road edge) to outer: a kerb face up to ``lift_m``, a
kerb band ``curb_m`` wide, the strip to ``side_m`` (lighter towards its outer
edge by ``edge_light``), then either an outer face back down or a low wall
``wall_w_m`` wide and ``wall_h_m`` tall.  The edges are offset from the road's
centre line with mitred vertex normals, so neighbouring segments meet
without gaps; a segment's side is left out where it would cross another
road's surface, which stops the strips at junctions.  Visual only: the
ground and the physics never see it."""

import numpy as np

_JUNCTION_PAD_M = 0.25   # a side stops this far short of another road's surface
_SINK_M = 0.1            # faces reach this far below the ground to hide seams


def reach(road):
    """Road edge to the outermost roadside, from the centre line."""
    return road.half_width + road.biome.roads.roadside.reach_m


def vertex_normals(pts):
    """Per-vertex unit left normals (-dz, dx) of polyline ``pts``, mitred:
    inner vertices take the bisector scaled by 1 / cos(half the turn), at most 2."""
    seg = np.diff(pts, axis=0)
    seg /= np.linalg.norm(seg, axis=1, keepdims=True)
    n = np.stack([-seg[:, 1], seg[:, 0]], axis=1)
    out = np.empty((len(pts), 2))
    out[0], out[-1] = n[0], n[-1]
    if len(n) > 1:
        mid = n[:-1] + n[1:]
        mid /= np.maximum(np.linalg.norm(mid, axis=1, keepdims=True), 1e-9)
        cos = np.sum(mid * n[:-1], axis=1, keepdims=True)
        out[1:-1] = mid / np.maximum(cos, 0.5)
    return out


def _bands(rs, hw):
    """(offset a, offset b, dy a, dy b, rgb a, rgb b, face) across one side;
    face 0 is a top (ground normal), -1 / +1 a vertical face looking at / away
    from the road."""
    lit = tuple(min(1.0, c * (1.0 + rs.edge_light)) for c in rs.color)
    lift, top = rs.lift_m, hw + rs.side_m
    out = []
    if lift > 0.0:
        out.append((hw, hw, 0.0, lift, rs.curb_color, rs.curb_color, -1))
    if rs.curb_m > 0.0:
        out.append((hw, hw + rs.curb_m, lift, lift, rs.curb_color, rs.curb_color, 0))
    out.append((hw + rs.curb_m, top, lift, lift, rs.color, lit, 0))
    if rs.wall_h_m > 0.0:
        h, w, c = lift + rs.wall_h_m, rs.wall_w_m, rs.wall_color
        dark = tuple(0.85 * v for v in c)
        out += [(top, top, -_SINK_M, h, dark, c, -1), (top, top + w, h, h, c, c, 0),
                (top + w, top + w, h, -_SINK_M, c, dark, 1)]
    elif lift > 0.0:
        out.append((top, top, lift, -_SINK_M, lit, lit, 1))
    return out


def strips(roads, ground, x0, z0, x1, z1, lift):
    """Roadside geometry for the segments of ``roads`` whose midpoint lies in
    [x0, x1) x [z0, z1): (pos, normal, color, uv, tris) or None.  ``lift`` is
    the road strips' own lift, which the roadside heights add to."""
    jobs = []    # (road, segment indices, side) after the junction cut
    for r in roads:
        rs = r.biome.roads.roadside
        if rs.side_m <= 0.0:
            continue
        mid = 0.5 * (r.pts[:-1] + r.pts[1:])
        sel = np.nonzero((mid[:, 0] >= x0) & (mid[:, 0] < x1)
                         & (mid[:, 1] >= z0) & (mid[:, 1] < z1))[0]
        if not len(sel):
            continue
        nv = vertex_normals(r.pts)
        for f in (-1.0, 1.0):
            corners = np.concatenate([
                r.pts[k] + f * o * nv[k]
                for k in (sel, sel + 1) for o in (r.half_width, reach(r))])
            keep = np.ones(len(sel), bool)
            (bx0, bz0), (bx1, bz1) = corners.min(axis=0), corners.max(axis=0)
            for o in roads:
                if o is r or not o.overlaps(bx0, bz0, bx1, bz1):
                    continue
                t = o.terms(corners[:, 0], corners[:, 1],
                            reach=o.half_width + _JUNCTION_PAD_M)
                if t is not None:
                    hit = (t[0] < o.half_width + _JUNCTION_PAD_M).reshape(4, -1).any(axis=0)
                    keep &= ~hit
            if keep.any():
                jobs.append((r, nv, sel[keep], f))
    if not jobs:
        return None
    xz, dy, face, col, uv, quads = [], [], [], [], [], []
    for r, nv, ks, f in jobs:
        s = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
        for oa, ob, ya, yb, ca, cb, fc in _bands(r.biome.roads.roadside, r.half_width):
            base = sum(len(p) for p in xz)
            p = np.stack([r.pts[ks] + f * oa * nv[ks], r.pts[ks + 1] + f * oa * nv[ks + 1],
                          r.pts[ks + 1] + f * ob * nv[ks + 1], r.pts[ks] + f * ob * nv[ks]],
                         axis=1).reshape(-1, 2)
            m = len(ks)
            xz.append(p)
            dy.append(np.tile([ya, ya, yb, yb], m))
            # vertical faces: horizontal normals, out of the side (f * nv) or back at the road
            hn = (np.stack([nv[ks], nv[ks + 1], nv[ks + 1], nv[ks]], axis=1)
                  .reshape(-1, 2) * f * fc)
            face.append(np.concatenate([hn, np.full((4 * m, 1), float(fc != 0))], axis=1))
            col.append(np.tile([[*ca, 1.0], [*ca, 1.0], [*cb, 1.0], [*cb, 1.0]], (m, 1)))
            across = (ob - oa) if fc == 0 else (yb - ya)
            uv.append(np.stack([np.tile([0.0, 0.0, across, across], m),
                                np.stack([s[ks], s[ks + 1], s[ks + 1], s[ks]], axis=1).ravel()],
                               axis=1))
            q = base + 4 * np.arange(m)[:, None]
            quads.append(np.concatenate([q + [0, 1, 2], q + [0, 2, 3]]))
    xz, dy, face = np.concatenate(xz), np.concatenate(dy), np.concatenate(face)
    h, hx, hz = ground.sample(xz[:, 0], xz[:, 1])
    gn = np.stack([-hx, np.ones_like(h), -hz], axis=-1)
    gn /= np.linalg.norm(gn, axis=-1, keepdims=True)
    hv = np.stack([face[:, 0], np.zeros(len(h)), face[:, 1]], axis=-1)
    hv /= np.maximum(np.linalg.norm(hv, axis=-1, keepdims=True), 1e-9)
    n = np.where(face[:, 2:3] > 0.0, hv, gn)
    p3 = np.stack([xz[:, 0], h + lift + dy, xz[:, 1]], axis=-1)
    tris = np.concatenate(quads)
    # wound like the road strips: the physics-frame cross product points into the solid
    a, b, c = (p3[tris[:, i]] for i in range(3))
    cross = np.cross(b - a, c - a)
    tris = np.where((np.sum(cross * n[tris[:, 0]], axis=1) > 0.0)[:, None],
                    tris[:, [0, 2, 1]], tris)
    return p3, n, np.concatenate(col), np.concatenate(uv), tris



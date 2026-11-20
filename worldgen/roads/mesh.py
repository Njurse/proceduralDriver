"""Road strips for a chunk: a quad per segment, a round fan on the outside
of each bend (the flattened ground is round there too, so a sharp corner
neither narrows nor lets the shoulder poke through), plus dashed centre
markings, as numpy arrays (world.chunks turns them into Panda3D geometry)."""

import math

import numpy as np

MARKING_HALF_WIDTH = 0.12  # dashed centre line half-width (m)
MARKING_LIFT = 0.02        # centre-line lift above the road strip


def strips(roads, ground, x0, z0, x1, z1, lift):
    """Road strip geometry for the segments of ``roads`` whose midpoint
    lies in [x0, x1) x [z0, z1): (pos, normal, color, uv, tris) arrays or
    None.  Vertices sit ``lift`` above ``ground``, the flattened surface."""
    pos, col, uv, tris, lifts = [], [], [], [], []
    for r in roads:
        mid = 0.5 * (r.pts[:-1] + r.pts[1:])
        sel = np.nonzero(
            (mid[:, 0] >= x0) & (mid[:, 0] < x1)
            & (mid[:, 1] >= z0) & (mid[:, 1] < z1))[0]
        if not len(sel):
            continue
        seg  = np.diff(r.pts, axis=0)
        seg /= np.linalg.norm(seg, axis=1, keepdims=True)
        side = np.stack([-seg[:, 1], seg[:, 0]], axis=1) * r.half_width   # per segment
        s    = np.concatenate([[0.0], np.cumsum(np.hypot(r.dx, r.dz))])
        for k in sel:
            base = sum(len(p) for p in pos)
            quad = []
            for p in (k, k + 1):
                for f in (-1.0, 0.0, 1.0):
                    quad.append((*(r.pts[p] + f * side[k]), f, s[p]))
            quad = np.array(quad)
            pos.append(quad[:, :2])
            lifts.extend([0.0] * 6)
            shade = 1.0 - 0.12 * np.abs(quad[:, 2:3])
            col.append(np.concatenate(
                [np.array(r.color) * shade, np.ones((6, 1))], axis=1))
            uv.append(np.stack(
                [(quad[:, 2] + 1.0) * r.half_width, quad[:, 3]], axis=1))
            tris.append(base + np.array(
                [[0, 1, 3], [1, 4, 3], [1, 2, 4], [2, 5, 4]]))
            if r.markings is not None and k % 2 == 0:
                f    = MARKING_HALF_WIDTH / r.half_width
                dash = np.array([
                    (*(r.pts[p] + g * f * side[k]), 0.0, s[p])
                    for p in (k, k + 1) for g in (-1.0, 1.0)])
                pos.append(dash[:, :2])
                lifts.extend([MARKING_LIFT] * 4)
                col.append(np.concatenate(
                    [np.tile(r.markings, (4, 1)), np.ones((4, 1))], axis=1))
                uv.append(dash[:, [2, 3]])
                tris.append(base + 6 + np.array([[0, 1, 2], [1, 3, 2]]))
            if k + 1 < len(seg):
                _bend(r, k, side, s, pos, col, uv, tris, lifts)
    if not pos:
        return None
    xz  = np.concatenate(pos)
    h, hx, hz = ground.sample(xz[:, 0], xz[:, 1])
    n   = np.stack([-hx, np.ones_like(h), -hz], axis=-1)
    n  /= np.linalg.norm(n, axis=-1, keepdims=True)
    p3  = np.stack(
        [xz[:, 0], h + lift + np.array(lifts), xz[:, 1]], axis=-1)
    tris = np.concatenate(tris)
    a, b, c = (xz[tris[:, i]] for i in range(3))
    cross = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
           - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    tris = np.where((cross < 0.0)[:, None], tris[:, [0, 2, 1]], tris)
    return p3, n, np.concatenate(col), np.concatenate(uv), tris



def _bend(r, k, side, s, pos, col, uv, tris, lifts):
    """Fan on the outside of the bend at r.pts[k + 1], between the side
    edges of segments k and k + 1 (nothing on a straight)."""
    a, b = side[k], side[k + 1]
    turn = math.atan2(a[0] * b[1] - a[1] * b[0], a @ b)
    if abs(turn) < 0.01:
        return
    f = -1.0 if turn > 0.0 else 1.0          # the outside edge
    c = r.pts[k + 1]
    n = max(1, math.ceil(abs(turn) / math.radians(15.0)))
    base = sum(len(p) for p in pos)
    ang = math.atan2(a[1], a[0]) + np.linspace(0.0, turn, n + 1)
    rim = np.stack([np.cos(ang), np.sin(ang)], axis=1) * f * r.half_width
    pos.append(np.concatenate([c[None, :], c + rim]))
    lifts.extend([0.0] * (n + 2))
    shade = np.concatenate([[1.0], np.full(n + 1, 0.88)])[:, None]
    col.append(np.concatenate([np.array(r.color) * shade, np.ones((n + 2, 1))], axis=1))
    uv.append(np.stack([np.concatenate([[r.half_width], np.full(n + 1, (f + 1.0) * r.half_width)]),
                        np.full(n + 2, s[k + 1])], axis=1))
    tris.append(base + np.stack([np.zeros(n, int), np.arange(1, n + 1), np.arange(2, n + 2)],
                                axis=1))

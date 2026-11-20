"""Terrain chunks as plain numpy meshes (no Panda3D).

Chunk (i, k) covers x in [i * size, (i + 1) * size] and z in
[k * size, (k + 1) * size]; ``chunk_mesh`` grids it for any Ground, painted
by the ground's ``colors`` (or the default palette).  The game turns these
into GeomNodes (game/world_view.py).
"""

import math
from dataclasses import dataclass

import numpy as np

from worldgen.palette import terrain_colors

CHUNK_SIZE_M = 64.0      # metres per square chunk
CHUNK_RES = 32           # quads per chunk side (2 m spacing)
DETAIL_TILE_M = 8.0      # metres per repeat of the detail texture


@dataclass
class ChunkMesh:
    """Vertices in physics coordinates (x, y up, z)."""

    pos: np.ndarray      # (n, 3)
    normal: np.ndarray   # (n, 3)
    color: np.ndarray    # (n, 4)
    uv: np.ndarray       # (n, 2)
    tris: np.ndarray     # (m, 3) vertex indices


def chunk_mesh(terrain, ci, ck, size=CHUNK_SIZE_M, res=CHUNK_RES):
    """Grid mesh of chunk (ci, ck): (res + 1)^2 vertices, 2 res^2 triangles.

    Vertex x/z are computed as (chunk * res + index) * step so neighbouring
    chunks produce bit-identical edge vertices.
    """
    size, res = float(size), int(res)
    step = size / res
    idx = np.arange(res + 1)
    gx, gz = np.meshgrid((ci * res + idx) * step, (ck * res + idx) * step)
    x, z = gx.ravel(), gz.ravel()
    h, dhx, dhz = terrain.sample(x, z)
    n = np.stack([-dhx, np.ones_like(h), -dhz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    pos = np.stack([x, h, z], axis=-1)
    if hasattr(terrain, "colors"):  # biome-aware grounds paint themselves
        color = terrain.colors(x, z, h, n)
    else:
        color = terrain_colors(h, n, terrain.amplitude)
    tile = getattr(terrain, "detail_tile_m", DETAIL_TILE_M)

    row = res + 1
    q = (np.arange(res)[:, None] * row + np.arange(res)[None, :]).ravel()
    # two triangles per quad, counter-clockwise seen from above in Panda's
    # (x, y = physics z, z up) frame, so the top faces survive back-face culling
    tris = np.concatenate([np.stack([q, q + 1, q + row], axis=-1),
                           np.stack([q + 1, q + row + 1, q + row], axis=-1)])
    return ChunkMesh(pos=pos, normal=n, color=color,
                     uv=np.stack([x, z], axis=-1) / tile, tris=tris)


def chunks_around(x, z, radius, size=CHUNK_SIZE_M):
    """Chunk keys within ``radius`` chunks of the one under (x, z), nearest first."""
    size = float(size)
    ci, ck = math.floor(x / size), math.floor(z / size)
    keys = [(ci + di, ck + dk) for di in range(-radius, radius + 1)
            for dk in range(-radius, radius + 1)]
    return sorted(keys, key=lambda k: (k[0] - ci) ** 2 + (k[1] - ck) ** 2)

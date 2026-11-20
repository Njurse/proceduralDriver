"""Streams the terrain as square chunk meshes round the car.

``chunk_mesh`` is pure numpy (tested without Panda); ``ChunkStreamer`` turns
meshes into Panda3D GeomNodes, building the nearest missing chunks first and
dropping chunks that fall out of range.  A ground with a ``road_mesh(ci, ck)``
(world.world.World) gets its road strips added to each chunk.  Chunk (i, k) covers physics x in
[i * size, (i + 1) * size] and z in [k * size, (k + 1) * size].
"""

import math
from dataclasses import dataclass

import numpy as np
from panda3d.core import (
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexArrayFormat,
    GeomVertexData,
    GeomVertexFormat,
    InternalName,
    PNMImage,
)

import config
from core.noise import fbm_2d
from core.textures import _to_texture

DETAIL_TILE_M = 8.0      # metres per repeat of the detail texture

# Vertex colours by height (fraction of amplitude) and steepness.
_LOW = np.array([0.14, 0.28, 0.10])    # valley floor
_MID = np.array([0.32, 0.47, 0.19])    # grass
_HIGH = np.array([0.66, 0.60, 0.33])   # dry hilltops
_STEEP = np.array([0.48, 0.38, 0.25])  # bare earth on the steeper slopes


@dataclass
class ChunkMesh:
    """Vertices in physics coordinates (x, y up, z)."""

    pos: np.ndarray      # (n, 3)
    normal: np.ndarray   # (n, 3)
    color: np.ndarray    # (n, 4)
    uv: np.ndarray       # (n, 2)
    tris: np.ndarray     # (m, 3) vertex indices


def terrain_colors(height, normal, amplitude, ramp=None, steep_deg=11.0):
    """RGBA per vertex from height (m above the base level) and unit normals
    (..., 3).  ``ramp`` has low/mid/high/steep RGB (world.biomes.ColorRamp);
    slopes blend to ``steep`` over steep_deg +- 2.2 degrees."""
    low, mid, high, steep_rgb = ((_LOW, _MID, _HIGH, _STEEP) if ramp is None else
                                 (np.asarray(c) for c in (ramp.low, ramp.mid, ramp.high,
                                                          ramp.steep)))
    t = np.clip(np.asarray(height) / (0.5 * max(amplitude, 1e-6)), -1.0, 1.0)[..., None]
    lo = mid + (low - mid) * np.clip(-t, 0.0, 1.0)
    rgb = lo + (high - mid) * np.clip(t, 0.0, 1.0)
    c0, c1 = (math.cos(math.radians(steep_deg + d)) for d in (-2.2, 2.2))
    steep = np.clip((c0 - normal[..., 1:2]) / (c0 - c1), 0.0, 1.0)
    rgb = rgb + (steep_rgb - rgb) * steep
    alpha = np.ones(rgb.shape[:-1] + (1,))
    return np.concatenate([rgb, alpha], axis=-1)


def chunk_mesh(terrain, ci, ck, size=None, res=None):
    """Grid mesh of chunk (ci, ck): (res + 1)^2 vertices, 2 res^2 triangles.

    Vertex x/z are computed as (chunk * res + index) * step so neighbouring
    chunks produce bit-identical edge vertices.
    """
    size = float(config.TERRAIN_CHUNK_SIZE if size is None else size)
    res = int(config.TERRAIN_CHUNK_RES if res is None else res)
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


def chunks_around(x, z, radius, size=None):
    """Chunk keys within ``radius`` chunks of the one under (x, z), nearest first."""
    size = float(config.TERRAIN_CHUNK_SIZE if size is None else size)
    ci, ck = math.floor(x / size), math.floor(z / size)
    keys = [(ci + di, ck + dk) for di in range(-radius, radius + 1)
            for dk in range(-radius, radius + 1)]
    return sorted(keys, key=lambda k: (k[0] - ci) ** 2 + (k[1] - ck) ** 2)


def _noise_tile(size, seed, frequency, lo, hi):
    """Tileable fBm grey texture with its range mapped to [lo, hi] (uint8)."""
    u = np.linspace(0.0, 1.0, size, endpoint=False)
    gu, gv = np.meshgrid(u, u)

    def f(du, dv):
        return (fbm_2d((gu - du) * frequency, (gv - dv) * frequency, seed, 4)
                + 0.5 * fbm_2d((gu - du) * 4.0 * frequency,
                               (gv - dv) * 4.0 * frequency, seed + 1, 3))

    # blend four shifted copies so the noise wraps in both u and v
    tile = ((f(0, 0) * (1 - gu) + f(1, 0) * gu) * (1 - gv)
            + (f(0, 1) * (1 - gu) + f(1, 1) * gu) * gv)
    tile = (tile - tile.min()) / max(np.ptp(tile), 1e-9)
    grey = (lo + (hi - lo) * tile).astype(np.uint8)
    return np.stack([grey, grey, grey], axis=-1)


def detail_texture(path=None, size=256, seed=7):
    """Tileable grey speckle (0.72-1.0) that modulates the vertex colours:
    the grey levels of the image at ``path`` stretched into that range, or
    procedural noise when no path is given."""
    if path is not None:
        img = PNMImage(Filename.fromOsSpecific(str(path)))
        w, h = img.getXSize(), img.getYSize()
        grey = np.array([[img.getBright(x, y) for x in range(w)] for y in range(h)])
        lo, hi = np.percentile(grey, [1.0, 99.0])
        tile = np.clip((grey - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        grey = (184.0 + 71.0 * tile).astype(np.uint8)
        return _to_texture(np.stack([grey, grey, grey], axis=-1), "terrain_detail",
                           repeat=True)
    return _to_texture(_noise_tile(size, seed, 16.0, 184.0, 255.0),
                       "terrain_detail", repeat=True)


def road_detail_texture(size=128, seed=17):
    """Fine, high-contrast grey speckle (0.45-1.0) for the road strips: twice
    the noise frequency of the terrain texture, so the grain stays small when
    the roads tile it every ``config.ROAD_DETAIL_TILE_M`` metres."""
    return _to_texture(_noise_tile(size, seed, 32.0, 115.0, 255.0),
                       "road_detail", repeat=True)


_FORMAT = None


def _vertex_format():
    global _FORMAT
    if _FORMAT is None:
        arr = GeomVertexArrayFormat()
        arr.addColumn(InternalName.getVertex(), 3, Geom.NT_float32, Geom.C_point)
        arr.addColumn(InternalName.getNormal(), 3, Geom.NT_float32, Geom.C_normal)
        arr.addColumn(InternalName.getColor(), 4, Geom.NT_float32, Geom.C_color)
        arr.addColumn(InternalName.getTexcoord(), 2, Geom.NT_float32, Geom.C_texcoord)
        _FORMAT = GeomVertexFormat.registerFormat(GeomVertexFormat(arr))
    return _FORMAT


def mesh_node(mesh, name):
    """Panda3D GeomNode for a ChunkMesh (physics (x, y, z) -> Panda (x, z, y))."""
    swap = [0, 2, 1]
    rows = np.concatenate([mesh.pos[:, swap], mesh.normal[:, swap], mesh.color, mesh.uv],
                          axis=1).astype(np.float32)
    vdata = GeomVertexData(name, _vertex_format(), Geom.UH_static)
    vdata.uncleanSetNumRows(len(rows))
    memoryview(vdata.modifyArray(0)).cast("B")[:] = rows.tobytes()

    prim = GeomTriangles(Geom.UH_static)
    prim.setIndexType(Geom.NT_uint32)
    handle = prim.modifyVertices()
    handle.uncleanSetNumRows(mesh.tris.size)
    memoryview(handle).cast("B")[:] = mesh.tris.astype(np.uint32).tobytes()

    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode(name)
    node.addGeom(geom)
    return node


class ChunkStreamer:
    """Keeps the chunks round a moving point loaded under ``parent``."""

    def __init__(self, terrain, parent, texture=None, road_texture=None, radius=None,
                 builds_per_frame=None):
        self.terrain = terrain
        self.road_texture = road_texture
        self.root = parent.attachNewNode("terrain")
        if texture is not None:
            self.root.setTexture(texture, 1)
        self.radius = int(config.TERRAIN_VIEW_CHUNKS if radius is None else radius)
        self.budget = int(config.TERRAIN_BUILDS_PER_FRAME if builds_per_frame is None
                          else builds_per_frame)
        self.loaded = {}

    def update(self, x, z, budget=None):
        """Build up to ``budget`` (default TERRAIN_BUILDS_PER_FRAME) missing
        chunks, nearest first, and drop chunks more than radius + 1 away.
        Returns the number built."""
        budget = self.budget if budget is None else budget
        wanted = chunks_around(x, z, self.radius)
        built = 0
        for key in wanted:
            if key in self.loaded:
                continue
            if built >= budget:
                break
            name = f"chunk_{key[0]}_{key[1]}"
            chunk = self.root.attachNewNode(mesh_node(chunk_mesh(self.terrain, *key), name))
            roads = (self.terrain.road_mesh(*key) if hasattr(self.terrain, "road_mesh")
                     else None)
            if roads is not None:
                road_node = chunk.attachNewNode(mesh_node(roads, name + "_roads"))
                if self.road_texture is not None:
                    # roads tile their own finer, higher-contrast noise
                    road_node.setTexture(self.road_texture, 1)
                # just above the flattened ground, and drawn over it where
                # the depth buffer can't tell them apart
                road_node.setDepthOffset(1)
            self.loaded[key] = chunk
            built += 1
        size = config.TERRAIN_CHUNK_SIZE
        ci, ck = math.floor(x / size), math.floor(z / size)
        for key in [k for k in self.loaded
                    if max(abs(k[0] - ci), abs(k[1] - ck)) > self.radius + 1]:
            self.loaded.pop(key).removeNode()
        return built

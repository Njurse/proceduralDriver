"""The world in Panda3D: streams worldgen chunk meshes round the car.

``ChunkStreamer`` turns worldgen.chunk meshes into GeomNodes, building the
nearest missing chunks first and dropping chunks that fall out of range.  A
ground with a ``road_mesh(ci, ck)`` (world.world.World) gets its road strips
added to each chunk.  The detail textures that modulate the vertex colours
live here too.
"""

import logging
import math
import time

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
from game.structure_view import style_structures
from worldgen.chunk import chunk_mesh, chunks_around

log = logging.getLogger(__name__)


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
                 builds_per_frame=None, build_ms=None, spec=None):
        self.terrain = terrain
        self.spec = terrain.spec if spec is None else spec   # chunk size and res
        self.road_texture = road_texture
        self.root = parent.attachNewNode("terrain")
        if texture is not None:
            self.root.setTexture(texture, 1)
        self.radius = int(config.TERRAIN_VIEW_CHUNKS if radius is None else radius)
        self.budget = int(config.TERRAIN_BUILDS_PER_FRAME if builds_per_frame is None
                          else builds_per_frame)
        self.build_ms = float(config.TERRAIN_BUILD_MS if build_ms is None else build_ms)
        self.loaded = {}
        self._roadless = set()      # loaded, road strips still to add (a step of their own)
        self._bare = set()          # loaded, structures still to add (another step)
        self._batch = [0, 0.0, 0]   # built, ms, dropped since the last summary

    def reset(self, terrain):
        """Drop every chunk and stream ``terrain`` from now on (new seed)."""
        for node in self.loaded.values():
            node.removeNode()
        self.loaded.clear()
        self._roadless.clear()
        self._bare.clear()
        self.terrain = terrain
        self.spec = terrain.spec

    def update(self, x, z, budget=None):
        """Build up to ``budget`` (default TERRAIN_BUILDS_PER_FRAME) missing
        chunks, nearest first, and drop chunks more than radius + 1 away.
        Per frame (no ``budget``) a chunk's ground, its road strips and its
        structures are separate steps, and building stops once ``build_ms`` is spent, so
        streaming costs about one step a frame.  Returns the steps taken."""
        per_frame = budget is None
        stop = time.perf_counter() + self.build_ms / 1000.0 if per_frame else math.inf
        budget = self.budget if budget is None else budget
        roads = getattr(self.terrain, "roads", None)
        if roads is not None:
            roads.prefetch(x, z, (self.radius + 1) * self.spec.chunk_size_m)
        has_roads = hasattr(self.terrain, "road_mesh")
        has_structures = hasattr(self.terrain, "structure_mesh")
        built, left = 0, False
        for key in chunks_around(x, z, self.radius, self.spec.chunk_size_m):
            fresh = key not in self.loaded
            if not fresh and key not in self._roadless and key not in self._bare:
                continue
            if built >= budget or time.perf_counter() >= stop:
                left = True
                break
            t0 = time.perf_counter()
            if fresh:
                name = f"chunk_{key[0]}_{key[1]}"
                self.loaded[key] = self.root.attachNewNode(
                    mesh_node(chunk_mesh(self.terrain, *key, self.spec), name))
                self._batch[0] += 1
                if has_roads:
                    self._roadless.add(key)
                if has_structures:
                    self._bare.add(key)
            more = not (fresh and per_frame)   # per frame: one step a chunk a frame
            if more and key in self._roadless:
                self._roadless.discard(key)
                self._add_roads(key)
                more = not per_frame
            if more and key in self._bare:
                self._bare.discard(key)
                self._add_structures(key)
            built += 1
            self._batch[1] += (time.perf_counter() - t0) * 1000.0
        size = self.spec.chunk_size_m
        ci, ck = math.floor(x / size), math.floor(z / size)
        for key in [k for k in self.loaded
                    if max(abs(k[0] - ci), abs(k[1] - ck)) > self.radius + 1]:
            self.loaded.pop(key).removeNode()
            self._roadless.discard(key)
            self._bare.discard(key)
            self._batch[2] += 1
        b = self._batch
        if not left and (b[0] or b[2]):   # nothing left to build: one line per batch
            log.info("chunks: +%d in %.0f ms (%.1f ms each), -%d, %d loaded",
                     b[0], b[1], b[1] / max(b[0], 1), b[2], len(self.loaded))
            self._batch = [0, 0.0, 0]
        return built

    def _add_structures(self, key):
        mesh = self.terrain.structure_mesh(*key)
        if mesh is not None:
            name = f"chunk_{key[0]}_{key[1]}_structures"
            style_structures(self.loaded[key].attachNewNode(mesh_node(mesh, name)))

    def _add_roads(self, key):
        roads = self.terrain.road_mesh(*key)
        if roads is None:
            return
        name = f"chunk_{key[0]}_{key[1]}_roads"
        road_node = self.loaded[key].attachNewNode(mesh_node(roads, name))
        if self.road_texture is not None:
            # roads tile their own finer, higher-contrast noise
            road_node.setTexture(self.road_texture, 1)
        # just above the flattened ground, and drawn over it where the depth
        # buffer can't tell them apart
        road_node.setDepthOffset(1)

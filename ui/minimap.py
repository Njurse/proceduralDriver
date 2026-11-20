"""HUD minimap, top right: the chunk grid round the car, north up.

Loaded (drawn) chunks are bright, discovered ones dim, the rest dark; each
chunk shows its biome colours, hillshade and roads (world.worldmap).  The
map image is a small texture redrawn every MINIMAP_REFRESH_FRAMES frames at
most, and only when something changed; the car arrow in the middle is its
own little triangle that turns with the car.
"""

import math

import numpy as np
from panda3d.core import (
    CardMaker,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    TransparencyAttrib,
)

import config
from core.textures import _to_texture, upload_rgba
from world.worldmap import TileCache, minimap_image

BORDER_PX = 2
MARGIN_PX = 14


def _arrow(size):
    """A white arrow pointing up (+z), centred on the origin."""
    vdata = GeomVertexData("arrow", GeomVertexFormat.getV3c4(), Geom.UH_static)
    pos, col = GeomVertexWriter(vdata, "vertex"), GeomVertexWriter(vdata, "color")
    pts = [(0.0, size), (-0.6 * size, -0.7 * size), (0.0, -0.35 * size),
           (0.6 * size, -0.7 * size)]
    for x, z in pts:
        pos.addData3(x, 0.0, z)
        col.addData4(1.0, 1.0, 1.0, 1.0)
    prim = GeomTriangles(Geom.UH_static)
    prim.addVertices(0, 1, 2)
    prim.addVertices(0, 2, 3)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("minimap_arrow")
    node.addGeom(geom)
    return node


class Minimap:
    def __init__(self, base, world):
        self.base = base
        self.size = config.MINIMAP_SIZE_PX
        self.tex_px = config.MINIMAP_TEXTURE_PX
        self.zoom = 0
        self.frame = 0
        self._dirty = True
        self._last = None
        self.cache = TileCache(world, max(config.MINIMAP_ZOOMS))
        self.texture = _to_texture(np.zeros((self.tex_px, self.tex_px, 3), np.uint8),
                                   "minimap")
        self.root = base.a2dTopRight.attachNewNode("minimap")
        s, m = self.size, MARGIN_PX
        border = CardMaker("minimap_border")
        border.setFrame(-m - s - BORDER_PX, -m + BORDER_PX, -m - s - BORDER_PX, -m + BORDER_PX)
        self.root.attachNewNode(border.generate()).setColor(0.85, 0.85, 0.80, 1.0)
        card = CardMaker("minimap_card")
        card.setFrame(-m - s, -m, -m - s, -m)
        self.card = self.root.attachNewNode(card.generate())
        self.card.setTexture(self.texture)
        self.arrow = self.root.attachNewNode(_arrow(7.0))
        self.arrow.setPos(-m - s / 2.0, 0.0, -m - s / 2.0)
        self.root.setTransparency(TransparencyAttrib.MNone)
        self.root.setLightOff(1)
        self.layout()

    def layout(self):
        """One local unit = one window pixel (aspect2d is 2 units tall)."""
        h = self.base.win.getYSize() if self.base.win is not None else config.WINDOW_HEIGHT
        self.root.setScale(2.0 / max(h, 1))

    def set_world(self, world):
        self.cache = TileCache(world, max(config.MINIMAP_ZOOMS))
        self._dirty = True

    def toggle_zoom(self):
        self.zoom = (self.zoom + 1) % len(config.MINIMAP_ZOOMS)
        self._dirty = True

    def show(self, visible):
        (self.root.show if visible else self.root.hide)()

    def update(self, pos, heading, discovered, loaded, force=False):
        """Per frame: turn the arrow, sample a few chunk tiles, and redraw the
        map image when due and something changed (or ``force``)."""
        self.arrow.setR(math.degrees(heading))
        size = config.TERRAIN_CHUNK_SIZE
        ci, ck = math.floor(pos[0] / size), math.floor(pos[1] / size)
        todo = sorted((k for k in loaded if k not in self.cache.tiles),
                      key=lambda k: (k[0] - ci) ** 2 + (k[1] - ck) ** 2)
        built = self.cache.build(todo, config.MINIMAP_TILES_PER_FRAME)
        if not built and not todo:
            seen = sorted((k for k in discovered if k not in self.cache.tiles
                           and abs(k[0] - ci) < 40 and abs(k[1] - ck) < 40),
                          key=lambda k: (k[0] - ci) ** 2 + (k[1] - ck) ** 2)
            built = self.cache.build(seen, config.MINIMAP_TILES_PER_FRAME)
        self._dirty |= bool(built)
        self.frame += 1
        ppc = config.MINIMAP_ZOOMS[self.zoom]
        mpp = size / ppc
        state = (round(pos[0] / mpp), round(pos[1] / mpp), len(loaded), len(discovered))
        self._dirty |= state != self._last
        if not force and (not self._dirty or self.frame % config.MINIMAP_REFRESH_FRAMES):
            return False
        img = minimap_image(self.cache, (state[0] * mpp, state[1] * mpp), discovered,
                            loaded, ppc, self.tex_px)
        upload_rgba(self.texture, img)
        self._last, self._dirty = state, False
        return True

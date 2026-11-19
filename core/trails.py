"""Colored trails for ships, backed by one reusable dynamic geometry buffer."""

import logging

from panda3d.core import (
    Geom,
    GeomLines,
    GeomNode,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    TransparencyAttrib,
)

import config

log = logging.getLogger("galaxy_explorer.trails")


class Trail:
    """A fading line of world-space points with in-place vertex updates.

    Rebuilding a ``LineSegs`` node every frame allocated a fresh geometry and
    node each time.  Here the vertex data is allocated once and only rewritten,
    which keeps trail updates cheap for the player and every traffic ship.
    """

    def __init__(self, parent, color, max_points=None, spacing=None):
        self.parent = parent
        self.color = tuple(color)
        self.max_points = max_points or config.TRAIL_MAX_POINTS
        self.spacing = spacing or config.TRAIL_POINT_SPACING
        self.points = []
        self.last = None

        self.node = parent.attachNewNode("trail")
        self.node.setLightOff(1)
        self.node.setMaterialOff(1)
        self.node.setTransparency(TransparencyAttrib.MAlpha)
        self.node.setDepthWrite(False)
        self.node.setBin("transparent", 15)

        self.geom_node = GeomNode("trail_geom")
        self.node.attachNewNode(self.geom_node)
        vertex_format = GeomVertexFormat.getV3c4()
        self.vertex_data = GeomVertexData(
            "trail_vertices", vertex_format, Geom.UHDynamic)
        self.vertex_writer = GeomVertexWriter(self.vertex_data, "vertex")
        self.color_writer = GeomVertexWriter(self.vertex_data, "color")
        self.geom = Geom(self.vertex_data)
        self.lines = GeomLines(Geom.UHDynamic)
        self.geom.addPrimitive(self.lines)
        self.geom_node.addGeom(self.geom)
        self._rebuild()

    def update(self, position):
        if self.last is None or (position - self.last).length() >= self.spacing:
            self.points.append(position.__class__(position))
            self.last = position.__class__(position)
            if len(self.points) > self.max_points:
                self.points.pop(0)
        self._rebuild()

    def set_color(self, color):
        self.color = tuple(color)
        self._rebuild()

    def _rebuild(self):
        segments = max(0, len(self.points) - 1)
        rows = segments * 2
        self.vertex_data.setNumRows(rows)
        self.vertex_writer.setRow(0)
        self.color_writer.setRow(0)
        self.lines.clearVertices()
        row = 0
        for index in range(segments):
            alpha = 0.15 + 0.7 * (index / max(segments, 1))
            for point in (self.points[index], self.points[index + 1]):
                self.vertex_writer.addData3f(point.x, point.y, point.z)
                self.color_writer.addData4f(
                    self.color[0], self.color[1], self.color[2], alpha)
                self.lines.addVertex(row)
                row += 1
        self.geom_node.setGeom(0, self.geom)

    def destroy(self):
        try:
            self.node.removeNode()
        except Exception as exc:
            log.debug("[trail] removal failed: %s", exc)

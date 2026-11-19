"""Procedural meshes: spheres, heightfields, point clouds and glows."""

import math

import numpy as np
from panda3d.core import (
    CardMaker,
    ColorBlendAttrib,
    Geom,
    GeomNode,
    GeomPoints,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    NodePath,
    RenderModeAttrib,
    TransparencyAttrib,
)


def additive_blend(node):
    """Enable additive blending (Panda3D has no TransparencyAttrib.MAdd)."""
    node.setAttrib(ColorBlendAttrib.make(
        ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
        ColorBlendAttrib.OOne))
    return node


def make_uv_sphere(radius=1.0, rings=28, segments=40, name="sphere"):
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows((rings + 1) * (segments + 1))
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    tw = GeomVertexWriter(vdata, "texcoord")

    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        for seg in range(segments + 1):
            theta = math.tau * seg / segments
            x = sin_phi * math.cos(theta)
            y = sin_phi * math.sin(theta)
            z = cos_phi
            vw.addData3f(x * radius, y * radius, z * radius)
            nw.addData3f(x, y, z)
            tw.addData2f(seg / segments, 1.0 - ring / rings)

    tris = GeomTriangles(Geom.UHStatic)
    stride = segments + 1
    for ring in range(rings):
        for seg in range(segments):
            a = ring * stride + seg
            b = a + stride
            tris.addVertices(a, a + 1, b)
            tris.addVertices(a + 1, b + 1, b)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setTwoSided(True)
    return result


def make_heightfield(heights, world_size, colors, name="terrain"):
    """Build a terrain mesh. heights/colors are (N, N[, 4]) numpy arrays."""
    heights = np.asarray(heights, dtype=np.float64)
    grid = heights.shape[0]
    colors = np.asarray(colors, dtype=np.float64)
    if colors.shape[-1] == 3:
        colors = np.concatenate(
            [colors, np.ones(colors.shape[:2] + (1,))], axis=-1)

    spacing = world_size / max(grid - 1, 1)
    d_row, d_col = np.gradient(heights, spacing, spacing)
    nx = -d_col
    ny = -d_row
    nz = np.ones_like(heights)
    length = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx /= length
    ny /= length
    nz /= length

    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(grid * grid)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")

    half = world_size * 0.5
    for row in range(grid):
        y = row / max(grid - 1, 1) * world_size - half
        for col in range(grid):
            x = col / max(grid - 1, 1) * world_size - half
            vw.addData3f(x, y, float(heights[row, col]))
            nw.addData3f(float(nx[row, col]), float(ny[row, col]),
                         float(nz[row, col]))
            cw.addData4f(*[float(c) for c in colors[row, col]])

    tris = GeomTriangles(Geom.UHStatic)
    for row in range(grid - 1):
        for col in range(grid - 1):
            a = row * grid + col
            b = a + 1
            c = a + grid
            d = c + 1
            tris.addVertices(a, b, c)
            tris.addVertices(b, d, c)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)


def make_sky_dome(radius, top_color, horizon_color, rings=24, segments=48,
                  name="sky"):
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows((rings + 1) * (segments + 1))
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")

    top = tuple(top_color)
    horizon = tuple(horizon_color)
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        sin_phi, cos_phi = math.sin(phi), math.cos(phi)
        for seg in range(segments + 1):
            theta = math.tau * seg / segments
            x = sin_phi * math.cos(theta)
            y = sin_phi * math.sin(theta)
            z = cos_phi
            vw.addData3f(x * radius, y * radius, z * radius)
            t = max(0.0, min(1.0, (z + 0.08) / 1.08))
            cw.addData4f(
                horizon[0] + (top[0] - horizon[0]) * t,
                horizon[1] + (top[1] - horizon[1]) * t,
                horizon[2] + (top[2] - horizon[2]) * t,
                1.0,
            )

    tris = GeomTriangles(Geom.UHStatic)
    stride = segments + 1
    for ring in range(rings):
        for seg in range(segments):
            a = ring * stride + seg
            b = a + stride
            tris.addVertices(a, a + 1, b)
            tris.addVertices(a + 1, b + 1, b)
    tris.closePrimitive()

    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setTwoSided(True)
    return result


def make_points(positions, colors, size=1.6, dynamic=False, name="points"):
    positions = np.asarray(positions, dtype=np.float64)
    colors = np.asarray(colors, dtype=np.float64)
    if colors.shape[-1] == 3:
        colors = np.concatenate(
            [colors, np.ones(colors.shape[:1] + (1,))], axis=-1)

    usage = Geom.UHDynamic if dynamic else Geom.UHStatic
    fmt = GeomVertexFormat.getV3c4()
    vdata = GeomVertexData(name, fmt, usage)
    vdata.setNumRows(len(positions))
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for (x, y, z), color in zip(positions, colors):
        vw.addData3f(x, y, z)
        cw.addData4f(*[float(c) for c in color])

    points = GeomPoints(usage)
    points.addConsecutiveVertices(0, len(positions))
    points.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(points)
    node = GeomNode(name)
    node.addGeom(geom)

    result = NodePath(node)
    result.setLightOff(1)
    result.setMaterialOff(1)
    result.setAttrib(RenderModeAttrib.make(RenderModeAttrib.MPoint, size))
    return result


def make_glow_sphere(radius, color, intensity=1.0, name="glow"):
    """Soft additive halo built from concentric translucent spheres."""
    root = NodePath(name)
    layers = ((1.0, 1.00), (1.42, 0.38), (2.05, 0.13))
    for idx, (scale, weight) in enumerate(layers):
        shell = make_uv_sphere(radius * scale, 10, 16, f"{name}_layer{idx}")
        shell.setColor(
            color[0] * intensity * weight,
            color[1] * intensity * weight,
            color[2] * intensity * weight,
            1.0,
        )
        additive_blend(shell)
        shell.setLightOff(1)
        shell.setMaterialOff(1)
        shell.setDepthWrite(False)
        shell.setBin("transparent", 20 + idx)
        shell.reparentTo(root)
    return root


def make_quad(corners, color, name="quad", double_sided=True):
    """A single flat quad from four corner vertices (counter-clockwise)."""
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(4)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    for corner in corners:
        vw.addData3f(float(corner[0]), float(corner[1]), float(corner[2]))
        cw.addData4f(*[float(c) for c in color])
    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)
    tris.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setLightOff(1)
    result.setMaterialOff(1)
    if double_sided:
        result.setTwoSided(True)
    return result


def make_cone(radius, height, segments=18, name="cone"):
    """A cone with its apex at the origin opening toward +Z (a beam volume).

    Used for landing-pad guidance beams; combine with ``additive_blend`` for
    a translucent light-cone look.  Panda3D's ``panda3d.core`` exposes no
    ``Cone`` primitive, so this builds the geometry directly.
    """
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(segments + 2)
    vw = GeomVertexWriter(vdata, "vertex")
    cw = GeomVertexWriter(vdata, "color")
    vw.addData3f(0.0, 0.0, 0.0)               # apex
    cw.addData4f(1.0, 1.0, 1.0, 1.0)
    for seg in range(segments + 1):
        angle = math.tau * seg / segments
        vw.addData3f(radius * math.cos(angle), radius * math.sin(angle),
                     height)
        cw.addData4f(1.0, 1.0, 1.0, 0.0)      # fade toward the wide end
    tris = GeomTriangles(Geom.UHStatic)
    for seg in range(segments):
        tris.addVertices(0, 1 + seg, 2 + seg)
    tris.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setTwoSided(True)
    result.setLightOff(1)
    result.setMaterialOff(1)
    return result


def make_ground_plane(size, z=0.0, name="ground", repeat=1.0):
    """Horizontal X-Y quad with +Z normals and tiled UVs."""
    fmt = GeomVertexFormat.getV3n3t2()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    vdata.setNumRows(4)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    tw = GeomVertexWriter(vdata, "texcoord")
    half = size * 0.5
    corners = ((-half, -half), (half, -half), (half, half), (-half, half))
    for x, y in corners:
        vw.addData3f(x, y, z)
        nw.addData3f(0.0, 0.0, 1.0)
        tw.addData2f((x + half) / size * repeat,
                     (y + half) / size * repeat)
    tris = GeomTriangles(Geom.UHStatic)
    tris.addVertices(0, 1, 2)
    tris.addVertices(0, 2, 3)
    tris.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    result = NodePath(node)
    result.setTwoSided(True)
    return result


def make_plane(size, color, name="plane", alpha=1.0):
    cm = CardMaker(name)
    half = size * 0.5
    cm.setFrame(-half, half, -half, half)
    plane = NodePath(cm.generate())
    plane.setColor(color[0], color[1], color[2], alpha)
    if alpha < 1.0:
        plane.setTransparency(TransparencyAttrib.MAlpha)
    plane.setLightOff(1)
    plane.setMaterialOff(1)
    return plane


def make_textured_plane(texture, size, color=(1, 1, 1, 1), name="textured"):
    cm = CardMaker(name)
    half = size * 0.5
    cm.setFrame(-half, half, -half, half)
    plane = NodePath(cm.generate())
    plane.setTexture(texture)
    plane.setColor(*color)
    if color[3] < 1.0:
        plane.setTransparency(TransparencyAttrib.MAlpha)
    plane.setLightOff(1)
    plane.setMaterialOff(1)
    return plane


def make_box(sx, sy, sz, name="box"):
    """A solid box centered at the origin, with per-face normals."""
    hx, hy, hz = sx * 0.5, sy * 0.5, sz * 0.5
    fmt = GeomVertexFormat.getV3n3c4()
    vdata = GeomVertexData(name, fmt, Geom.UHStatic)
    # 6 faces * 4 vertices
    vdata.setNumRows(24)
    vw = GeomVertexWriter(vdata, "vertex")
    nw = GeomVertexWriter(vdata, "normal")
    cw = GeomVertexWriter(vdata, "color")
    faces = [
        # (normal, four corners)
        ((0, 0, 1), [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)]),
        ((0, 0, -1), [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)]),
        ((0, 1, 0), [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)]),
        ((0, -1, 0), [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]),
        ((1, 0, 0), [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)]),
        ((-1, 0, 0), [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)]),
    ]
    tris = GeomTriangles(Geom.UHStatic)
    base = 0
    for normal, corners in faces:
        for (x, y, z) in corners:
            vw.addData3f(x, y, z)
            nw.addData3f(*normal)
            cw.addData4f(0.7, 0.75, 0.85, 1.0)
        tris.addVertices(base, base + 1, base + 2)
        tris.addVertices(base, base + 2, base + 3)
        base += 4
    tris.closePrimitive()
    geom = Geom(vdata)
    geom.addPrimitive(tris)
    node = GeomNode(name)
    node.addGeom(geom)
    return NodePath(node)

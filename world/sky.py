"""Sky dome: the skybox panorama on an inward-facing hemisphere.

assets/world/landscape/skybox.png is a single panorama strip (652x306):
sky above, a thin tree line along the bottom rows, no ground.  Its columns
run round the horizon and its rows from the horizon (bottom) upwards, so
u follows the azimuth and v the elevation.  Its left and right edges match
(mean edge difference 2.5/255, about twice the column-to-column change), so
it wraps once round the circle.
The top rows stretch over the upper dome (the panorama only reaches ~40 deg).
"""

import math

import numpy as np
from panda3d.core import (
    CompassEffect,
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    PNMImage,
    Texture,
)

from core.textures import use_nearest

SEGMENTS = 48            # around the horizon
RINGS = 16               # horizon to zenith
BELOW_HORIZON_DEG = -6.0  # dome skirt, hidden by the ground
TOP_DEG = 70.0           # elevation where the image's top row lands
HORIZON_ROW = 0.035      # fraction of the image height below the tree line tops


def _v(elevation_deg):
    """Texture v (0 = image bottom) for an elevation."""
    t = (elevation_deg - BELOW_HORIZON_DEG) / (TOP_DEG - BELOW_HORIZON_DEG)
    return min(1.0, HORIZON_ROW + (1.0 - HORIZON_ROW) * max(t, 0.0))


def dome_node(radius):
    """Hemisphere of ``radius`` metres, faces wound to be seen from inside."""
    vdata = GeomVertexData("sky", GeomVertexFormat.getV3t2(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    uv = GeomVertexWriter(vdata, "texcoord")
    for r in range(RINGS + 1):
        e = BELOW_HORIZON_DEG + (90.0 - BELOW_HORIZON_DEG) * r / RINGS
        ce, se = math.cos(math.radians(e)), math.sin(math.radians(e))
        for s in range(SEGMENTS + 1):
            a = 2.0 * math.pi * s / SEGMENTS
            vertex.addData3(radius * ce * math.sin(a), radius * ce * math.cos(a), radius * se)
            uv.addData2(s / SEGMENTS, _v(e))
    prim = GeomTriangles(Geom.UH_static)
    row = SEGMENTS + 1
    for r in range(RINGS):
        for s in range(SEGMENTS):
            q = r * row + s
            # counter-clockwise seen from the centre
            prim.addVertices(q, q + 1, q + row)
            prim.addVertices(q + 1, q + row + 1, q + row)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("sky_dome")
    node.addGeom(geom)
    return node


def horizon_colour(img):
    """Mean colour of the hazy band just above the tree line (for the fog)."""
    h = img.getYSize()
    rows = range(int(h * 0.80), int(h * 0.90))
    cols = range(0, img.getXSize(), 4)
    rgb = np.array([[img.getXel(x, y) for x in cols] for y in rows])
    return tuple(float(c) for c in rgb.reshape(-1, 3).mean(axis=0))


def build_sky(png_path, camera, render, radius):
    """Attach the dome to ``camera`` (position only); returns (dome, horizon rgb)."""
    img = PNMImage(Filename.fromOsSpecific(str(png_path)))
    tex = Texture("skybox")
    tex.load(img)
    tex.setWrapU(Texture.WM_repeat)
    tex.setWrapV(Texture.WM_clamp)
    use_nearest(tex)
    dome = camera.attachNewNode(dome_node(radius))
    dome.setEffect(CompassEffect.make(render, CompassEffect.P_rot))  # no camera turn
    dome.setTexture(tex, 1)
    dome.setLightOff(1)
    dome.setFogOff(1)
    dome.setBin("background", 0)
    dome.setDepthWrite(False)
    dome.setDepthTest(False)
    return dome, horizon_colour(img)

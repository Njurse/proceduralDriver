"""The car on screen: the OBJ body, the Driver 2 style wheels, and the
physics-to-Panda3D pose sync (a DriveSim mixin)."""

import logging
import math
import os

import numpy as np
from panda3d.core import (
    CardMaker,
    ColorAttrib,
    CullFaceAttrib,
    Filename,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Mat4,
    MaterialAttrib,
    NodePath,
    TextureAttrib,
    TransparencyAttrib,
    getModelPath,
)

import config
from core.textures import _to_texture, load_color_keyed, seamless_fbm, use_nearest
from core.units import DR2_UNITS_PER_METRE

log = logging.getLogger(__name__)


MAX_CAR_EXTENT_M = 20.0      # a longer model is in Driver 2 units, not metres
TYRE_WIDTH_M = 0.14          # gap between a wheel's outer and inner planes
TYRE_BAND_RADIUS = 0.94      # tread diamond corners, fraction of the wheel radius

# Physics frame (x, y up, z forward) -> Panda3D (x, y forward, z up).  This
# swap is a reflection; the C physics frame is itself mirrored relative to
# the screen, so steering right turns right on screen.
PHYS_TO_PANDA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def _mat4(a, t=(0.0, 0.0, 0.0)):
    """Column-vector 3x3 transform + translation -> Panda's row-vector Mat4."""
    return Mat4(a[0, 0], a[1, 0], a[2, 0], 0.0,
                a[0, 1], a[1, 1], a[2, 1], 0.0,
                a[0, 2], a[1, 2], a[2, 2], 0.0,
                t[0], t[1], t[2], 1.0)


def _flat_colours_from_materials(model):
    """Give untextured faces (wheel wells, the "flat" material) their material
    diffuse as a flat colour; unlit, they would otherwise draw white."""
    for path in model.findAllMatches("**/+GeomNode"):
        node = path.node()
        for i in range(node.getNumGeoms()):
            state = node.getGeomState(i)
            if state.hasAttrib(TextureAttrib) or not state.hasAttrib(MaterialAttrib):
                continue
            material = state.getAttrib(MaterialAttrib).getMaterial()
            if material is None or not material.hasDiffuse():
                continue
            colour = material.getDiffuse()
            node.setGeomState(i, state.setAttrib(ColorAttrib.makeFlat(colour)))


def _model_unit_scale(model, name):
    """Metres per model unit: 1, or 1/150 for an OBJ still in Driver 2 units.

    A car exported straight from Driver 2 units (or re-exported from Blender
    without the /150) is ~750 m long: the chase camera then sits inside it
    and every face is culled, so the car looks invisible.
    """
    bounds = model.getTightBounds()
    if not bounds:
        return 1.0
    extent = max(bounds[1] - bounds[0])
    if extent <= MAX_CAR_EXTENT_M:
        return 1.0
    log.warning("[car] %s is %.0f units long: treating it as Driver 2 units (/%d)",
                name, extent, DR2_UNITS_PER_METRE)
    return 1.0 / DR2_UNITS_PER_METRE


def _tyre_band(radius, width):
    """Black tread: a diamond (square on its corner) of four quads round the
    axle (x), from the outer plane at x = 0 to the inner one at x = width."""
    vdata = GeomVertexData("tyre", GeomVertexFormat.getV3n3(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    corners = [(radius * math.cos(a), radius * math.sin(a))
               for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
    prim = GeomTriangles(Geom.UH_static)
    for k in range(4):
        (y0, z0), (y1, z1) = corners[k], corners[(k + 1) % 4]
        ny, nz = (y0 + y1) / 2.0, (z0 + z1) / 2.0
        length = math.hypot(ny, nz)
        for x, y, z in ((0.0, y0, z0), (0.0, y1, z1), (width, y1, z1), (width, y0, z0)):
            vertex.addData3(x, y, z)
            normal.addData3(0.0, ny / length, nz / length)
        prim.addVertices(4 * k, 4 * k + 1, 4 * k + 2)
        prim.addVertices(4 * k, 4 * k + 2, 4 * k + 3)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("tyre_band")
    node.addGeom(geom)
    return node


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class CarView:
    """The car model, its wheels, and keeping them on the physics pose."""

    def _build_car(self):
        import os
        obj_path = getattr(config, "HAVANA_MODEL_OBJ", None)
        # Make the OBJ's own directory visible to Panda3D's model path so its
        # relative texture references (textures/PAGE_36.png) resolve.
        try:
            getModelPath().appendDirectory(Filename.fromOsSpecific(os.path.dirname(obj_path)))
        except Exception as exc:
            log.debug("[car] model path append failed: %s", exc)
        rel = None
        try:
            rel = os.path.relpath(obj_path, config.ROOT)
        except Exception:
            rel = obj_path
        try:
            # noCache: the OBJs were rescaled in place, so a model-cache entry
            # from the old Driver 2 unit file must not be reused.
            self.car = self.loader.loadModel(Filename.fromOsSpecific(obj_path),
                                             noCache=True)
        except Exception as exc:
            log.warning("[car] load failed: %s", exc)
            self.car = NodePath("fallback_car")
        if self.car.isEmpty():
            self.car = NodePath("fallback_car")

        # The car body node carries the physics pose (see _sync_car); its local
        # frame is the C physics body frame in metres.  The OBJ is in metres
        # too, in Driver 2 model space (y down, nose at -z), which is the
        # physics body frame negated (cars.c UpdateCarDrawMatrix), so a -1
        # scale maps it in.  Both origins are the car's centre of gravity.
        self.car_body = self.render.attachNewNode("car_body")
        # Unlit like the PlayStation: the car shows its texture colours.  Lit,
        # an OBJ material with Ka 1 (Blender's default) summed ambient + sun
        # past 1.0 and washed the paint out to white.
        self.car_body.setLightOff(1)
        _flat_colours_from_materials(self.car)
        self.car.setScale(-_model_unit_scale(self.car, rel))
        self.car.setPos(*self.vehicle.geo.model_offset)
        # The OBJ faces are wound for Driver 2's y-down screen space, so cull
        # the other side.
        self.car.setAttrib(CullFaceAttrib.makeReverse())
        self.car.reparentTo(self.car_body)

        # Verify the OBJ's texture references resolved.  Panda leaves the mesh
        # untextured when the relative paths are missing, so fall back to a
        # procedural grey instead of rendering a blank/pink car.
        self._verify_model_textures(obj_path)
        car_textures = self.car.findAllTextures()
        for tex in car_textures:
            use_nearest(tex)
        tex_count = len(car_textures)
        log.info("[car] model=%s textures=%d", rel, tex_count)
        if tex_count == 0:
            log.warning("[car] no textures resolved for %s; using procedural grey",
                        rel)
            self.car.setTexture(self._grey_texture(), 1)

        # Wheels, placed every frame where cars.c DrawCarWheels draws them: an
        # outer and an inner textured plane facing the same way (the texture's
        # white background keyed out), joined by a black diamond tread band.
        self.wheel_nodes = []
        wheel_tex = None
        wheel_tex_path = self.cos.wheel_textures.get("clean")
        if wheel_tex_path and os.path.exists(wheel_tex_path):
            wheel_tex = load_color_keyed(wheel_tex_path, "wheel_clean")
        else:
            log.warning("[car] wheel texture missing: %s", wheel_tex_path)
        radius = self.vehicle.geo.wheel_radius
        # card local x/z span the plane, its normal (y) lies along the axle (x)
        disc_mat = np.array([[0.0, 1.0, 0.0],
                             [2.0 * radius, 0.0, 0.0],
                             [0.0, 0.0, 2.0 * radius]])
        self.wheel_mounts = []
        for i in range(4):
            mount = self.car_body.attachNewNode("wheel")
            spinner = mount.attachNewNode("spinner")
            inboard = -math.copysign(TYRE_WIDTH_M, self.vehicle.wheel_visual_offset(i)[0])
            for name, x in (("outer", 0.0), ("inner", inboard)):
                cm = CardMaker(f"wheel_{name}")
                cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                plane = spinner.attachNewNode(cm.generate())
                plane.setMat(_mat4(disc_mat, (x, 0.0, 0.0)))
                if wheel_tex is not None:
                    plane.setTexture(wheel_tex, 1)
                plane.setTwoSided(True)
                plane.setTransparency(TransparencyAttrib.MBinary)
            band = spinner.attachNewNode(_tyre_band(radius * TYRE_BAND_RADIUS, inboard))
            band.setTwoSided(True)
            band.setTextureOff(1)
            band.setColor(0.04, 0.04, 0.04, 1.0, 1)
            self.wheel_mounts.append(mount)
            self.wheel_nodes.append(spinner)
        self.wheel_spins = [0.0] * 4

    def _verify_model_textures(self, obj_path):
        """Log a warning for every texture referenced by the MTL that is missing."""
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        try:
            with open(mtl_path) as handle:
                lines = handle.readlines()
        except Exception as exc:
            log.debug("[car] mtl check failed for %s: %s", mtl_path, exc)
            return
        base = os.path.dirname(obj_path)
        missing = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "map_Kd":
                tex = os.path.normpath(os.path.join(base, parts[1]))
                if not os.path.exists(tex):
                    missing.append(tex)
        if missing:
            log.warning("[car] missing model textures: %s", missing)

    def _grey_texture(self, size=64):
        """Neutral procedural texture used when the model's textures fail."""
        base = np.full((size, size), 128.0)
        speckle = seamless_fbm(size, size, 12.0, 999, 3) * 24.0 - 12.0
        grey = np.clip(base + speckle, 70, 190).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)
        rgba = np.concatenate(
            [rgb, np.full((size, size, 1), 255, np.uint8)], axis=-1)
        return _to_texture(rgba, "car_grey", repeat=True)

    def _sync_car(self, dt):
        v = self.vehicle
        self.car_body.setMat(_mat4(PHYS_TO_PANDA @ v.rot, PHYS_TO_PANDA @ v.pos))

        for i, (mount, spinner) in enumerate(zip(self.wheel_mounts, self.wheel_nodes,
                                                 strict=True)):
            steer = v.wheel_angle if i in (0, 2) else 0.0
            mount.setMat(_mat4(_rot_y(steer), v.wheel_visual_offset(i)))
            self.wheel_spins[i] = (self.wheel_spins[i] + v.wheels[i].wheel_speed * dt) % math.tau
            spinner.setMat(_mat4(_rot_x(self.wheel_spins[i])))

"""The player's .glb spaceship: loading, orientation, materials and thrusters."""

import logging
import math

from panda3d.core import (
    AmbientLight,
    GeomNode,
    MaterialAttrib,
    NodePath,
    PointLight,
    Vec3,
    Vec4,
)

import config
from core.geometry import additive_blend, make_uv_sphere
from core.ship_profiles import SHIP_PROFILES

log = logging.getLogger("galaxy_explorer.ship")

_MATERIAL_STYLES = {
    "Body": ((0.58, 0.65, 0.74, 1.0), (0.16, 0.19, 0.24, 1.0),
             (0.90, 0.95, 1.00, 1.0), 48.0),
    "Engine": ((0.14, 0.15, 0.17, 1.0), (0.04, 0.04, 0.05, 1.0),
               (1.00, 0.62, 0.24, 1.0), 24.0),
    "Detail": ((0.94, 0.55, 0.16, 1.0), (0.26, 0.11, 0.02, 1.0),
               (1.00, 0.96, 0.82, 1.0), 80.0),
}


class Spaceship:
    """Visual ship. Flight state lives in :mod:`core.flight`."""

    def __init__(self, base, profile=None):
        self.base = base
        self.root = NodePath("ship_root")
        self.body = NodePath("ship_body")
        self.body.reparentTo(self.root)
        self.profile = profile or SHIP_PROFILES[0]
        self.trail_color = tuple(self.profile["trail"])
        self.jet_origin = None
        self._time = 0.0
        self._throttle = 0.0
        self._boost = False
        self._load_model()
        self._build_lights()
        self._build_thrusters()

    def _build_lights(self):
        """Ship-local fill light so the hull reads in every environment."""
        ambient = AmbientLight("ship_ambient")
        ambient.setColor(Vec4(0.28, 0.31, 0.38, 1.0))
        self.root.setLight(self.root.attachNewNode(ambient))

        fill = PointLight("ship_fill")
        fill.setColor(Vec4(0.62, 0.68, 0.82, 1.0))
        fill.setAttenuation(Vec3(1.0, 0.0, 0.0))
        fill_node = self.root.attachNewNode(fill)
        fill_node.setPos(0.0, -3.2, 2.4)
        self.root.setLight(fill_node)

    # -- construction -------------------------------------------------------
    def _load_model(self):
        # Clear previous model children.
        for child in list(self.body.getChildren()):
            try:
                child.removeNode()
            except Exception as exc:
                log.debug("[ship] old model cleanup failed: %s", exc)
        self.jet_origin = None

        obj_path = self.profile.get("obj")
        glb = self.profile.get("glb", config.SHIP_GLB)
        scale = float(self.profile.get("scale", config.SHIP_SCALE))
        invert_vertical = bool(self.profile.get("invert_vertical",
                                                getattr(config, "CAR_MODEL_INVERT_VERTICAL", False)))
        model = None
        if obj_path:
            try:
                import os
                # Panda3D resolves relative paths against its model path (ROOT);
                # absolute Windows paths get mangled by assimp, so convert.
                load_path = obj_path
                try:
                    load_path = os.path.relpath(obj_path, config.ROOT)
                except ValueError:
                    pass
                model = self.base.loader.loadModel(load_path)
                if model is None or model.isEmpty():
                    model = None
            except Exception as exc:
                log.warning("[ship] could not load obj %s: %s", obj_path, exc)
                model = None
        else:
            try:
                import gltf
                model = NodePath(gltf.load_model(glb))
            except Exception as exc:
                log.warning("[ship] could not load %s: %s", glb, exc)
                model = None
        if model is None or (hasattr(model, "isEmpty") and model.isEmpty()) or not model.getChildren():
            log.warning("[ship] using fallback model for %s",
                        self.profile.get("id", "unknown"))
            model = self._fallback_model()
            obj_path = None

        # Track whether Y should be flipped for car models (authored with
        # inverted vertical coordinates).  Applied in the centering block below.
        flip_y = bool(obj_path is not None and invert_vertical)

        # Find jet_origin node if present.
        try:
            matches = model.findAllMatches("**/*jet_origin*")
            if matches.getNumPaths() > 0:
                self.jet_origin = matches.getPath(0)
        except Exception as exc:
            log.debug("[ship] jet_origin lookup failed: %s", exc)
            self.jet_origin = None

        try:
            model.setH(180.0)
            if flip_y:
                model.setScale(scale, -scale, scale)
            else:
                model.setScale(scale)
            low, high = model.getTightBounds()
            model.setPos(-(low + high) * 0.5)
        except Exception as exc:
            log.debug("[ship] model centering failed: %s", exc)
            model.setH(180.0)
            if flip_y:
                model.setScale(scale, -scale, scale)
            else:
                model.setScale(scale)
        model.reparentTo(self.body)
        self._style_materials(model)
        self.trail_color = tuple(self.profile["trail"])

    def set_profile(self, profile):
        """Swap the visual model to a new ship profile."""
        if isinstance(profile, str):
            from core.ship_profiles import get_profile
            profile = get_profile(profile)
        self.profile = profile
        self.trail_color = tuple(profile["trail"])
        self._load_model()
        # Rebuild thrusters (reuses existing glow/flame nodes removed first).
        try:
            if hasattr(self, "glow") and self.glow is not None:
                self.glow.removeNode()
        except Exception as exc:
            log.debug("[ship] old glow cleanup failed: %s", exc)
        try:
            if hasattr(self, "flame") and self.flame is not None:
                self.flame.removeNode()
        except Exception as exc:
            log.debug("[ship] old flame cleanup failed: %s", exc)
        self._build_thrusters()

    def _fallback_model(self):
        root = NodePath("fallback_ship")
        hull = make_uv_sphere(1.0, 14, 20, "fallback_hull")
        hull.setScale(0.34, 1.75, 0.30)
        hull.setColor(0.58, 0.65, 0.74, 1.0)
        hull.reparentTo(root)
        wing = make_uv_sphere(1.0, 8, 12, "fallback_wing")
        wing.setScale(1.35, 0.34, 0.07)
        wing.setColor(0.42, 0.48, 0.56, 1.0)
        wing.reparentTo(root)
        canopy = make_uv_sphere(1.0, 10, 14, "fallback_canopy")
        canopy.setScale(0.26, 0.55, 0.22)
        canopy.setPos(0.0, 0.35, 0.20)
        canopy.setColor(0.10, 0.24, 0.38, 1.0)
        canopy.reparentTo(root)
        return root

    def _style_materials(self, model):
        for node_path in model.findAllMatches("**"):
            node = node_path.node()
            if not isinstance(node, GeomNode):
                continue
            for index in range(node.getNumGeoms()):
                state = node.getGeomState(index)
                attrib = state.getAttrib(MaterialAttrib.getClassSlot())
                if attrib is None:
                    continue
                material = attrib.getMaterial()
                style = _MATERIAL_STYLES.get(material.getName())
                if style is None:
                    continue
                diffuse, ambient, specular, shininess = style
                material.setDiffuse(Vec4(*diffuse))
                material.setAmbient(Vec4(*ambient))
                material.setSpecular(Vec4(*specular))
                material.setShininess(shininess)

    def _build_thrusters(self):
        if self.jet_origin is not None:
            try:
                rear = float(self.jet_origin.getY())
            except Exception as exc:
                log.debug("[ship] jet origin position unavailable: %s", exc)
                rear = -1.95
        else:
            rear = -1.95
        self.glow = make_uv_sphere(0.28, 8, 12, "engine_glow")
        self.glow.setPos(0.0, rear, 0.0)
        self.glow.setColor(1.0, 0.55, 0.20, 1.0)
        additive_blend(self.glow)
        self.glow.setLightOff(1)
        self.glow.setMaterialOff(1)
        self.glow.setDepthWrite(False)
        self.glow.setBin("transparent", 22)
        self.glow.reparentTo(self.root)

        self.flame = make_uv_sphere(1.0, 10, 14, "engine_flame")
        self.flame.setScale(0.15, 0.55, 0.15)
        self.flame.setPos(0.0, rear - 0.45, 0.0)
        self.flame.setColor(1.0, 0.55, 0.18, 1.0)
        additive_blend(self.flame)
        self.flame.setLightOff(1)
        self.flame.setMaterialOff(1)
        self.flame.setDepthWrite(False)
        self.flame.setBin("transparent", 21)
        self.flame.reparentTo(self.root)

    # -- runtime ------------------------------------------------------------
    def attach_to(self, parent):
        self.root.reparentTo(parent)

    def set_visible(self, visible):
        if visible:
            self.root.show()
        else:
            self.root.hide()

    def set_transform(self, position, quat):
        self.root.setPos(position)
        self.root.setQuat(quat)

    def update(self, dt, throttle, boost):
        self._time += dt
        self._throttle = abs(throttle)
        self._boost = boost
        flicker = 0.88 + 0.12 * math.sin(self._time * 31.0)
        intensity = self._throttle * (1.45 if boost else 1.0)
        length = 0.20 + 1.15 * intensity * flicker
        width = 0.10 + 0.10 * intensity
        self.flame.setScale(width, length, width)
        self.flame.setPos(0.0, -1.95 - length * 0.55, 0.0)
        if boost and throttle > 0.05:
            color = (0.48, 0.72, 1.0)
        else:
            color = (1.0, 0.55, 0.18)
        self.flame.setColor(color[0], color[1], color[2], 1.0)
        glow_scale = 0.22 + 0.55 * intensity
        self.glow.setScale(glow_scale)
        self.glow.setColor(color[0] * 0.85, color[1] * 0.85,
                           color[2] * 0.85, 1.0)
        if intensity > 0.02:
            self.flame.show()
            self.glow.show()
        else:
            self.flame.hide()
            self.glow.hide()

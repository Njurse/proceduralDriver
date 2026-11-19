"""Atmospheric sky dome, sun disc, cloud layers and their animation."""

from panda3d.core import (
    NodePath,
    TextureStage,
    TransparencyAttrib,
    Vec3,
)

import config
from core.geometry import make_glow_sphere, make_ground_plane, make_sky_dome
from core.textures import cloud_texture


class Sky:
    def __init__(self, base, palette, seed):
        self.base = base
        self.palette = palette
        self.root = NodePath("sky")
        self.time = 0.0
        self.sun_direction = Vec3(-0.42, 0.55, 0.62)
        self.sun_direction.normalize()

        self.dome = make_sky_dome(
            config.ATMO_SKY_RADIUS, palette["sky_top"],
            palette["sky_horizon"], rings=22, segments=44, name="sky_dome")
        self.dome.setLightOff(1)
        self.dome.setFogOff(1)
        self.dome.setDepthWrite(False)
        self.dome.setBin("background", 10)
        self.dome.reparentTo(self.root)

        self.sun = make_glow_sphere(210.0, palette["sun"], 1.25,
                                    "atmosphere_sun")
        self.sun.setFogOff(1)
        self.sun.setBin("transparent", 6)
        self.sun.reparentTo(self.root)

        self.cloud_layers = []
        self._pending_seed = seed

    def attach_clouds(self, coverage, speed):
        if coverage <= 0.12:
            return
        size = config.ATMO_WORLD_SIZE * 4.0
        for index in range(2):
            layer_coverage = max(0.10, coverage - index * 0.10)
            texture = cloud_texture(
                config.CLOUD_TEXTURE_WIDTH, config.CLOUD_TEXTURE_HEIGHT,
                self._pending_seed + index * 37, layer_coverage,
                self.palette["cloud"])
            plane = make_ground_plane(size, 0.0, f"cloud_{index}")
            stage = TextureStage(f"cloud_stage_{index}")
            plane.setTexture(stage, texture)
            plane.setTexScale(stage, 6.0 + 3.0 * index, 6.0 + 3.0 * index)
            plane.setTransparency(TransparencyAttrib.MAlpha)
            plane.setColor(1.0, 1.0, 1.0, 0.70 - index * 0.16)
            plane.setLightOff(1)
            plane.setMaterialOff(1)
            plane.setFogOff(1)
            plane.setDepthWrite(False)
            plane.setBin("transparent", 8 + index)
            plane.setPos(0.0, 0.0, config.ATMO_CLOUD_ALTITUDE
                         + index * 130.0)
            plane.reparentTo(self.root)
            self.cloud_layers.append({
                "plane": plane,
                "stage": stage,
                "speed": speed * (1.0 + index * 0.35),
                "altitude": config.ATMO_CLOUD_ALTITUDE + index * 130.0,
            })

    def update(self, dt, camera_pos, ship_pos):
        self.time += dt
        self.dome.setPos(camera_pos)
        self.sun.setPos(
            camera_pos + self.sun_direction * (config.ATMO_SKY_RADIUS * 0.72))
        for layer in self.cloud_layers:
            plane = layer["plane"]
            plane.setPos(ship_pos.x, ship_pos.y, layer["altitude"])
            u = (self.time * layer["speed"] + ship_pos.x * 0.00011)
            v = (self.time * layer["speed"] * 0.45 + ship_pos.y * 0.00011)
            plane.setTexOffset(layer["stage"], u, v)

    def destroy(self):
        self.root.removeNode()
        self.cloud_layers.clear()

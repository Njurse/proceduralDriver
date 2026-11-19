"""Composes a planet's atmosphere: terrain, water, sky, fog, weather, light."""

import logging
import random

from panda3d.core import (
    AmbientLight,
    DirectionalLight,
    Fog,
    Quat,
    Vec3,
    Vec4,
)

import config
from core.flight import _align_up, _surface_normal
from core.ship import Spaceship
from core.ship_profiles import random_flyable
from world.sky import Sky
from world.terrain import Terrain
from world.veins import spawn_veins
from world.weather import Weather

log = logging.getLogger("galaxy_explorer.atmosphere")


class AtmosphereScene:
    def __init__(self, base, system_info, planet):
        self.base = base
        self.info = system_info
        self.planet = planet
        self.palette = planet.palette
        self.root = base.render.attachNewNode(
            f"atmosphere_{system_info.index}_{planet.index}")
        self.time = 0.0

        self.terrain = Terrain(planet.seed, planet.archetype)
        self.terrain.build(self.root)
        self.veins = spawn_veins(
            base, self.root, planet.seed, self.terrain.height_at,
            config.ATMO_WORLD_SIZE, count=config.VEIN_COUNT,
            min_dist=config.VEIN_MIN_DIST)

        self.landed_ships = self._spawn_landed_ships(
            base, planet.seed, self.terrain)

        self.sky = Sky(base, self.palette, planet.seed)
        clouds = planet.archetype["clouds"]
        self.sky.attach_clouds(clouds["coverage"], clouds["speed"])
        self.sky.root.reparentTo(self.root)

        self.weather = Weather(base, planet.archetype, planet.seed)
        if self.weather.node is not None:
            self.weather.node.reparentTo(self.root)

        self._build_lights()

        fog = Fog("atmosphere_fog")
        fog.setColor(Vec4(self.palette["fog"][0], self.palette["fog"][1],
                          self.palette["fog"][2], 1.0))
        fog.setExpDensity(config.ATMO_FOG_DENSITY)
        self.fog = fog
        self.root.setFog(fog)

    def _build_lights(self):
        sun = DirectionalLight("atmosphere_sun_light")
        sun.setColor(Vec4(self.palette["sun"][0], self.palette["sun"][1],
                          self.palette["sun"][2], 1.0))
        sun.setDirection(-self.sky.sun_direction)
        self.sun_light = self.root.attachNewNode(sun)
        self.root.setLight(self.sun_light)

        ambient = AmbientLight("atmosphere_ambient")
        color = self.palette["ambient"]
        ambient.setColor(Vec4(color[0] * 0.60, color[1] * 0.60,
                              color[2] * 0.60, 1.0))
        self.ambient = self.root.attachNewNode(ambient)
        self.root.setLight(self.ambient)

    # -- scene API ----------------------------------------------------------
    def height_at(self, x, y):
        return self.terrain.height_at(x, y)

    def entry_position(self):
        return self.terrain.entry_position()

    def update(self, dt, camera_pos, ship_pos):
        self.time += dt
        self.sky.update(dt, camera_pos, ship_pos)
        if self.weather.active:
            self.weather.update(dt, camera_pos)
        # Distance LOD: only nearby veins and landed ships draw geometry.
        for vein in self.veins:
            if vein.depleted:
                continue
            if (ship_pos - vein.position).length() <= config.VEIN_LOD_DIST:
                vein.root.show()
            else:
                vein.root.hide()
        for entry in self.landed_ships:
            if (ship_pos - entry["position"]).length() <= config.LANDED_SHIP_LOD_DIST:
                entry["ship"].root.show()
            else:
                entry["ship"].root.hide()

    def set_visible(self, visible):
        if visible:
            self.root.show()
        else:
            self.root.hide()

    def _spawn_landed_ships(self, base, seed, terrain):
        rng = random.Random(seed + 777)
        count = rng.randint(config.LANDED_SHIP_MIN, config.LANDED_SHIP_MAX)
        result = []
        size = terrain.size
        half = size * 0.5
        water = terrain.water_level
        for i in range(count):
            # random position on terrain
            for _ in range(20):
                x = rng.uniform(-half * 0.7, half * 0.7)
                y = rng.uniform(-half * 0.7, half * 0.7)
                h = terrain.height_at(x, y)
                if water is not None and h < water + 2:
                    continue
                break
            normal = _surface_normal(terrain.height_at, x, y)
            profile = random_flyable(rng)
            ship = Spaceship(base, profile)
            ship.attach_to(self.root)
            ship.set_transform(Vec3(x, y, h + 0.5), _align_up(Quat(), normal))
            # random yaw around normal
            ship.root.setH(ship.root.getH() + rng.uniform(0, 360))
            result.append({"ship": ship, "profile": profile,
                           "position": Vec3(x, y, h)})
        return result

    def destroy(self):
        for vein in getattr(self, "veins", []):
            try:
                vein.destroy()
            except Exception as exc:
                log.debug("[atmosphere] vein cleanup failed: %s", exc)
        self.veins = []
        for entry in getattr(self, "landed_ships", []):
            try:
                entry["ship"].root.removeNode()
            except Exception as exc:
                log.debug("[atmosphere] landed ship cleanup failed: %s", exc)
        self.landed_ships = []
        self.weather.destroy()
        self.sky.destroy()
        self.terrain.destroy()
        self.root.removeNode()

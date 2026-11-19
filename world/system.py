"""Star-system data and its renderable scene (sun, orbits, planets)."""

import math
from dataclasses import dataclass, field

from panda3d.core import (
    AmbientLight,
    LineSegs,
    Material,
    PointLight,
    TextNode,
    TransparencyAttrib,
    Vec3,
    Vec4,
)

import config
from core.geometry import additive_blend, make_glow_sphere, make_uv_sphere
from core.textures import cloud_texture, planet_texture
from world.planet import build_planets

STAR_TYPES = (
    ("G-type yellow dwarf", (1.00, 0.93, 0.62), 4.0),
    ("K-type orange dwarf", (1.00, 0.72, 0.42), 3.0),
    ("M-type red dwarf", (1.00, 0.48, 0.32), 2.2),
    ("F-type white star", (1.00, 0.98, 0.86), 1.6),
    ("B-type blue giant", (0.58, 0.72, 1.00), 0.9),
    ("A-type blue-white", (0.76, 0.86, 1.00), 1.2),
)


def pick_star(rng):
    total = sum(star[2] for star in STAR_TYPES)
    roll = rng.random() * total
    for star in STAR_TYPES:
        roll -= star[2]
        if roll <= 0.0:
            return star[0], star[1]
    return STAR_TYPES[0][0], STAR_TYPES[0][1]


def generate_system(index, name, seed, position, rng, star_override=None):
    if star_override is not None:
        star_type, star_color = star_override
    else:
        star_type, star_color = pick_star(rng)
    count = rng.randint(config.SYSTEM_PLANET_MIN, config.SYSTEM_PLANET_MAX)
    planets = build_planets(rng, seed, count)
    return SystemInfo(index=index, name=name, seed=seed,
                      position=Vec3(position), star_color=star_color,
                      star_type=star_type, planets=planets)


@dataclass
class SystemInfo:
    index: int
    name: str
    seed: int
    position: Vec3
    star_color: tuple
    star_type: str
    planets: list = field(default_factory=list)

    @property
    def outer_radius(self):
        if not self.planets:
            return config.PLANET_ORBIT_BASE + 200.0
        return max(planet.orbit_radius + planet.radius
                   for planet in self.planets)

    @property
    def key(self):
        return str(self.index)


class StarSystemScene:
    """Builds and animates one system.  Parented under ``base.render``."""

    def __init__(self, base, info):
        self.base = base
        self.info = info
        self.root = base.render.attachNewNode(f"system_{info.index}")
        self.time = 0.0
        self._records = []
        self.labels = []
        self._build_sun()
        self._build_planets()

    # -- construction -------------------------------------------------------
    def _build_sun(self):
        color = self.info.star_color
        self.sun = make_uv_sphere(config.SUN_RADIUS, 24, 32, "sun_core")
        self.sun.setColor(color[0], color[1], color[2], 1.0)
        self.sun.setLightOff(1)
        self.sun.setMaterialOff(1)
        self.sun.reparentTo(self.root)

        self.sun_glow = make_glow_sphere(config.SUN_RADIUS * 1.05, color,
                                         1.15, "sun_glow")
        self.sun_glow.reparentTo(self.root)

        light = PointLight("sun_light")
        light.setColor(Vec4(color[0], color[1], color[2], 1.0))
        light.setAttenuation(Vec3(1.0, 0.0, 0.0))
        self.light = self.root.attachNewNode(light)
        self.root.setLight(self.light)

        ambient = AmbientLight("system_ambient")
        ambient.setColor(Vec4(0.20, 0.23, 0.30, 1.0))
        self.ambient = self.root.attachNewNode(ambient)
        self.root.setLight(self.ambient)

    def _orbit_ring(self, radius):
        lines = LineSegs("orbit")
        lines.setThickness(1.1)
        lines.setColor(0.55, 0.66, 0.82, 0.22)
        steps = 128
        for index in range(steps + 1):
            angle = math.tau * index / steps
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            if index == 0:
                lines.moveTo(x, y, 0.0)
            else:
                lines.drawTo(x, y, 0.0)
        node = self.root.attachNewNode(lines.create())
        node.setTransparency(TransparencyAttrib.MAlpha)
        node.setLightOff(1)
        node.setDepthWrite(False)
        return node

    def _build_planets(self):
        for planet in self.info.planets:
            archetype = planet.archetype
            palette = planet.palette
            self._orbit_ring(planet.orbit_radius)

            texture = planet_texture(
                config.PLANET_TEXTURE_WIDTH, config.PLANET_TEXTURE_HEIGHT,
                palette, planet.seed, frozen_caps=archetype["caps"])
            sphere = make_uv_sphere(planet.radius, 26, 38,
                                    f"planet_{planet.index}")
            sphere.setTexture(texture)
            material = Material(f"planet_mat_{planet.index}")
            material.setDiffuse(Vec4(1.0, 1.0, 1.0, 1.0))
            material.setAmbient(Vec4(0.16, 0.17, 0.20, 1.0))
            material.setSpecular(Vec4(0.10, 0.10, 0.12, 1.0))
            material.setShininess(9.0)
            sphere.setMaterial(material)
            sphere.reparentTo(self.root)
            position = planet.orbit_position(self.time)
            sphere.setPos(position)
            sphere.setR(12.0)

            cloud = None
            coverage = archetype["clouds"]["coverage"]
            if coverage > 0.15:
                cloud_tex = cloud_texture(
                    config.CLOUD_TEXTURE_WIDTH, config.CLOUD_TEXTURE_HEIGHT,
                    planet.seed + 17, coverage, palette["cloud"])
                cloud = make_uv_sphere(planet.radius * 1.018, 20, 30,
                                       f"clouds_{planet.index}")
                cloud.setTexture(cloud_tex)
                cloud.setTransparency(TransparencyAttrib.MAlpha)
                cloud.setLightOff(1)
                cloud.setMaterialOff(1)
                cloud.setDepthWrite(False)
                cloud.setBin("transparent", 12)
                cloud.setPos(position)
                cloud.setR(12.0)
                cloud.reparentTo(self.root)

            halo = make_uv_sphere(planet.radius * 1.05, 18, 24,
                                  f"halo_{planet.index}")
            halo.setColor(palette["sky_horizon"][0] * 0.22,
                          palette["sky_horizon"][1] * 0.22,
                          palette["sky_horizon"][2] * 0.22, 1.0)
            additive_blend(halo)
            halo.setLightOff(1)
            halo.setMaterialOff(1)
            halo.setDepthWrite(False)
            halo.setBin("transparent", 10)
            halo.setPos(position)
            halo.reparentTo(self.root)

            label_node = TextNode(f"label_{planet.index}")
            label_node.setText(planet.name)
            label_node.setTextColor(0.82, 0.90, 1.0, 0.72)
            label_node.setAlign(TextNode.ACenter)
            label = self.root.attachNewNode(label_node)
            label.setScale(2.2)
            label.setBillboardPointEye()
            label.setLightOff(1)
            label.setTransparency(TransparencyAttrib.MAlpha)
            label.setPos(position + Vec3(0.0, 0.0, -planet.radius - 5.0))
            self.labels.append(label)

            self._records.append({
                "planet": planet,
                "sphere": sphere,
                "cloud": cloud,
                "halo": halo,
                "label": label,
            })

    # -- runtime ------------------------------------------------------------
    def planet_position(self, planet):
        return planet.orbit_position(self.time)

    def update(self, dt):
        self.time += dt
        for record in self._records:
            planet = record["planet"]
            position = planet.orbit_position(self.time)
            spin = self.time * planet.spin_speed * 10.0
            record["sphere"].setPos(position)
            record["sphere"].setH(spin)
            if record["cloud"] is not None:
                record["cloud"].setPos(position)
                record["cloud"].setH(-spin * 1.35)
            record["halo"].setPos(position)
            record["label"].setPos(
                position + Vec3(0.0, 0.0, -planet.radius - 5.0))
        self.sun.setH(self.time * 4.0)

    def nearest_planet(self, position):
        """Return ``(planet, surface_distance, world_position)``."""
        best = (None, float("inf"), None)
        for record in self._records:
            planet = record["planet"]
            planet_pos = planet.orbit_position(self.time)
            distance = (position - planet_pos).length() - planet.radius
            if distance < best[1]:
                best = (planet, distance, planet_pos)
        return best

    def set_visible(self, visible):
        if visible:
            self.root.show()
        else:
            self.root.hide()

    def destroy(self):
        self.root.removeNode()
        self._records.clear()
        self.labels.clear()

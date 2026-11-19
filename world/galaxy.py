"""Seeded spiral galaxy: system placement, markers and starfield."""

import math
import random

import numpy as np
from panda3d.core import TextNode, TransparencyAttrib, Vec3

import config
from core.geometry import make_glow_sphere, make_points
from world.naming import unique_name
from world.system import generate_system


def make_starfield(count, radius, seed):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(count, 3))
    lengths = np.linalg.norm(directions, axis=1, keepdims=True)
    directions /= np.maximum(lengths, 1e-6)
    distances = radius * (0.45 + 0.55 * rng.random(count))[:, None]
    positions = directions * distances

    warmth = rng.random(count)
    colors = np.empty((count, 4), dtype=np.float64)
    colors[:, 0] = 0.55 + 0.40 * warmth
    colors[:, 1] = 0.62 + 0.34 * (1.0 - np.abs(warmth - 0.55) * 1.4)
    colors[:, 2] = 0.78 + 0.22 * (1.0 - warmth)
    colors[:, 3] = 1.0
    colors[: max(count // 18, 1)] = (1.0, 1.0, 1.0, 1.0)
    return make_points(positions, colors, size=1.7, name="starfield")


def _arm_dust(seed, count=1400):
    rng = np.random.default_rng(seed)
    arm = rng.integers(0, config.GALAXY_ARM_COUNT, count)
    fraction = rng.random(count) ** 0.72
    angle = (arm * math.tau / config.GALAXY_ARM_COUNT
             + fraction * config.GALAXY_ARM_TWIST
             + rng.normal(scale=0.14, size=count))
    radius = config.GALAXY_RADIUS * (0.16 + 0.92 * fraction)
    radius += rng.normal(scale=45.0, size=count)
    x = np.cos(angle) * radius
    y = np.sin(angle) * radius
    z = rng.normal(scale=34.0, size=count)
    positions = np.stack([x, y, z], axis=1)

    brightness = 0.10 + 0.30 * rng.random(count)
    colors = np.stack([
        brightness * (0.65 + 0.35 * rng.random(count)),
        brightness * (0.72 + 0.28 * rng.random(count)),
        brightness * 1.25,
        np.ones(count),
    ], axis=1)
    return make_points(positions, colors, size=3.4, name="arm_dust")


class Galaxy:
    def __init__(self, seed):
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.systems = []
        self._generate()

    def _generate(self):
        used_names = set()
        self.systems.append(generate_system(
            index=0,
            name=unique_name(self.rng, used_names),
            seed=self.seed + 101,
            position=Vec3(0.0, 0.0, 0.0),
            rng=self.rng,
            star_override=("G-type yellow dwarf", (1.00, 0.93, 0.62)),
        ))

        attempts = 0
        while len(self.systems) < config.GALAXY_SYSTEM_COUNT and attempts < 200000:
            attempts += 1
            fraction = self.rng.uniform(0.30, 1.0)
            arm = self.rng.randrange(config.GALAXY_ARM_COUNT)
            angle = (arm * math.tau / config.GALAXY_ARM_COUNT
                     + fraction * config.GALAXY_ARM_TWIST
                     + self.rng.gauss(0.0, 0.22))
            radius = config.GALAXY_RADIUS * fraction
            position = Vec3(
                math.cos(angle) * radius,
                math.sin(angle) * radius,
                self.rng.uniform(-45.0, 45.0),
            )
            if position.length() < config.GALAXY_START_DIST + 260.0:
                continue
            if any((position - system.position).length() < 420.0
                   for system in self.systems):
                continue
            index = len(self.systems)
            self.systems.append(generate_system(
                index=index,
                name=unique_name(self.rng, used_names),
                seed=self.seed + 101 + index * 4507,
                position=position,
                rng=self.rng,
            ))

    # -- queries ------------------------------------------------------------
    @property
    def home(self):
        return self.systems[0]

    @property
    def start_position(self):
        return Vec3(0.0, -config.GALAXY_START_DIST, 0.0)

    def system_at(self, index):
        if index is None or index < 0 or index >= len(self.systems):
            return None
        return self.systems[index]

    def nearest_system(self, position):
        best = (None, float("inf"))
        for system in self.systems:
            distance = (system.position - position).length()
            if distance < best[1]:
                best = (system, distance)
        return best


class GalaxyScene:
    """System markers, labels and the galactic dust disk."""

    def __init__(self, base, galaxy):
        self.base = base
        self.galaxy = galaxy
        self.root = base.render.attachNewNode("galaxy_scene")
        self.labels = []
        self._build()

    def _build(self):
        dust = _arm_dust(self.galaxy.seed + 55)
        dust.reparentTo(self.root)

        for system in self.galaxy.systems:
            glow = make_glow_sphere(21.0, system.star_color, 1.5,
                                    f"marker_{system.index}")
            glow.setPos(system.position)
            glow.reparentTo(self.root)

            label_node = TextNode(f"system_label_{system.index}")
            label_node.setText(system.name)
            label_node.setTextColor(0.80, 0.88, 1.0, 0.78)
            label_node.setAlign(TextNode.ACenter)
            label = self.root.attachNewNode(label_node)
            label.setScale(8.5)
            label.setPos(system.position + Vec3(0.0, 0.0, 78.0))
            label.setBillboardPointEye()
            label.setLightOff(1)
            label.setTransparency(TransparencyAttrib.MAlpha)
            self.labels.append(label)

    def set_visible(self, visible):
        if visible:
            self.root.show()
        else:
            self.root.hide()

    def update(self, dt):
        pass

    def destroy(self):
        self.root.removeNode()
        self.labels.clear()

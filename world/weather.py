"""Per-planet particle weather: rain, snow, ash and dust."""

import numpy as np
from panda3d.core import GeomVertexWriter, Vec3

import config
from core.geometry import make_points
from core.mathutil import clamp


class Weather:
    def __init__(self, base, archetype, seed):
        self.base = base
        self.kind = archetype["particles"]["kind"]
        self.node = None
        self.writer = None
        self.count = 0
        self.time = 0.0
        self.radius = config.ATMO_PARTICLE_RADIUS
        self._rng = np.random.default_rng(seed + 431)
        self._center = Vec3(0.0, 0.0, 0.0)

        if self.kind == "none":
            return
        density = clamp(archetype["particles"]["density"], 0.0, 1.0)
        if density <= 0.02:
            return
        self.count = max(70, int(config.ATMO_PARTICLE_COUNT * density))
        self._speed = float(archetype["particles"]["speed"])

        radius = self.radius
        rng = self._rng
        local = np.empty((self.count, 3), dtype=np.float64)
        local[:, 0] = rng.uniform(-radius, radius, self.count)
        local[:, 1] = rng.uniform(-radius, radius, self.count)
        local[:, 2] = rng.uniform(-radius * 0.55, radius * 0.85,
                                  self.count)
        self.local = local
        self._phase = rng.uniform(0.0, np.pi * 2.0, self.count)

        color = archetype["particles"]["color"]
        colors = np.tile(np.array([color[0], color[1], color[2], 1.0]),
                         (self.count, 1))
        self.node = make_points(local.copy(), colors,
                                size=float(archetype["particles"]["size"]),
                                dynamic=True, name="weather_particles")
        geom_node = self.node.node()
        vdata = geom_node.modifyGeom(0).modifyVertexData()
        self.writer = GeomVertexWriter(vdata, "vertex")

    @property
    def active(self):
        return self.count > 0

    def update(self, dt, center):
        if not self.active:
            return
        self.time += dt
        local = self.local
        radius = self.radius
        rng = self._rng

        if self.kind in ("rain", "snow", "ash"):
            local[:, 2] -= self._speed * dt
            if self.kind == "snow":
                local[:, 0] += np.sin(self.time * 2.1 + self._phase) * 9.0 * dt
                local[:, 1] += np.cos(self.time * 1.7 + self._phase) * 7.0 * dt
            elif self.kind == "rain":
                local[:, 0] += 9.0 * dt
            else:
                local[:, 0] += 3.5 * dt
            below = local[:, 2] < -radius * 0.65
            wrapped = int(below.sum())
            if wrapped:
                local[below, 2] += radius * 1.5
                local[below, 0] = rng.uniform(-radius, radius, wrapped)
                local[below, 1] = rng.uniform(-radius, radius, wrapped)
            np.clip(local[:, 0], -radius, radius, out=local[:, 0])
            np.clip(local[:, 1], -radius, radius, out=local[:, 1])
        else:
            local[:, 0] += self._speed * dt
            local[:, 1] += self._speed * 0.35 * dt
            local[:, 0] = np.where(local[:, 0] > radius, -radius,
                                   local[:, 0])
            local[:, 1] = np.where(local[:, 1] > radius, -radius,
                                   local[:, 1])

        cx, cy, cz = center[0], center[1], center[2]
        writer = self.writer
        for index in range(self.count):
            writer.setRow(index)
            writer.addData3f(cx + local[index, 0],
                             cy + local[index, 1],
                             cz + local[index, 2])

    def destroy(self):
        if self.node is not None:
            self.node.removeNode()
        self.node = None
        self.writer = None
        self.count = 0

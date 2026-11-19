"""Heightfield terrain generation, tinting, water and height sampling."""

import numpy as np
from panda3d.core import Material, TransparencyAttrib, Vec4

import config
from core.geometry import make_ground_plane, make_heightfield
from core.noise import fbm_2d, terrain_height


class Terrain:
    def __init__(self, seed, archetype):
        self.seed = seed
        self.archetype = archetype
        self.palette = archetype["palette"]
        self.params = dict(archetype["terrain"])
        self.size = config.ATMO_WORLD_SIZE
        self.grid = config.ATMO_GRID
        self.amplitude = float(self.params["amplitude"])

        coords = np.linspace(-self.size * 0.5, self.size * 0.5, self.grid)
        self._coords = coords
        grid_x, grid_y = np.meshgrid(coords, coords)
        self.heights = terrain_height(grid_x, grid_y, seed, self.params)

        water = self.params.get("water")
        self.water_level = (None if water is None
                            else self.amplitude * float(water))
        self.mesh = None
        self.water = None

    # -- sampling -----------------------------------------------------------
    def height_at(self, x, y):
        half = self.size * 0.5
        gx = (float(x) + half) / self.size * (self.grid - 1)
        gy = (float(y) + half) / self.size * (self.grid - 1)
        gx = min(max(gx, 0.0), self.grid - 1.001)
        gy = min(max(gy, 0.0), self.grid - 1.001)
        x0 = int(gx)
        y0 = int(gy)
        fx = gx - x0
        fy = gy - y0
        x1 = min(x0 + 1, self.grid - 1)
        y1 = min(y0 + 1, self.grid - 1)
        h00 = self.heights[y0, x0]
        h10 = self.heights[y0, x1]
        h01 = self.heights[y1, x0]
        h11 = self.heights[y1, x1]
        top = h00 * (1.0 - fx) + h10 * fx
        bottom = h01 * (1.0 - fx) + h11 * fx
        return float(top * (1.0 - fy) + bottom * fy)

    # -- colours ------------------------------------------------------------
    def _colors(self):
        heights = self.heights
        amplitude = self.amplitude
        water = (self.water_level if self.water_level is not None
                 else -amplitude * 0.75)
        normalised = np.clip((heights - water) / (amplitude * 0.9), 0.0, 1.0)

        stops = (0.0, 0.26, 0.55, 0.80, 1.0)
        keys = ("water", "low", "mid", "high", "peak")
        rgb = np.empty(heights.shape + (3,), dtype=np.float64)
        flat = normalised.ravel()
        for channel in range(3):
            values = [self.palette[key][channel] for key in keys]
            rgb[..., channel] = np.interp(
                flat, stops, values).reshape(heights.shape)

        grid_x, grid_y = np.meshgrid(self._coords, self._coords)
        moisture = fbm_2d(grid_x * 0.0011 + 11.0, grid_y * 0.0011 - 7.0,
                          self.seed + 5, 3)
        rgb *= (0.88 + 0.24 * moisture)[..., None]

        if self.water_level is not None:
            underwater = heights < self.water_level
            if underwater.any():
                depth = np.clip(
                    (self.water_level - heights) / (amplitude * 0.35),
                    0.0, 1.0)
                deep = np.asarray(self.palette["deep"], dtype=np.float64)
                blended = (rgb * (1.0 - depth[..., None])
                           + deep * depth[..., None])
                rgb[underwater] = blended[underwater]

        rgba = np.empty(heights.shape + (4,), dtype=np.float64)
        rgba[..., :3] = np.clip(rgb, 0.0, 1.0)
        rgba[..., 3] = 1.0
        return rgba

    # -- scene --------------------------------------------------------------
    def build(self, parent):
        self.mesh = make_heightfield(self.heights, self.size, self._colors(),
                                     "terrain")
        material = Material("terrain_material")
        material.setDiffuse(Vec4(1.0, 1.0, 1.0, 1.0))
        material.setAmbient(Vec4(0.52, 0.52, 0.52, 1.0))
        material.setSpecular(Vec4(0.02, 0.02, 0.02, 1.0))
        material.setShininess(2.0)
        self.mesh.setMaterial(material)
        self.mesh.reparentTo(parent)

        if self.water_level is not None:
            water_color = self.palette["water"]
            self.water = make_ground_plane(self.size * 1.7, 0.0, "water")
            self.water.setPos(0.0, 0.0, self.water_level)
            self.water.setColor(water_color[0], water_color[1],
                                water_color[2], 0.86)
            self.water.setTransparency(TransparencyAttrib.MAlpha)
            self.water.setLightOff(1)
            self.water.setMaterialOff(1)
            self.water.setDepthWrite(False)
            self.water.setBin("transparent", 4)
            self.water.reparentTo(parent)
        return self.mesh

    def entry_position(self):
        spawn_y = -self.size * 0.08
        ground = self.height_at(0.0, spawn_y)
        return (0.0, spawn_y, ground + config.ATMO_ENTRY_ALTITUDE)

    def destroy(self):
        for node in (self.mesh, self.water):
            if node is not None:
                node.removeNode()
        self.mesh = None
        self.water = None

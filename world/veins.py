"""Procedural mineral veins scattered over a planet's terrain."""

import logging
import math
import random

from panda3d.core import Vec3

import config
from core.geometry import make_glow_sphere, make_uv_sphere
from core.materials import pick
from core.mathutil import clamp

log = logging.getLogger("galaxy_explorer.veins")


class Vein:
    """A single minable mineral node on a planet's surface."""

    def __init__(self, base, parent, position, material, seed):
        self.base = base
        self.material = material
        self.seed = seed
        self.position = Vec3(position)
        # Harder materials need several mining cycles and yield fewer units.
        tier = material.get("tier", "common")
        self.hardness = max(1, int(config.MINING_HARDNESS.get(tier, 1)))
        low, high = config.MINING_YIELD.get(tier, (1, 1))
        self.yield_amount = random.Random(seed + 733).randint(low, high)
        self.last_yield = 0
        self.hp = float(self.hardness)
        self.max_hp = float(self.hardness)
        self.radius = random.uniform(1.6, 3.2)
        self.depleted = False

        self.root = parent.attachNewNode(f"vein_{material['id']}_{seed}")
        self.root.setPos(self.position)

        # Procedural morphology: crystal cluster / nodule / plate.
        rng = random.Random(seed)
        morph = rng.choice(("cluster", "nodule", "plate"))
        self._build(morph, rng)

        # Glow halo for visibility at distance.
        glow = make_glow_sphere(self.radius * 1.1, material["color"],
                                intensity=0.6 + material["glow"], name="vein_glow")
        glow.reparentTo(self.root)
        glow.setScale(1.0)

        # Slight random yaw so clusters don't look identical.
        self.root.setHpr(rng.uniform(0, 360), rng.uniform(-8, 8),
                        rng.uniform(-8, 8))

    def _build(self, morph, rng):
        color = self.material["color"]
        glow = self.material["glow"]
        if morph == "cluster":
            count = rng.randint(5, 11)
            for i in range(count):
                r = self.radius * rng.uniform(0.25, 0.65)
                sphere = make_uv_sphere(r, 8, 12, f"vein_crystal_{i}")
                sphere.setColor(color[0], color[1], color[2], 1.0)
                sphere.setPos(
                    rng.uniform(-self.radius * 0.7, self.radius * 0.7),
                    rng.uniform(-self.radius * 0.7, self.radius * 0.7),
                    rng.uniform(0.0, self.radius * 0.9),
                )
                sphere.reparentTo(self.root)
            if glow > 0:
                core = make_uv_sphere(self.radius * 0.35, 6, 8, "vein_core")
                core.setColor(color[0] * 1.3, color[1] * 1.3, color[2] * 1.3, 1.0)
                core.reparentTo(self.root)
        elif morph == "nodule":
            body = make_uv_sphere(self.radius, 10, 14, "vein_nodule")
            body.setColor(color[0] * 0.85, color[1] * 0.85, color[2] * 0.85, 1.0)
            body.reparentTo(self.root)
            for i in range(rng.randint(2, 5)):
                bump = make_uv_sphere(self.radius * rng.uniform(0.25, 0.45),
                                      6, 8, f"vein_bump_{i}")
                bump.setColor(color[0], color[1], color[2], 1.0)
                bump.setPos(
                    rng.uniform(-self.radius * 0.6, self.radius * 0.6),
                    rng.uniform(-self.radius * 0.6, self.radius * 0.6),
                    self.radius * 0.6,
                )
                bump.reparentTo(self.root)
        else:  # plate
            # Flat-ish cluster: squashed sphere + shards.
            body = make_uv_sphere(self.radius, 8, 12, "vein_plate")
            body.setScale(1.0, 1.0, 0.35)
            body.setColor(color[0], color[1], color[2], 1.0)
            body.reparentTo(self.root)
            for i in range(rng.randint(3, 7)):
                shard = make_uv_sphere(self.radius * rng.uniform(0.15, 0.35),
                                       5, 7, f"vein_shard_{i}")
                shard.setColor(color[0] * 1.1, color[1] * 1.1, color[2] * 1.1, 1.0)
                shard.setPos(
                    rng.uniform(-self.radius * 0.8, self.radius * 0.8),
                    rng.uniform(-self.radius * 0.8, self.radius * 0.8),
                    rng.uniform(0.1, 0.6),
                )
                shard.reparentTo(self.root)

    def hit(self, origin, direction):
        """Return (hit_bool, distance) for a ray vs this vein's bounding sphere."""
        oc = self.position - origin
        tca = oc.dot(direction)
        if tca < 0.0:
            return False, 0.0
        d2 = oc.dot(oc) - tca * tca
        r = self.radius * 1.15
        if d2 > r * r:
            return False, 0.0
        thc = math.sqrt(max(r * r - d2, 0.0))
        t0 = tca - thc
        if t0 < 0.0:
            t0 = tca + thc
        return True, float(t0)

    def damage(self, amount):
        if self.depleted:
            return False
        self.hp = max(0.0, self.hp - amount)
        if self.hp <= 0.0:
            self.depleted = True
            self.last_yield = self.yield_amount
            try:
                self.root.hide()
            except Exception as exc:
                log.debug("[veins] could not hide depleted vein: %s", exc)
            return True
        return False

    def destroy(self):
        try:
            self.root.removeNode()
        except Exception as exc:
            log.debug("[veins] vein cleanup failed: %s", exc)


def poisson_positions(size, count, min_dist, rng, density_mask=None):
    """Bridson-style Poisson disk sampling over a square [-size/2, size/2]."""
    half = size * 0.5
    cell = min_dist / math.sqrt(2.0)
    grid_w = int(size / cell) + 1
    grid = [[None] * grid_w for _ in range(grid_w)]
    samples = []
    active = []

    def grid_index(x, y):
        return int((x + half) / cell), int((y + half) / cell)

    # Seed.
    x0 = rng.uniform(-half, half)
    y0 = rng.uniform(-half, half)
    if density_mask is None or density_mask(x0, y0) > rng.random():
        samples.append((x0, y0))
        active.append((x0, y0))
        gx, gy = grid_index(x0, y0)
        if 0 <= gx < grid_w and 0 <= gy < grid_w:
            grid[gy][gx] = (x0, y0)

    attempts = 0
    max_attempts = count * 30
    while active and len(samples) < count and attempts < max_attempts:
        attempts += 1
        idx = rng.randrange(len(active))
        px, py = active[idx]
        found = False
        for _ in range(30):
            ang = rng.uniform(0, math.tau)
            rad = rng.uniform(min_dist, min_dist * 2.0)
            nx = px + math.cos(ang) * rad
            ny = py + math.sin(ang) * rad
            if nx < -half or nx > half or ny < -half or ny > half:
                continue
            if density_mask is not None and density_mask(nx, ny) <= rng.random():
                continue
            gi, gj = grid_index(nx, ny)
            if not (0 <= gi < grid_w and 0 <= gj < grid_w):
                continue
            ok = True
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    j, i = gj + dy, gi + dx
                    if 0 <= i < grid_w and 0 <= j < grid_w and grid[j][i] is not None:
                        sx, sy = grid[j][i]
                        if (sx - nx) ** 2 + (sy - ny) ** 2 < min_dist * min_dist:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                samples.append((nx, ny))
                active.append((nx, ny))
                grid[gj][gi] = (nx, ny)
                found = True
                break
        if not found:
            active.pop(idx)
    return samples


def spawn_veins(base, parent, seed, height_sampler, size, count=60, min_dist=90.0):
    """Scatter veins over a terrain. Returns list[Vein]."""
    rng = random.Random(seed + 911)
    # Density mask: veins prefer mid/high elevations.
    def density(x, y):
        h = height_sampler(x, y)
        # Normalize roughly; amplitude ~ a few hundred.
        return 0.35 + 0.65 * clamp((h + 100.0) / 400.0, 0.0, 1.0)

    samples = poisson_positions(size, count, min_dist, rng, density)
    veins = []
    for i, (x, y) in enumerate(samples):
        h = height_sampler(x, y)
        # Skip underwater-ish low spots occasionally.
        if h < -50 and rng.random() < 0.6:
            continue
        material = pick(random.Random(seed + i * 37))
        vein = Vein(base, parent, (x, y, h + 0.2), material, seed + i * 101)
        veins.append(vein)
    return veins

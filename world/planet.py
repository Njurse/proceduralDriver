"""Planet metadata: archetype, orbit and naming."""

import math
from dataclasses import dataclass

from panda3d.core import Vec3

import config
from world.archetypes import get_archetype, pick_archetype
from world.naming import planet_name, unique_name


@dataclass
class PlanetInfo:
    index: int
    name: str
    archetype_id: str
    seed: int
    radius: float
    orbit_radius: float
    orbit_phase: float
    orbit_speed: float
    spin_speed: float

    @property
    def archetype(self):
        return get_archetype(self.archetype_id)

    @property
    def palette(self):
        return self.archetype["palette"]

    def orbit_position(self, time):
        angle = self.orbit_phase + time * self.orbit_speed
        return Vec3(math.cos(angle) * self.orbit_radius,
                    math.sin(angle) * self.orbit_radius, 0.0)

    def key(self, system_index):
        return f"{system_index}:{self.index}"

    def summary(self):
        archetype = self.archetype
        return (f"{self.name} - {archetype['label']}  "
                f"({archetype['weather']})")


def build_planets(rng, seed, count):
    """Create ``count`` deterministic planets for one system."""
    used = set()
    planets = []
    for index in range(count):
        archetype = pick_archetype(rng)
        orbit_radius = (config.PLANET_ORBIT_BASE
                        + index * config.PLANET_ORBIT_STEP
                        + rng.uniform(-config.PLANET_ORBIT_JITTER,
                                      config.PLANET_ORBIT_JITTER))
        orbit_radius = max(orbit_radius, config.PLANET_ORBIT_BASE * 0.55)
        base_radius = rng.uniform(15.0, 26.0)
        if archetype["id"] == "ocean":
            base_radius *= 1.18
        elif archetype["id"] == "volcanic":
            base_radius *= 0.92
        planets.append(PlanetInfo(
            index=index,
            name=unique_name(rng, used, generator=planet_name),
            archetype_id=archetype["id"],
            seed=seed + index * 7919,
            radius=base_radius,
            orbit_radius=orbit_radius,
            orbit_phase=rng.uniform(0.0, math.tau),
            orbit_speed=rng.uniform(0.012, 0.030),
            spin_speed=rng.uniform(0.03, 0.11) * (1 if rng.random() > 0.25
                                                 else -1),
        ))
    return planets

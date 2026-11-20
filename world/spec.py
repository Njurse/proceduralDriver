"""The game's worldgen settings: biome files under assets/ and config.py."""

import config
from worldgen import biomes as _biomes
from worldgen.spec import RoadSpec, WorldSpec
from worldgen.traffic import TrafficSpec

DEFINITIONS_DIR = config.ASSETS / "world" / "definitions"


def load_biome(name):
    """assets/world/definitions/<name>.json (or a path to a .json)."""
    return _biomes.load_biome(name, DEFINITIONS_DIR, config.ASSETS)


def load_biomes(names=None):
    """The named biomes, config.WORLD_BIOMES by default."""
    return _biomes.load_biomes(config.WORLD_BIOMES if names is None else names,
                               DEFINITIONS_DIR, config.ASSETS)


def world_spec(seed=None, biomes=None):
    """WorldSpec for ``seed`` (config.TERRAIN_SEED by default) over ``biomes``
    (config.WORLD_BIOMES by default)."""
    return WorldSpec(seed=int(config.TERRAIN_SEED if seed is None else seed),
                     biomes=tuple(load_biomes() if biomes is None else biomes),
                     region_scale_m=config.REGION_SCALE_M, region_jitter=config.REGION_JITTER,
                     region_band_m=config.REGION_BAND_M,
                     region_warp_m=config.REGION_WARP_M,
                     region_warp_scale_m=config.REGION_WARP_SCALE_M,
                     chunk_size_m=config.TERRAIN_CHUNK_SIZE,
                     chunk_res=config.TERRAIN_CHUNK_RES,
                     roads=RoadSpec(min_junction_deg=config.ROAD_MIN_JUNCTION_DEG,
                                    lift_m=config.ROAD_LIFT_M,
                                    detail_tile_m=config.ROAD_DETAIL_TILE_M))


def traffic_spec():
    """TrafficSpec from config's TRAFFIC_* settings."""
    return TrafficSpec(max_cars=config.TRAFFIC_MAX, speed_ms=config.TRAFFIC_SPEED_MS,
                       spawn_min_m=config.TRAFFIC_SPAWN_MIN_M, spawn_m=config.TRAFFIC_SPAWN_M,
                       despawn_m=config.TRAFFIC_DESPAWN_M, gap_m=config.TRAFFIC_GAP_M,
                       accel=config.TRAFFIC_ACCEL, brake=config.TRAFFIC_BRAKE)

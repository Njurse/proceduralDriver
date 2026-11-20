"""WorldSpec: everything a world is generated from, passed in explicitly
(the game builds one from config.py and its biome files)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RoadSpec:
    min_junction_deg: float = 45.0   # no two roads leave a junction closer than this
    lift_m: float = 0.04             # road strips sit this far above the flattened ground
    detail_tile_m: float = 2.5       # metres per repeat of the road detail texture


@dataclass(frozen=True)
class WorldSpec:
    seed: int
    biomes: tuple                    # worldgen.biomes.Biome, at least one
    region_scale_m: float = 900.0    # spacing of the region sites that splotch the biomes
    region_jitter: float = 0.35      # site shift, as a fraction of the spacing
    region_band_m: float = 240.0     # width of the blend between neighbouring regions
    chunk_size_m: float = 64.0       # metres per square chunk (meshes, road strips)
    roads: RoadSpec = RoadSpec()

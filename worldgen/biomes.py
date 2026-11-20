"""Biome definitions: assets/world/definitions/<name>.json.

A definition may carry any fields of its own (kept in ``Biome.extra``); the
generator reads two sections and fills every missing key from the defaults
below:

``terrain``  the biome's height field (worldgen.height.Terrain: fbm, ridged
             or terraced noise) and the colour ramp its chunks are painted with
``roads``    the roads the biome grows (world.roads), with
             optional dashed centre ``markings``
``weight``   (top level) how often worldgen.regions picks the biome for a region
``region_size`` (top level) the biome claims blocks of this many x this many
             region sites, so 3 makes regions about three times as wide

A bad value or an unknown key inside those sections raises ValueError naming
the file and the key, since a typo would otherwise fall back to the default
without a word.  The ground detail texture is shared by every chunk, so the
world takes ``detail_texture`` and ``detail_tile_m`` from its first biome.
"""

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

# Every road surface drives like tarmac for now (core.car_forces only treats
# grass differently); the name is there for when the tyres learn about sand.
ROAD_SURFACES = ("asphalt", "concrete", "packed_sand", "dirt", "gravel")
NOISE_TYPES = ("fbm", "ridged", "terraced")

RGB = tuple[float, float, float]


@dataclass(frozen=True)
class ColorRamp:
    """Vertex colours: low/mid/high by height, steep on slopes."""

    low: RGB = (0.14, 0.28, 0.10)
    mid: RGB = (0.32, 0.47, 0.19)
    high: RGB = (0.66, 0.60, 0.33)
    steep: RGB = (0.48, 0.38, 0.25)


@dataclass(frozen=True)
class TerrainRules:
    noise_scale_m: float = 140.0     # wavelength of the first octave
    amplitude_m: float = 10.0        # peak-to-peak height, about
    octaves: int = 4
    persistence: float = 0.30        # amplitude ratio between octaves
    lacunarity: float = 2.0          # frequency ratio between octaves
    base_height_m: float = 0.0       # the biome's mean ground level
    steep_slope_deg: float = 11.0    # slopes past this take the steep colour
    noise_type: str = "fbm"          # one of NOISE_TYPES
    terrace_steps: int = 6           # "terraced": shelves across the amplitude
    terrace_ramp: float = 0.35       # "terraced": share of each step that slopes
    colors: ColorRamp = field(default_factory=ColorRamp)
    detail_texture: str = "world/landscape/ground_texture_noise.png"  # under assets/
    detail_tile_m: float = 8.0       # metres per repeat of the detail texture


@dataclass(frozen=True)
class RoadRules:
    grid_spacing_m: float = 200.0    # distance between parallel roads
    jitter_m: float = 25.0           # random shift of each grid node
    curviness_m: float = 40.0        # sideways bow of each road between nodes
    width_m: float = 16.0            # road surface width
    shoulder_m: float = 10.0         # terrain blends back from the road over this
    max_grade_pct: float = 8.0       # steepest climb along a road
    surface: str = "asphalt"         # one of ROAD_SURFACES
    color: RGB = (0.30, 0.30, 0.32)
    flatten: bool = True             # cut/fill the terrain to the road's profile
    markings: RGB | None = None      # dashed centre line colour, or none
    grid: float = 0.0                # 0 organic roads .. 1 a city grid
    grid_heading_deg: float | None = None  # grid heading; none: random per region
    loop_odds: float = 0.12          # organic: odds a short non-tree link makes a loop
    l_corner_odds: float = 0.0       # organic: odds a road turns a square L corner
    corner_radius_m: float | None = None  # bend radius; none: one smooth curve, 0: sharp


@dataclass(frozen=True)
class Biome:
    name: str
    terrain: TerrainRules
    roads: RoadRules
    extra: dict                      # every other top-level field, as read
    weight: float = 1.0              # relative odds of the biome per region
    region_size: int = 1             # regions span this many sites each way
    encroach: float = 0.0            # 0..1: how far its regions push into neighbours
    structures: tuple = ()           # worldgen.structures rules: {"gen": name, ...}


_LIMITS = {
    "noise_scale_m": (10.0, 5000.0),
    "amplitude_m": (0.0, 200.0),
    "octaves": (1, 8),
    "persistence": (0.05, 0.95),
    "lacunarity": (1.5, 4.0),
    "base_height_m": (-100.0, 100.0),
    "steep_slope_deg": (3.0, 60.0),
    "detail_tile_m": (0.5, 100.0),
    "grid_spacing_m": (50.0, 2000.0),
    "jitter_m": (0.0, 500.0),
    "curviness_m": (0.0, 500.0),
    "width_m": (3.0, 30.0),
    "shoulder_m": (0.5, 50.0),
    "max_grade_pct": (0.5, 30.0),
    "terrace_steps": (2, 24),
    "terrace_ramp": (0.1, 0.9),
    "weight": (0.0, 100.0),
    "region_size": (1, 6),
    "encroach": (0.0, 1.0),
    "grid": (0.0, 1.0),
    "grid_heading_deg": (-360.0, 360.0),
    "loop_odds": (0.0, 1.0),
    "l_corner_odds": (0.0, 1.0),
    "corner_radius_m": (0.0, 500.0),
}
_OPTIONAL_RGB = {"markings"}
_OPTIONAL_NUMBER = {"grid_heading_deg", "corner_radius_m"}


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _value(where, key, raw, default):
    """``raw`` checked against the type of ``default`` and _LIMITS[key]."""
    if key in _OPTIONAL_RGB and (raw is None or default is None):
        return None if raw is None else _value(where, key, raw, (0.0, 0.0, 0.0))
    if key in _OPTIONAL_NUMBER and (raw is None or default is None):
        return None if raw is None else _value(where, key, raw, 0.0)
    if isinstance(default, bool):
        if not isinstance(raw, bool):
            raise ValueError(f"{where}.{key}: expected true or false, got {raw!r}")
        return raw
    if isinstance(default, str):
        if not isinstance(raw, str):
            raise ValueError(f"{where}.{key}: expected a string, got {raw!r}")
        return raw
    if isinstance(default, tuple):
        if not (isinstance(raw, list) and len(raw) == 3
                and all(_is_number(c) and 0.0 <= c <= 1.0 for c in raw)):
            raise ValueError(f"{where}.{key}: expected [r, g, b] in 0..1, got {raw!r}")
        return tuple(float(c) for c in raw)
    if isinstance(default, int):
        if not (_is_number(raw) and float(raw).is_integer()):
            raise ValueError(f"{where}.{key}: expected an integer, got {raw!r}")
        value = int(raw)
    else:
        if not _is_number(raw):
            raise ValueError(f"{where}.{key}: expected a number, got {raw!r}")
        value = float(raw)
    lo, hi = _LIMITS[key]
    if not lo <= value <= hi:
        raise ValueError(f"{where}.{key}: {value} is outside {lo}..{hi}")
    return value


def _section(cls, raw, where):
    """A ``cls`` from a JSON object: defaults for missing keys, errors for bad ones."""
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"{where}: expected an object, got {type(raw).__name__}")
    default = cls()
    unknown = sorted(set(raw) - {f.name for f in dataclasses.fields(cls)})
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {', '.join(unknown)}")
    values = {}
    for key, value in raw.items():
        d = getattr(default, key)
        values[key] = (_section(type(d), value, f"{where}.{key}") if dataclasses.is_dataclass(d)
                       else _value(where, key, value, d))
    return cls(**values)


def parse_biome(data, where="biome", default_name="biome", assets_dir=None):
    """A Biome from a decoded definition (a dict).  With ``assets_dir`` the
    detail texture must exist under it."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: expected a JSON object at the top level")
    name = data.get("name", default_name)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where}.name: expected a non-empty string, got {name!r}")
    terrain = _section(TerrainRules, data.get("terrain"), f"{where}.terrain")
    roads = _section(RoadRules, data.get("roads"), f"{where}.roads")
    if terrain.noise_type not in NOISE_TYPES:
        raise ValueError(f"{where}.terrain.noise_type: {terrain.noise_type!r} is not one of "
                         f"{', '.join(NOISE_TYPES)}")
    if roads.surface not in ROAD_SURFACES:
        raise ValueError(f"{where}.roads.surface: {roads.surface!r} is not one of "
                         f"{', '.join(ROAD_SURFACES)}")
    if roads.jitter_m > 0.2 * roads.grid_spacing_m:
        raise ValueError(f"{where}.roads.jitter_m: {roads.jitter_m} is more than a fifth "
                         f"of grid_spacing_m ({roads.grid_spacing_m})")
    if assets_dir is not None and not (Path(assets_dir) / terrain.detail_texture).is_file():
        raise ValueError(f"{where}.terrain.detail_texture: no file assets/"
                         f"{terrain.detail_texture}")
    weight = _value(where, "weight", data.get("weight", 1.0), 1.0)
    size = _value(where, "region_size", data.get("region_size", 1), 1)
    encroach = _value(where, "encroach", data.get("encroach", 0.0), 0.0)
    rules = data.get("structures", [])
    if not (isinstance(rules, list) and all(isinstance(r, dict) and isinstance(
            r.get("gen"), str) for r in rules)):
        raise ValueError(f"{where}.structures: expected a list of objects with a "
                         f"\"gen\" name, got {rules!r}")
    extra = {k: v for k, v in data.items()
             if k not in ("name", "terrain", "roads", "weight", "region_size", "encroach",
                          "structures")}
    return Biome(name=name, terrain=terrain, roads=roads, extra=extra, weight=weight,
                 region_size=size, encroach=encroach,
                 structures=tuple(dict(r) for r in rules))


def load_biome(name, definitions_dir, assets_dir=None):
    """Load ``<definitions_dir>/<name>.json`` (or a path to a .json)."""
    path = Path(name) if str(name).endswith(".json") else Path(definitions_dir) / f"{name}.json"
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{path.name}: empty definition (no terrain or roads rules yet)")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}: not valid JSON ({exc})") from exc
    return parse_biome(data, path.name, path.stem, assets_dir)


def load_biomes(names, definitions_dir, assets_dir=None):
    """The named biomes, in order."""
    return tuple(load_biome(n, definitions_dir, assets_dir) for n in names)

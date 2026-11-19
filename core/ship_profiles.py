"""Ship profile catalog: model, trail color, engine sound rate, scale."""

import config


def _wav(name):
    return (config.ASSETS / "sounds" / "spaceship" / name).as_posix()


SHIP_PROFILES = [
    {"id": "havana", "name": "Havana",
     "obj": config.HAVANA_MODEL_OBJ, "invert_vertical": config.CAR_MODEL_INVERT_VERTICAL,
     "trail": (0.90, 0.55, 0.20),
     "engine_rate": 1.00, "scale": config.HAVANA_SHIP_SCALE,
     "engine_seed": 505, "engine_wav": (config.ASSETS / "sounds" / "vehicle" / "havana_model0" / "engine.wav").as_posix()},
    {"id": "sparrow", "name": "Sparrow",
     "glb": config.SHIP_GLB, "trail": (0.35, 0.85, 1.00),
     "engine_rate": 1.00, "scale": config.SHIP_SCALE,
     "engine_seed": 101, "engine_wav": _wav("engine_1.wav")},
    {"id": "talon", "name": "Talon",
     "glb": config.SHIP_T2_GLB, "trail": (1.00, 0.45, 0.18),
     "engine_rate": 1.18, "scale": 1.00,
     "engine_seed": 202, "engine_wav": _wav("engine_2.wav")},
    {"id": "wraith", "name": "Wraith",
     "glb": config.SHIP_T3_GLB, "trail": (0.55, 1.00, 0.45),
     "engine_rate": 0.92, "scale": 1.10,
     "engine_seed": 303, "engine_wav": _wav("engine_3.wav")},
    {"id": "vortex", "name": "Vortex",
     "glb": config.SHIP_T4_GLB, "trail": (0.85, 0.30, 1.00),
     "engine_rate": 1.30, "scale": 1.05,
     "engine_seed": 404, "engine_wav": _wav("engine_4.wav")},
]

FREIGHTER_PROFILE = {"id": "freighter", "name": "Freighter",
    "glb": config.SHIP_F1_GLB, "trail": (0.70, 0.70, 0.70),
    "engine_rate": 0.60, "scale": 2.60, "freighter": True,
    "engine_seed": 999, "engine_wav": _wav("engine_freighter.wav")}

FLYABLE_IDS = [p["id"] for p in SHIP_PROFILES]


def get_profile(profile_id):
    for p in SHIP_PROFILES:
        if p["id"] == profile_id:
            return p
    if profile_id == FREIGHTER_PROFILE["id"]:
        return FREIGHTER_PROFILE
    return SHIP_PROFILES[0]


def random_flyable(rng):
    return rng.choice(SHIP_PROFILES)


def random_traffic_profile(rng, freighter=False):
    if freighter:
        return FREIGHTER_PROFILE
    return rng.choice(SHIP_PROFILES)

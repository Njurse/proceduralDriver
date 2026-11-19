"""Catalog of fictional minable materials and their visual traits."""


MATERIALS = [
    {"id": "verris", "name": "Verris", "tier": "common",
     "color": (0.42, 0.78, 0.38), "value": 4, "glow": 0.0},
    {"id": "korathite", "name": "Korathite", "tier": "common",
     "color": (0.78, 0.34, 0.18), "value": 5, "glow": 0.0},
    {"id": "ferrisite", "name": "Ferrisite", "tier": "common",
     "color": (0.55, 0.58, 0.62), "value": 3, "glow": 0.0},
    {"id": "solvium", "name": "Solvium", "tier": "uncommon",
     "color": (0.86, 0.78, 0.22), "value": 12, "glow": 0.15},
    {"id": "lumenite", "name": "Lumenite", "tier": "uncommon",
     "color": (0.22, 0.82, 0.74), "value": 14, "glow": 0.35},
    {"id": "threnite", "name": "Threnite", "tier": "uncommon",
     "color": (0.52, 0.22, 0.78), "value": 13, "glow": 0.25},
    {"id": "pylarite", "name": "Pylarite", "tier": "uncommon",
     "color": (0.20, 0.74, 0.92), "value": 15, "glow": 0.30},
    {"id": "aetherite", "name": "Aetherite", "tier": "rare",
     "color": (0.72, 0.86, 1.00), "value": 30, "glow": 0.55},
    {"id": "cinderglass", "name": "Cinderglass", "tier": "rare",
     "color": (0.92, 0.26, 0.14), "value": 28, "glow": 0.45},
    {"id": "noctis", "name": "Noctis", "tier": "rare",
     "color": (0.12, 0.08, 0.22), "value": 32, "glow": 0.40},
    {"id": "ourox", "name": "Ourox", "tier": "rare",
     "color": (0.96, 0.78, 0.20), "value": 34, "glow": 0.50},
    {"id": "quintessence", "name": "Quintessence", "tier": "legendary",
     "color": (0.95, 0.95, 1.00), "value": 80, "glow": 0.85},
]

_BY_ID = {m["id"]: m for m in MATERIALS}


def get(material_id):
    return _BY_ID.get(material_id, MATERIALS[0])


def pick(rng):
    """Weighted pick by tier."""
    roll = rng.random()
    if roll < 0.55:
        pool = [m for m in MATERIALS if m["tier"] == "common"]
    elif roll < 0.85:
        pool = [m for m in MATERIALS if m["tier"] == "uncommon"]
    elif roll < 0.97:
        pool = [m for m in MATERIALS if m["tier"] == "rare"]
    else:
        pool = [m for m in MATERIALS if m["tier"] == "legendary"]
    return rng.choice(pool)


def tier_weight(tier):
    return {"common": 0.55, "uncommon": 0.30, "rare": 0.12,
            "legendary": 0.03}.get(tier, 0.25)

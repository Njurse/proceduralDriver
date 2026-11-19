"""Versioned JSON save file with validated, crash-safe writes."""

import copy
import logging
import math
import time
from pathlib import Path

import config
from core.atomicfile import atomic_write_json, read_json

log = logging.getLogger("galaxy_explorer.save")

_MODES = ("galaxy", "system", "atmosphere")


def default_state():
    return {
        "version": config.SAVE_VERSION,
        "seed": None,
        "mode": "galaxy",
        "system_index": None,
        "planet_index": None,
        "ship": {
            "pos": [0.0, -config.GALAXY_START_DIST, 0.0],
            "hpr": [0.0, 0.0, 0.0],
            "throttle": 0.0,
        },
        "visited_systems": [],
        "discovered_planets": [],
        "play_time": 0.0,
        "saved_at": None,
        "galaxy_return": None,
        "settings": {
            "control_sensitivity": float(getattr(config, "CONTROL_SENSITIVITY", 1.0)),
            "mouse_sensitivity": float(getattr(config, "MOUSE_SENSITIVITY", 1.0)),
            "master_volume": float(getattr(config, "MASTER_VOLUME", 1.0)),
            "effects_volume": float(getattr(config, "EFFECTS_VOLUME", 1.0)),
            "invert_y": bool(getattr(config, "INVERT_Y", False)),
            "gamepad_deadzone": float(getattr(config, "GAMEPAD_DEADZONE", 0.16)),
        },
        "inventory": None,
        "fuel": None,
        "selected_ship": "sparrow",
    "on_foot": False,
    "foot": None,
    }


def _is_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _is_vec3(value):
    return (isinstance(value, (list, tuple)) and len(value) == 3
            and all(_is_number(v) for v in value))


def validate_state(data):
    """Return ``(valid, problems)`` for a decoded save payload."""
    problems = []
    if not isinstance(data, dict):
        return False, ["payload is not an object"]
    try:
        int(data.get("version", 1))
    except (TypeError, ValueError):
        problems.append("invalid version")
    mode = data.get("mode", "galaxy")
    if mode not in _MODES:
        problems.append(f"unknown mode {mode!r}")
    ship = data.get("ship")
    if not isinstance(ship, dict):
        problems.append("missing ship record")
    else:
        if not _is_vec3(ship.get("pos")):
            problems.append("invalid ship position")
        if not _is_vec3(ship.get("hpr")):
            problems.append("invalid ship orientation")
        if not _is_number(ship.get("throttle", 0.0)):
            problems.append("invalid throttle")
    for key in ("visited_systems", "discovered_planets"):
        if key in data and not isinstance(data[key], list):
            problems.append(f"{key} is not a list")
    for key in ("play_time",):
        if key in data and not _is_number(data[key]):
            problems.append(f"invalid {key}")
    for key in ("settings", "inventory", "fuel"):
        if key in data and data[key] is not None and not isinstance(data[key], dict):
            problems.append(f"invalid {key}")
    return not problems, problems


def _migrate(data):
    """Upgrade older save payloads to the current schema in place."""
    if not isinstance(data, dict):
        return data
    version = data.get("version", 1)
    try:
        version = int(version)
    except (TypeError, ValueError):
        version = 1

    settings = data.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    defaults = default_state()["settings"]
    for key, value in defaults.items():
        if key not in settings or settings[key] is None:
            settings[key] = copy.deepcopy(value)
    data["settings"] = settings
    if version < 2 and "inventory" not in data:
        data["inventory"] = None
    if version < 3 and "fuel" not in data:
        data["fuel"] = None
    if version < 4 and not data.get("selected_ship"):
        data["selected_ship"] = "sparrow"
    if version < 5:
        data.setdefault("on_foot", False)
        data.setdefault("foot", None)
    data["version"] = version if version > config.SAVE_VERSION else config.SAVE_VERSION
    return data


def _merge(default, data):
    """Recursively merge ``data`` onto ``default`` without losing new keys."""
    result = copy.deepcopy(default)
    for key, value in data.items():
        if key not in result:
            continue
        if isinstance(result[key], dict) and isinstance(value, dict):
            _merge(result[key], value)
        elif isinstance(result[key], list) and isinstance(value, list):
            result[key] = list(value)
        elif result[key] is None or isinstance(value, type(result[key])):
            result[key] = value
    return result


class SaveStore:
    """Loads and atomically writes the validated JSON save file."""

    def __init__(self, path=config.SAVE_PATH):
        self.path = Path(path)

    def load(self):
        state = default_state()
        data = read_json(self.path, backup=True)
        if data is None:
            return state
        valid, problems = validate_state(data)
        if not valid:
            log.warning("[save] ignoring invalid save (%s)", "; ".join(problems))
            return state
        return _merge(state, _migrate(data))

    def save(self, state):
        payload = dict(state)
        payload["version"] = config.SAVE_VERSION
        payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return atomic_write_json(self.path, payload, backup=True)

    def clear(self):
        for path in (self.path, self.path.with_suffix(self.path.suffix + ".bak")):
            try:
                path.unlink()
            except OSError:
                pass

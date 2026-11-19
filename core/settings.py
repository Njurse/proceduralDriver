"""Persistent player preferences, stored outside the game save.

These survive deleting ``save.json``: the save still mirrors these values for
backwards compatibility, but ``settings.json`` is the source of truth.
"""

import logging

import config
from core.atomicfile import atomic_write_json, read_json
from core.mathutil import clamp

log = logging.getLogger("galaxy_explorer.settings")


def default_settings():
    return {
        "control_sensitivity": float(
            getattr(config, "CONTROL_SENSITIVITY", 1.0)),
        "mouse_sensitivity": float(getattr(config, "MOUSE_SENSITIVITY", 1.0)),
        "master_volume": float(getattr(config, "MASTER_VOLUME", 1.0)),
        "effects_volume": float(getattr(config, "EFFECTS_VOLUME", 1.0)),
        "invert_y": bool(getattr(config, "INVERT_Y", False)),
        "gamepad_deadzone": float(
            getattr(config, "GAMEPAD_DEADZONE", 0.16)),
    }


def sanitize_settings(data):
    """Return a complete, in-range settings dict built from ``data``."""
    result = default_settings()
    if not isinstance(data, dict):
        return result
    for key in ("control_sensitivity", "mouse_sensitivity", "master_volume",
                "effects_volume", "gamepad_deadzone"):
        if key not in data:
            continue
        try:
            value = float(data[key])
        except (TypeError, ValueError):
            log.warning("[settings] ignoring invalid %s=%r", key, data[key])
            continue
        bounds = {
            "control_sensitivity": (0.1, 10.0),
            "mouse_sensitivity": (0.1, 5.0),
            "master_volume": (0.0, 1.0),
            "effects_volume": (0.0, 1.0),
            "gamepad_deadzone": (0.05, 0.4),
        }[key]
        result[key] = clamp(value, *bounds)
    if "invert_y" in data:
        result["invert_y"] = bool(data["invert_y"])
    return result


def load_settings(path=None, fallback=None):
    """Load prefs; if the file is missing, use ``fallback`` (e.g. save data)."""
    path = path or config.SETTINGS_PATH
    data = read_json(path, backup=True)
    if data is None:
        return sanitize_settings(fallback) if fallback else default_settings()
    return sanitize_settings(data)


def save_settings(data, path=None):
    path = path or config.SETTINGS_PATH
    return atomic_write_json(path, sanitize_settings(data), backup=True)

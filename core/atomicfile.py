"""Crash-safe JSON file writes shared by saves and settings.

Writes go to ``<path>.tmp`` first, are flushed to disk with ``os.fsync``, and
are then atomically renamed over the destination.  Optionally the previous
destination is retained as ``<path>.bak`` so a failed write never destroys the
last good file.
"""

import json
import logging
import os
from pathlib import Path

log = logging.getLogger("galaxy_explorer.atomicfile")


def atomic_write_json(path, payload, backup=False, indent=2):
    """Write ``payload`` to ``path`` safely. Returns True on success."""
    path = Path(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=indent, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        if backup and path.exists():
            try:
                path.replace(path.with_suffix(path.suffix + ".bak"))
            except OSError as exc:
                log.warning("[atomic] could not back up %s: %s", path, exc)
        os.replace(temporary, path)
        return True
    except (OSError, TypeError, ValueError) as exc:
        log.warning("[atomic] write failed for %s: %s", path, exc)
        return False


def read_json(path, backup=False):
    """Read JSON from ``path``; optionally fall back to its ``.bak`` file."""
    path = Path(path)
    candidates = [path]
    if backup:
        candidates.append(path.with_suffix(path.suffix + ".bak"))
    for candidate in candidates:
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return data
        except (OSError, ValueError) as exc:
            log.debug("[atomic] could not read %s: %s", candidate, exc)
    return None

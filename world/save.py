"""The world save file: saves/world.json.

Holds the world seed, the chunks the player has discovered, the biomes met
on the way and where the car was.  Discovered chunks are stored as runs per
chunk row, ``[cz, [[cx_first, count], ...]]``, which stays small for the
blobby areas a drive uncovers.  Writes are atomic (a temp file in the same
folder, then os.replace), so a crash mid-save leaves the old file intact.
"""

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import config

SAVE_VERSION = 1


@dataclass
class SaveState:
    seed: int
    discovered: set = field(default_factory=set)     # {(cx, cz)}
    biomes: set = field(default_factory=set)         # biome names met
    car: tuple | None = None                         # (x, z, heading)


def encode_chunks(chunks):
    """{(cx, cz)} -> [[cz, [[cx_first, count], ...]], ...], sorted."""
    rows = {}
    for cx, cz in chunks:
        rows.setdefault(int(cz), []).append(int(cx))
    out = []
    for cz in sorted(rows):
        runs = []
        for cx in sorted(rows[cz]):
            if runs and runs[-1][0] + runs[-1][1] == cx:
                runs[-1][1] += 1
            else:
                runs.append([cx, 1])
        out.append([cz, runs])
    return out


def decode_chunks(rows):
    return {(cx + k, cz) for cz, runs in rows for cx, n in runs for k in range(n)}


def to_json(state):
    car = None
    if state.car is not None:
        x, z, heading = state.car
        car = {"x": round(float(x), 3), "z": round(float(z), 3),
               "heading": round(float(heading), 5)}
    return {
        "version": SAVE_VERSION,
        "seed": int(state.seed),
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "chunk_size_m": config.TERRAIN_CHUNK_SIZE,
        "car": car,
        "discovered_biomes": sorted(state.biomes),
        "discovered": encode_chunks(state.discovered),
    }


def from_json(data):
    if not isinstance(data, dict) or data.get("version") != SAVE_VERSION:
        raise ValueError(f"unsupported save version {data.get('version')!r}"
                         if isinstance(data, dict) else "save is not a JSON object")
    car = data.get("car")
    return SaveState(
        seed=int(data["seed"]),
        discovered=decode_chunks(data.get("discovered", [])),
        biomes=set(data.get("discovered_biomes", [])),
        car=None if car is None else (float(car["x"]), float(car["z"]),
                                      float(car["heading"])))


def write_save(path, state):
    """Atomically write ``state`` to ``path``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(to_json(state), f, separators=(",", ":"))
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def read_save(path):
    """The SaveState at ``path``, or None when there is no file."""
    path = Path(path)
    if not path.is_file():
        return None
    return from_json(json.loads(path.read_text(encoding="utf-8")))


def free_path(path):
    """``path`` if nothing is there, else the first unused name beside it
    (world_2.json, world_3.json, ...), so nothing gets overwritten."""
    path = Path(path)
    candidate, k = path, 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{k}{path.suffix}")
        k += 1
    return candidate

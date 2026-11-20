"""Deterministic hashes (splitmix64): every random choice in worldgen.

``hash01`` takes plain integer keys (the scalar paths), ``hash01_arr`` a
seed, a salt and integer arrays.  Both give floats in [0, 1).
"""

import numpy as np

_MASK = (1 << 64) - 1


def _mix(h):
    h = ((h ^ (h >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    h = ((h ^ (h >> 27)) * 0x94D049BB133111EB) & _MASK
    return h ^ (h >> 31)


def hash01(*keys):
    """Deterministic float in [0, 1) from integer keys (splitmix64)."""
    h = 0x9E3779B97F4A7C15
    for k in keys:
        h = _mix((h + (k & _MASK)) & _MASK)
    return (h >> 11) / float(1 << 53)


def _mix_arr(h):
    """splitmix64 finaliser on uint64 arrays (wraps mod 2^64)."""
    with np.errstate(over="ignore"):
        h = (h ^ (h >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return h ^ (h >> np.uint64(31))


def hash01_arr(seed, salt, i, j):
    """Deterministic floats in [0, 1) for integer arrays i, j."""
    with np.errstate(over="ignore"):
        h = _mix_arr(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(salt))
        h = _mix_arr(h + np.asarray(i).astype(np.int64).astype(np.uint64))
        h = _mix_arr(h + np.asarray(j).astype(np.int64).astype(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)

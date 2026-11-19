"""Vectorised value-noise / fBm used for terrain, textures and transitions."""

import numpy as np

_MASK = np.int64(0x7FFFFFFF)
_PRIME_X = np.int64(374761393)
_PRIME_Y = np.int64(668265263)
_PRIME_S = np.int64(2147483647)


def _hash01(ix, iy, seed):
    """Deterministic integer hash -> float in [0, 1]."""
    n = (ix * _PRIME_X + iy * _PRIME_Y + np.int64(seed) * _PRIME_S) & _MASK
    n = ((n ^ (n >> 13)) * np.int64(1274126177)) & _MASK
    n = (n ^ (n >> 16)) & _MASK
    return (n & np.int64(0xFFFFFF)).astype(np.float64) / float(0xFFFFFF)


def value_noise_2d(x, y, seed=0):
    """Smooth 2D value noise. x/y may be scalars or numpy arrays."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = x - ix
    fy = y - iy
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)

    n00 = _hash01(ix, iy, seed)
    n10 = _hash01(ix + 1, iy, seed)
    n01 = _hash01(ix, iy + 1, seed)
    n11 = _hash01(ix + 1, iy + 1, seed)

    nx0 = n00 * (1.0 - ux) + n10 * ux
    nx1 = n01 * (1.0 - ux) + n11 * ux
    return nx0 * (1.0 - uy) + nx1 * uy


def fbm_2d(x, y, seed=0, octaves=5, persistence=0.5, lacunarity=2.0,
           ridged=False):
    """Fractal Brownian motion built on value noise, normalised to [0, 1]."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    total = np.zeros(np.broadcast(x, y).shape, dtype=np.float64)
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for octave in range(octaves):
        n = value_noise_2d(
            x * frequency + octave * 19.19,
            y * frequency - octave * 7.73,
            seed + octave * 101,
        )
        if ridged:
            n = 1.0 - np.abs(2.0 * n - 1.0)
        total += n * amplitude
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    return total / max(norm, 1e-9)


def terrain_height(x, y, seed, params):
    """Height in world units for a planet's terrain.

    `params` is a mapping with scale, amplitude, octaves, persistence and
    ridge. Works on scalars and numpy arrays alike.
    """
    scale = float(params.get("scale", 0.0016))
    amplitude = float(params.get("amplitude", 210.0))
    octaves = int(params.get("octaves", 5))
    persistence = float(params.get("persistence", 0.5))
    ridge = float(params.get("ridge", 0.45))

    base = fbm_2d(x * scale, y * scale, seed, octaves, persistence)
    ridged = fbm_2d(x * scale * 0.62 + 80.0, y * scale * 0.62 - 40.0,
                    seed + 911, octaves, persistence, ridged=True)
    continents = fbm_2d(x * scale * 0.23, y * scale * 0.23, seed + 37, 3, 0.55)

    n = base * (1.0 - ridge) + ridged * ridge
    n = n * 0.72 + continents * 0.28
    return (n - 0.5) * amplitude

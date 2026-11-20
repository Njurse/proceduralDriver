"""Vertex colours: a ramp by height (fraction of the amplitude), blended to
the steep colour on slopes."""

import math

import numpy as np

# Vertex colours by height (fraction of amplitude) and steepness.
_LOW = np.array([0.14, 0.28, 0.10])    # valley floor
_MID = np.array([0.32, 0.47, 0.19])    # grass
_HIGH = np.array([0.66, 0.60, 0.33])   # dry hilltops
_STEEP = np.array([0.48, 0.38, 0.25])  # bare earth on the steeper slopes


def terrain_colors(height, normal, amplitude, ramp=None, steep_deg=11.0):
    """RGBA per vertex from height (m above the base level) and unit normals
    (..., 3).  ``ramp`` has low/mid/high/steep RGB (worldgen.biomes.ColorRamp);
    slopes blend to ``steep`` over steep_deg +- 2.2 degrees."""
    low, mid, high, steep_rgb = ((_LOW, _MID, _HIGH, _STEEP) if ramp is None else
                                 (np.asarray(c) for c in (ramp.low, ramp.mid, ramp.high,
                                                          ramp.steep)))
    t = np.clip(np.asarray(height) / (0.5 * max(amplitude, 1e-6)), -1.0, 1.0)[..., None]
    lo = mid + (low - mid) * np.clip(-t, 0.0, 1.0)
    rgb = lo + (high - mid) * np.clip(t, 0.0, 1.0)
    c0, c1 = (math.cos(math.radians(steep_deg + d)) for d in (-2.2, 2.2))
    steep = np.clip((c0 - normal[..., 1:2]) / (c0 - c1), 0.0, 1.0)
    rgb = rgb + (steep_rgb - rgb) * steep
    alpha = np.ones(rgb.shape[:-1] + (1,))
    return np.concatenate([rgb, alpha], axis=-1)

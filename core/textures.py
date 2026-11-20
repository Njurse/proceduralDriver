"""Procedural texture factories (numpy arrays -> Panda3D textures).

Panda3D uploads ``F_rgba8`` ram images in BGR(A) byte order and expects rows
bottom-up, so :func:`_to_texture` swizzles and flips every array once.
"""

import numpy as np
from panda3d.core import Texture

from core.noise import fbm_2d


def _normalise_rgba(rgba):
    arr = np.asarray(rgba)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=np.uint8)
        arr = np.concatenate([arr, alpha], axis=2)
    return arr


def upload_rgba(texture, rgba):
    """Upload an ``(h, w, 3|4)`` array into an existing texture."""
    arr = _normalise_rgba(rgba)
    # Panda3D wants BGR(A) byte order and bottom-up rows.
    data = arr[..., [2, 1, 0, 3]]
    texture.setRamImage(np.ascontiguousarray(np.flipud(data)).tobytes())


def _to_texture(rgba, name, repeat=False, linear=True):
    """Create a Panda3D texture from an ``(h, w, 3|4)`` uint8 array."""
    arr = _normalise_rgba(rgba)
    height, width = arr.shape[:2]
    texture = Texture(name)
    texture.setup2dTexture(width, height, Texture.T_unsigned_byte,
                           Texture.F_rgba8)
    upload_rgba(texture, arr)
    texture.setMagfilter(Texture.FT_linear if linear else Texture.FT_nearest)
    texture.setMinfilter(Texture.FT_linear if linear else Texture.FT_nearest)
    if repeat:
        texture.setWrapU(Texture.WMRepeat)
        texture.setWrapV(Texture.WMRepeat)
    return texture


def seamless_fbm(width, height, frequency, seed, octaves=5):
    """fBm that wraps horizontally by cross-fading two offset samples."""
    u = np.linspace(0.0, 1.0, width, endpoint=False)
    v = np.linspace(0.0, 1.0, height, endpoint=False)
    grid_u, grid_v = np.meshgrid(u, v)
    left = fbm_2d(grid_u * frequency, grid_v * frequency, seed, octaves)
    right = fbm_2d((grid_u - 1.0) * frequency, grid_v * frequency,
                   seed, octaves)
    return left * (1.0 - grid_u) + right * grid_u


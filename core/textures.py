"""Procedural texture factories (numpy arrays -> Panda3D textures).

Panda3D uploads ``F_rgba8`` ram images in BGR(A) byte order and expects rows
bottom-up, so :func:`_to_texture` swizzles and flips every array once.
"""

import numpy as np
from panda3d.core import Texture

from core.noise import fbm_2d

_STOPS = (0.00, 0.30, 0.44, 0.60, 0.78, 0.92)


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


def _ramp(value, stops, colors):
    out = np.zeros(value.shape + (3,), dtype=np.float64)
    for channel in range(3):
        channel_stops = [color[channel] for color in colors]
        out[..., channel] = np.interp(value, stops, channel_stops)
    return out


def planet_texture(width, height, palette, seed, frozen_caps=0.6):
    """Rocky planet surface map tinted by an archetype palette."""
    elevation = seamless_fbm(width, height, 5.0, seed, 5)
    detail = seamless_fbm(width, height, 17.0, seed + 71, 3)
    continents = seamless_fbm(width, height, 2.2, seed + 13, 3)
    elevated = elevation * 0.62 + detail * 0.18 + continents * 0.20

    colors = [palette["deep"], palette["water"], palette["low"],
              palette["mid"], palette["high"], palette["peak"]]
    rgb = _ramp(np.clip(elevated, 0.0, 1.0), _STOPS, colors)

    # Darken steep/detailed areas slightly for extra surface variation.
    rgb *= (0.86 + 0.28 * detail)[..., None]

    # Polar caps fade in toward the poles.
    v = np.linspace(0.0, 1.0, height)[:, None]
    latitude = np.abs(v * 2.0 - 1.0)
    cap = np.clip((latitude - (1.0 - frozen_caps)) / max(frozen_caps, 1e-3),
                  0.0, 1.0)
    cap = cap * cap * (3.0 - 2.0 * cap)
    snow = np.asarray(palette["peak"], dtype=np.float64)
    rgb = rgb * (1.0 - cap[..., None]) + snow * cap[..., None]

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb * 255.0, 0, 255)
    rgba[..., 3] = 255
    return _to_texture(rgba, f"planet_{seed}", repeat=False)


def cloud_texture(width, height, seed, coverage, color):
    """Soft alpha cloud map; use with a transparent sphere shell."""
    noise = seamless_fbm(width, height, 9.0, seed + 991, 5)
    curve = seamless_fbm(width, height, 3.0, seed + 331, 3)
    pattern = noise * 0.68 + curve * 0.32
    threshold = 1.0 - coverage
    alpha = np.clip((pattern - threshold) / 0.22, 0.0, 1.0)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    alpha *= 0.92

    rgba = np.empty((height, width, 4), dtype=np.uint8)
    shade = 0.82 + 0.18 * noise
    rgba[..., 0] = np.clip(color[0] * shade * 255.0, 0, 255)
    rgba[..., 1] = np.clip(color[1] * shade * 255.0, 0, 255)
    rgba[..., 2] = np.clip(color[2] * shade * 255.0, 0, 255)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255)
    return _to_texture(rgba, f"cloud_{seed}", repeat=True)


def soft_dot_texture(size=64, color=(1.0, 1.0, 1.0), falloff=2.4):
    """Radial alpha sprite used for map markers and glows."""
    axis = np.linspace(-1.0, 1.0, size)
    x, y = np.meshgrid(axis, axis)
    radius = np.clip(np.sqrt(x * x + y * y), 0.0, 1.0)
    alpha = np.power(1.0 - radius, falloff)
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., 0] = int(color[0] * 255)
    rgba[..., 1] = int(color[1] * 255)
    rgba[..., 2] = int(color[2] * 255)
    rgba[..., 3] = np.clip(alpha * 255.0, 0, 255)
    return _to_texture(rgba, "soft_dot", repeat=False)

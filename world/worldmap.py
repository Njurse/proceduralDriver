"""Top-down map images of the world: the HUD minimap and the PNG export.

Both draw from ``chunk_tile``: one chunk sampled on a px x px grid of the
same World the car drives on, painted with its biome colours, hillshaded
from the north-west, with the roads laid on top.  Images are north-up: +z
(heading 0) is the top row and +x is to the right.
"""

import math
from datetime import datetime
from pathlib import Path

import numpy as np

import config

UNDISCOVERED = np.array([16, 18, 24], np.uint8)
_FLAT_SEEN = np.array([70, 74, 70], np.uint8)   # discovered, tile not built yet
_LIGHT = np.array([-1.0, 1.4, 1.0]) / math.sqrt(1.0 + 1.96 + 1.0)  # from the NW, above


def chunk_tile(world, ci, ck, px, size=None):
    """(rgb uint8 (px, px, 3), road mask (px, px)) for chunk (ci, ck); row 0
    is the chunk's north (max z) edge."""
    size = float(config.TERRAIN_CHUNK_SIZE if size is None else size)
    f = (np.arange(px) + 0.5) / px
    gx, gz = np.meshgrid((ci + f) * size, (ck + 1.0 - f) * size)
    h, hx, hz, on_road = world.query(gx, gz)
    n = np.stack([-hx, np.ones_like(h), -hz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    rgb = world.colors(gx, gz, h, n)[..., :3]
    shade = np.clip(1.0 + 2.5 * (n @ _LIGHT - _LIGHT[1]), 0.55, 1.35)[..., None]
    rgb = rgb * shade
    w = world.biome_weight(gx, gz)[..., None]
    c0, c1 = (np.array(b.roads.color) for b in world.biomes)
    road = np.where(w >= 0.5, c1, c0) * 0.85
    rgb = np.where(on_road[..., None], road, rgb)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), on_road


def downsample(tile, road, k):
    """Shrink a tile by k: block means, with any road in a block kept."""
    if k == 1:
        return tile
    p = tile.shape[0] // k
    rgb = tile.reshape(p, k, p, k, 3).mean(axis=(1, 3))
    any_road = road.reshape(p, k, p, k).any(axis=(1, 3))
    rmin = np.where(road[..., None], tile, 255).reshape(p, k, p, k, 3).min(axis=(1, 3))
    return np.where(any_road[..., None], rmin, rgb).astype(np.uint8)


class TileCache:
    """chunk_tile results per chunk for one world, built a few per call."""

    def __init__(self, world, px):
        self.world, self.px = world, px
        self.tiles = {}
        self._views = {}

    def get(self, key):
        return self.tiles.get(key)

    def view(self, key, px_per_chunk, dim, grid):
        """The tile at px_per_chunk, dimmed (discovered, not loaded) or not,
        with its chunk border darkened when ``grid``; cached."""
        vkey = (key, px_per_chunk, dim, grid)
        out = self._views.get(vkey)
        if out is None:
            hit = self.tiles.get(key)
            if hit is None:
                return _FLAT_SEEN
            out = downsample(*hit, self.px // px_per_chunk).astype(float)
            if grid:
                out[0, :] *= 0.7
                out[:, 0] *= 0.7
            if dim:
                out = out * 0.45 + UNDISCOVERED * 0.55
            out = self._views[vkey] = out.astype(np.uint8)
        return out

    def build(self, keys, budget):
        """Build up to ``budget`` of ``keys`` not cached yet, in order."""
        built = 0
        for key in keys:
            if built >= budget:
                break
            if key not in self.tiles:
                self.tiles[key] = chunk_tile(self.world, *key, self.px)
                built += 1
        return built


def minimap_image(cache, center, discovered, loaded, px_per_chunk, out_px, grid=True):
    """North-up RGB image (out_px, out_px, 3) centred on ``center`` (x, z).

    Loaded chunks are drawn at full brightness, discovered ones dimmed,
    undiscovered ones dark; chunks whose tile isn't cached yet show flat.
    """
    size = config.TERRAIN_CHUNK_SIZE
    mpp = size / px_per_chunk
    u = (np.arange(out_px) + 0.5 - out_px / 2) * mpp
    cx = np.floor((center[0] + u) / size).astype(int)               # per column
    cz = np.floor((center[1] - u) / size).astype(int)               # per row
    ix = np.floor(((center[0] + u) / size - cx) * px_per_chunk).astype(int)
    iz = np.floor((cz + 1 - (center[1] - u) / size) * px_per_chunk).astype(int)
    ix, iz = np.clip(ix, 0, px_per_chunk - 1), np.clip(iz, 0, px_per_chunk - 1)
    xs, zs = np.unique(cx), np.unique(cz)
    grid = grid and px_per_chunk >= 8   # chunk borders only when there's room
    grid_img = np.empty((len(zs), len(xs), px_per_chunk, px_per_chunk, 3), np.uint8)
    grid_img[:] = UNDISCOVERED
    for a, zk in enumerate(zs.tolist()):
        for b, xk in enumerate(xs.tolist()):
            key = (xk, zk)
            seen = key in loaded
            if seen or key in discovered:
                grid_img[a, b] = cache.view(key, px_per_chunk, not seen, grid)
    za = np.searchsorted(zs, cz)
    xb = np.searchsorted(xs, cx)
    return grid_img[za[:, None], xb[None, :], iz[:, None], ix[None, :]]


def export_world(world, discovered, path=None, m_per_px=None, max_px=16384):
    """Write every discovered chunk to one north-up PNG; returns (path, (w, h)).

    The image is filled a chunk at a time (one tile array each), so memory
    is the output image plus one tile.  When the area would pass ``max_px``
    on a side the scale is halved until it fits.
    """
    if not discovered:
        raise ValueError("nothing discovered yet")
    size = config.TERRAIN_CHUNK_SIZE
    px = max(1, round(size / float(config.EXPORT_M_PER_PX if m_per_px is None else m_per_px)))
    xs = [c[0] for c in discovered]
    zs = [c[1] for c in discovered]
    x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
    while px > 1 and max(x1 - x0 + 1, z1 - z0 + 1) * px > max_px:
        px //= 2
    img = np.empty(((z1 - z0 + 1) * px, (x1 - x0 + 1) * px, 3), np.uint8)
    img[:] = UNDISCOVERED
    for cx, cz in discovered:
        r, c = (z1 - cz) * px, (cx - x0) * px
        img[r:r + px, c:c + px] = chunk_tile(world, cx, cz, px)[0]
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(config.EXPORT_DIR) / f"world_{world.regions.seed}_{stamp}.png"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_png(path, img)
    return path, (img.shape[1], img.shape[0])


def write_png(path, rgb):
    """Save an (h, w, 3) uint8 array: pillow when present, else Panda3D."""
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        Image.fromarray(rgb, "RGB").save(path)
        return
    from panda3d.core import Filename, PNMImage
    h, w = rgb.shape[:2]
    img = PNMImage(w, h, 3)
    for y in range(h):
        for x in range(w):
            img.setXelVal(x, y, *(int(v) for v in rgb[y, x]))
    if not img.write(Filename.fromOsSpecific(str(path))):
        raise OSError(f"could not write {path}")

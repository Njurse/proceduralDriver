"""Full-screen rainbow-spiral / fBm dissolve used for world transitions."""

import numpy as np
from panda3d.core import (
    CardMaker,
    Texture,
    TransparencyAttrib,
    Vec4,
)

import config
from core.noise import fbm_2d
from core.textures import upload_rgba

_TAU = 2.0 * np.pi


def _smoothstep(edge0, edge1, value):
    span = max(edge1 - edge0, 1e-6)
    t = np.clip((value - edge0) / span, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Transition:
    """A short, fully opaque at the midpoint, procedural screen effect."""

    def __init__(self, base):
        self.base = base
        self.active = False
        self.progress = 0.0
        self.time = 0.0
        self.on_swap = None
        self._swapped = False
        self.width = config.TRANSITION_WIDTH
        self.height = config.TRANSITION_HEIGHT

        xs = np.linspace(-1.0, 1.0, self.width)
        ys = np.linspace(-1.0, 1.0, self.height) * (self.height / self.width)
        self._x, self._y = np.meshgrid(xs, ys)
        self._radius = np.hypot(self._x, self._y)
        self._angle = np.arctan2(self._y, self._x)

        self.texture = Texture("transition")
        self.texture.setup2dTexture(
            self.width, self.height, Texture.T_unsigned_byte,
            Texture.F_rgba8)
        self.texture.setMagfilter(Texture.FT_linear)
        self.texture.setMinfilter(Texture.FT_linear)

        maker = CardMaker("transition_card")
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        self.card = base.render2d.attachNewNode(maker.generate())
        self.card.setTexture(self.texture)
        self.card.setTransparency(TransparencyAttrib.MAlpha)
        self.card.setColor(Vec4(1.0, 1.0, 1.0, 1.0))
        self.card.setDepthTest(False)
        self.card.setBin("fixed", 200)
        self.card.hide()

    def play(self, on_swap=None):
        self.on_swap = on_swap
        self._swapped = False
        self.active = True
        self.time = 0.0
        self.progress = 0.0
        self.card.show()
        self._render(0.0)

    def stop(self):
        self.active = False
        self.card.hide()

    def update(self, dt):
        if not self.active:
            return
        self.time += dt
        progress = min(1.0, self.time / max(config.TRANSITION_SECONDS, 1e-3))
        self.progress = progress
        self._render(progress)

        if not self._swapped and progress >= 0.5:
            self._swapped = True
            if self.on_swap is not None:
                callback, self.on_swap = self.on_swap, None
                callback()
        if progress >= 1.0:
            self.active = False
            self.card.hide()

    # -- frame synthesis ----------------------------------------------------
    def _render(self, progress):
        radius = self._radius
        angle = self._angle

        spiral = 0.5 + 0.5 * np.sin(angle * 3.0 + radius * 19.0
                                    - progress * 26.0)
        hue = np.mod(angle / _TAU + radius * 0.55 + progress * 0.42, 1.0)
        phase = hue * _TAU
        red = 0.5 + 0.5 * np.cos(phase)
        green = 0.5 + 0.5 * np.cos(phase + _TAU / 3.0)
        blue = 0.5 + 0.5 * np.cos(phase + 2.0 * _TAU / 3.0)

        noise = fbm_2d(self._x * 2.6 - progress * 3.2,
                       self._y * 2.6 + progress * 2.5, 1234, 4)
        muscle = 0.42 + 0.72 * noise + 0.28 * spiral
        rgb = np.stack([red, green, blue], axis=-1) * muscle[..., None]

        core = np.exp(-radius * 2.3) * (0.45 + 0.55 * progress)
        rgb += core[..., None] * np.array([1.0, 0.96, 0.84])

        alpha = (_smoothstep(0.0, 0.20, progress)
                 * (1.0 - _smoothstep(0.80, 1.0, progress)))
        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(rgb * 255.0, 0, 255)
        rgba[..., 3] = np.clip(alpha * 255.0, 0, 255)
        upload_rgba(self.texture, rgba)

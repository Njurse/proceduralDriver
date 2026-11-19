"""Visual overlay for the pulse engine: countdown, vignette, speed lines, flash."""

import logging
import math

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, LineSegs, TextNode, TransparencyAttrib

import config

log = logging.getLogger("galaxy_explorer.pulse_fx")


class PulseFX:
    def __init__(self, base):
        self.base = base
        self.root = base.aspect2d.attachNewNode("pulse_fx_root")
        self.active = False
        self.countdown = 0.0
        self.flash = 0.0
        self.fov_rebound = 0.0
        self._phase = 0.0
        self._line_count = 32
        self._half_width = 1.0

        # Countdown text.
        self.countdown_text = OnscreenText(
            text="", parent=self.root, pos=(0.0, 0.0), scale=0.22,
            fg=(1.0, 0.95, 0.7, 1.0), align=TextNode.ACenter,
            shadow=(0, 0, 0, 0.8), mayChange=True)

        # Vignette (dark edges) during pulse.
        maker = CardMaker("pulse_vignette")
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        self.vignette = self.root.attachNewNode(maker.generate())
        self.vignette.setColor(0.05, 0.10, 0.25, 0.0)
        self.vignette.setTransparency(TransparencyAttrib.MAlpha)
        self.vignette.setDepthTest(False)
        self.vignette.setBin("background", 60)

        # Speed lines.
        self.lines = LineSegs("pulse_lines")
        self.lines.setThickness(2.0)
        self.lines_node = self.root.attachNewNode(self.lines.create())
        self.lines_node.setDepthTest(False)
        self.lines_node.setBin("fixed", 58)
        self.lines_node.hide()

        # Additive white flash on pullout.
        maker = CardMaker("pulse_flash")
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        self.flash_card = self.root.attachNewNode(maker.generate())
        self.flash_card.setColor(1.0, 1.0, 1.0, 0.0)
        # additive blend
        from panda3d.core import ColorBlendAttrib
        self.flash_card.setAttrib(ColorBlendAttrib.make(
            ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
            ColorBlendAttrib.OOne))
        self.flash_card.setTransparency(TransparencyAttrib.MAlpha)
        self.flash_card.setDepthTest(False)
        self.flash_card.setBin("fixed", 90)

        self.root.hide()

    def layout(self, aspect):
        """Stretch full-screen pulse effects to cover ultrawide windows."""
        half_width = max(float(aspect), 0.5)
        self._half_width = half_width
        self.vignette.setScale(half_width, 1.0, 1.0)
        self.flash_card.setScale(half_width, 1.0, 1.0)
        self.lines_node.setScale(half_width, 1.0, 1.0)

    def set_active(self, active):
        self.active = active
        if active:
            self.root.show()
            # No darkening vignette; keep a faint additive tint only.
            self.vignette.setColor(0.05, 0.10, 0.25, 0.0)
            self.lines_node.show()
        else:
            self.vignette.setColor(0.05, 0.10, 0.25, 0.0)
            self.lines_node.hide()
            if not self.countdown and self.flash <= 0:
                self.root.hide()

    def set_countdown(self, value):
        self.countdown = value
        if value and value > 0:
            self.root.show()
            self.countdown_text.setText(str(int(math.ceil(value))))
            self.countdown_text.show()
        else:
            self.countdown_text.hide()
            if not self.active and self.flash <= 0:
                self.root.hide()

    def trigger_pullout(self):
        self.flash = config.PULSE_PULLOUT_FLASH
        self.fov_rebound = config.PULSE_PULLOUT_FLASH + 0.35
        self.root.show()
        self.flash_card.setColor(1.0, 1.0, 1.0, 0.85)

    def update(self, dt):
        # Animate pullout flash.
        if self.flash > 0:
            self.flash = max(0.0, self.flash - dt)
            alpha = self.flash / max(config.PULSE_PULLOUT_FLASH, 1e-3)
            self.flash_card.setColor(1.0, 1.0, 1.0, 0.85 * alpha)
            if self.flash <= 0:
                self.flash_card.setColor(1.0, 1.0, 1.0, 0.0)

        # Animate speed lines when active.
        if self.active:
            self._phase += dt * 6.0
            segs = LineSegs("pulse_lines")
            segs.setThickness(2.0)
            for i in range(self._line_count):
                ang = (i / self._line_count) * math.tau + self._phase * 0.3
                r1 = 0.05 + 0.1 * math.sin(self._phase + i)
                r2 = 1.0
                x1 = math.cos(ang) * r1
                y1 = math.sin(ang) * r1
                x2 = math.cos(ang) * r2
                y2 = math.sin(ang) * r2
                segs.moveTo(x1, 0.0, y1)
                segs.drawTo(x2, 0.0, y2)
            try:
                self.lines_node.removeNode()
            except Exception as exc:
                log.debug("[pulse_fx] old lines removal failed: %s", exc)
            self.lines_node = self.root.attachNewNode(segs.create())
            self.lines_node.setDepthTest(False)
            self.lines_node.setBin("fixed", 58)
            self.lines_node.setScale(self._half_width, 1.0, 1.0)
            # additive blend for speed lines
            from panda3d.core import ColorBlendAttrib, TransparencyAttrib
            self.lines_node.setAttrib(ColorBlendAttrib.make(
                ColorBlendAttrib.MAdd, ColorBlendAttrib.OIncomingAlpha,
                ColorBlendAttrib.OOne))
            self.lines_node.setTransparency(TransparencyAttrib.MAlpha)
        else:
            try:
                self.lines_node.hide()
            except Exception as exc:
                log.debug("[pulse_fx] lines hide failed: %s", exc)

        if not self.active and self.countdown <= 0 and self.flash <= 0:
            self.root.hide()

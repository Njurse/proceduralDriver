"""On-screen telemetry, prompts, message banner and crosshair."""

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    CardMaker,
    LineSegs,
    TextNode,
    TransparencyAttrib,
    Vec4,
)

import config
from core.mathutil import clamp

_HUD = (0.78, 0.90, 0.98, 1.0)
_DIM = (0.48, 0.60, 0.72, 1.0)
_ACCENT = (0.55, 0.95, 1.00, 1.0)


class HUD:
    def __init__(self, base):
        self.base = base
        self.root = base.aspect2d.attachNewNode("hud")
        self._message_left = 0.0
        self._message_total = 1.0
        self._message_color = (1.0, 0.92, 0.62)

        def text(pos, scale, align=TextNode.ALeft, color=_HUD, parent=None):
            return OnscreenText(
                text="", parent=parent or self.root, pos=pos, scale=scale,
                fg=color, align=align, shadow=(0, 0, 0, 0.75),
                mayChange=True)

        self.title = text((-1.72, 0.90), 0.058)
        self.subtitle = text((-1.72, 0.825), 0.036, color=_DIM)
        self.controls = text((1.72, 0.90), 0.030, align=TextNode.ARight,
                             color=_DIM)
        self.telemetry = text((-1.72, -0.50), 0.040)
        self.prompt = text((0.0, -0.70), 0.056, align=TextNode.ACenter,
                           color=_ACCENT)
        self.message = text((0.0, 0.70), 0.062, align=TextNode.ACenter,
                            color=(1.0, 0.92, 0.62, 1.0))
        self.message.setFg(Vec4(1.0, 0.92, 0.62, 0.0))

        self.controls.setText(
            "W/S throttle - A/D roll\n"
            "Arrows/mouse look - Shift pulse\n"
            "Space brake - X/F land - R mine\n"
            "E interact - I inventory - M map - P pause")
        self._bars = []
        self._bar_width = 0.46
        self._build_gauges()
        self._build_crosshair()
        self._build_help()
        self.set_prompt("")
        self.layout(base.getAspectRatio())

    # -- layout -------------------------------------------------------------
    def layout(self, aspect):
        """Reflow the whole HUD inside the window at any aspect ratio.

        ``aspect2d`` spans ``[-aspect, aspect]`` horizontally and ``[-1, 1]``
        vertically, so every position is derived from those bounds instead of
        hard-coded 16:9 offsets.  This keeps the HUD on screen for 4:3,
        16:10, 16:9 and ultrawide displays alike.
        """
        half_width = max(float(aspect), 0.5)
        margin = clamp(half_width * 0.035, 0.06, 0.16)
        left = -half_width + margin
        right = half_width - margin

        self.title.setPos(left, 0.90)
        self.subtitle.setPos(left, 0.832)
        self.controls.setPos(right, 0.88)
        self.telemetry.setPos(left, -0.50)
        self.prompt.setPos(0.0, -0.70)
        self.message.setPos(0.0, 0.72)

        bar_width = clamp((right - left) * 0.30, 0.20, 0.46)
        self._bar_width = bar_width
        label_x = left + bar_width + 0.055
        for background, fill, y in self._bars:
            background.setPos(left, 0.0, y)
            background.setScale(bar_width, 1.0, 0.014)
            fill.setPos(left, 0.0, y)
            fill.setScale(min(fill.getSx(), bar_width), 1.0, 0.010)
        self.speed_label.setPos(label_x, -0.742)
        self.throttle_label.setPos(label_x, -0.797)
        self.mining_label.setPos(label_x, -0.852)
        self.fuel_label.setPos(label_x, -0.907)
        self.pulse_label.setPos(label_x, -0.962)

        self.help_dim.setScale(max(1.0, half_width / 2.0), 1.0, 1.0)
        self.help_text.setPos(-min(half_width - 0.25, 1.0), 0.42)

    # -- construction -------------------------------------------------------
    def _bar(self, y, width=0.46):
        maker = CardMaker("bar")
        maker.setFrame(0.0, 1.0, 0.0, 1.0)
        background = self.root.attachNewNode(maker.generate())
        background.setPos(-1.72, 0.0, y)
        background.setScale(width, 1.0, 0.014)
        background.setColor(0.03, 0.05, 0.08, 0.55)
        background.setTransparency(TransparencyAttrib.MAlpha)
        background.setDepthTest(False)
        background.setBin("fixed", 55)
        fill = self.root.attachNewNode(maker.generate())
        fill.setPos(-1.72, 0.0, y)
        fill.setScale(0.001, 1.0, 0.010)
        fill.setColor(0.36, 0.86, 1.0, 0.9)
        fill.setTransparency(TransparencyAttrib.MAlpha)
        fill.setDepthTest(False)
        fill.setBin("fixed", 56)
        self._bars.append((background, fill, y))
        return fill

    def _build_gauges(self):
        self.speed_fill = self._bar(-0.735)
        self.throttle_fill = self._bar(-0.790)
        self.mining_fill = self._bar(-0.845)
        self.fuel_fill = self._bar(-0.900)
        self.pulse_fill = self._bar(-0.955)
        self.speed_label = OnscreenText(
            text="SPD", parent=self.root, pos=(-1.22, -0.742), scale=0.028,
            fg=_DIM, align=TextNode.ALeft, mayChange=True)
        self.throttle_label = OnscreenText(
            text="THR", parent=self.root, pos=(-1.22, -0.797), scale=0.028,
            fg=_DIM, align=TextNode.ALeft, mayChange=True)
        self.mining_label = OnscreenText(
            text="MINE", parent=self.root, pos=(-1.22, -0.852), scale=0.028,
            fg=_DIM, align=TextNode.ALeft, mayChange=True)
        self.fuel_label = OnscreenText(
            text="FUEL", parent=self.root, pos=(-1.22, -0.907), scale=0.028,
            fg=_DIM, align=TextNode.ALeft, mayChange=True)
        self.pulse_label = OnscreenText(
            text="PULSE", parent=self.root, pos=(-1.22, -0.962), scale=0.028,
            fg=_DIM, align=TextNode.ALeft, mayChange=True)

    def _build_crosshair(self):
        lines = LineSegs("crosshair")
        lines.setThickness(2.0)
        lines.setColor(0.70, 0.94, 1.0, 0.85)
        for x0, y0, x1, y1 in ((-0.020, 0.0, -0.007, 0.0),
                               (0.007, 0.0, 0.020, 0.0),
                               (0.0, -0.020, 0.0, -0.007),
                               (0.0, 0.007, 0.0, 0.020)):
            lines.moveTo(x0, 0.0, y0)
            lines.drawTo(x1, 0.0, y1)
        self.crosshair = self.root.attachNewNode(lines.create())
        self.crosshair.setDepthTest(False)
        self.crosshair.setBin("fixed", 57)

    def _build_help(self):
        """Full-screen F1 reference so the compact HUD hint can stay small."""
        self.help_visible = False
        self.help_root = self.base.aspect2d.attachNewNode("hud_help")

        maker = CardMaker("help_dim")
        maker.setFrame(-2.0, 2.0, -1.4, 1.4)
        self.help_dim = self.help_root.attachNewNode(maker.generate())
        self.help_dim.setColor(0.008, 0.012, 0.028, 0.88)
        self.help_dim.setTransparency(TransparencyAttrib.MAlpha)
        self.help_dim.setDepthTest(False)
        self.help_dim.setBin("background", 80)

        maker = CardMaker("help_card")
        maker.setFrame(-1.05, 1.05, -0.24, 0.48)
        card = self.help_root.attachNewNode(maker.generate())
        card.setColor(0.05, 0.07, 0.12, 0.94)
        card.setTransparency(TransparencyAttrib.MAlpha)
        card.setDepthTest(False)
        card.setBin("background", 81)

        help_text = (
            "FLIGHT (all axes are relative to the ship)\n"
            "  W/S throttle   A/D roll   arrows/mouse pitch & yaw\n"
            "  Shift boost / hold to charge pulse   Space brake / pull out\n"
            "  X or F land / hold throttle to lift off\n"
            "  R or left mouse mine   E interact / hold to swap ships\n"
            "  On foot: E exit/enter ship  WASD walk  Shift sprint  Space jump\n"
            "  M map   I inventory   P pause   F1 close this help\n"
            "\n"
            "GAMEPAD\n"
            "  left stick pitch/roll   right stick + LB/RB yaw\n"
            "  triggers throttle   Y boost   B brake/back   A mine/interact\n"
            "  X land   Select map/back   Start pause/back\n"
            "\n"
            "GALAXY MAP\n"
            "  A/D or arrows select   Enter warp   V visited only\n"
            "  / search   wheel or +/- zoom   Shift+arrows pan   Home reset\n"
            "\n"
            "OPTIONS (pause menu) control stick/mouse sensitivity, invert Y and deadzone"
        )
        self.help_text = OnscreenText(
            text=help_text, parent=self.help_root, pos=(-1.0, 0.42),
            scale=0.030, fg=_HUD, align=TextNode.ALeft,
            shadow=(0, 0, 0, 0.8))
        self.help_root.hide()

    def set_help_visible(self, visible):
        self.help_visible = bool(visible)
        if self.help_visible:
            self.help_root.show()
            self.controls.hide()
        else:
            self.help_root.hide()
            self.controls.show()

    # -- updates ------------------------------------------------------------
    def set_visible(self, visible):
        if visible:
            self.root.show()
            if self.help_visible:
                self.help_root.show()
        else:
            self.root.hide()
            self.help_root.hide()

    def set_prompt(self, text):
        self.prompt.setText(text or "")

    def set_crosshair_color(self, color):
        try:
            self.crosshair.setColor(*color)
        except Exception:
            pass

    def show_message(self, text, seconds=None, color=None):
        self._message_total = seconds or config.MESSAGE_SECONDS
        self._message_left = self._message_total
        self._message_color = tuple(color) if color else (1.0, 0.92, 0.62)
        self.message.setText(text)

    def update(self, dt, telemetry):
        mode = telemetry.get("mode", "galaxy")
        location = telemetry.get("location", "")
        speed = telemetry.get("speed", 0.0)
        throttle = telemetry.get("throttle", 0.0)
        boost = telemetry.get("boost", False)
        seed = telemetry.get("seed")
        gamepad = telemetry.get("gamepad", False)

        self.title.setText(location)
        on_foot = bool(telemetry.get("on_foot", False))
        subtitle = "ON FOOT" if on_foot else mode.upper()
        if seed is not None:
            subtitle += f"  -  seed {seed}"
        if gamepad:
            subtitle += "  -  gamepad"
        self.subtitle.setText(subtitle)

        lines = [f"SPEED  {speed:5.0f} u/s", f"THROTTLE  {throttle * 100:4.0f}%"]
        if mode == "atmosphere":
            lines.append(f"ALTITUDE  {telemetry.get('altitude', 0.0):6.0f}")
        else:
            lines.append(f"RANGE  {telemetry.get('distance', 0.0):6.0f}")
        position = telemetry.get("position")
        if position is not None:
            lines.append(f"POS  {position[0]:.0f}  {position[1]:.0f}  "
                         f"{position[2]:.0f}")
        self.telemetry.setText("\n".join(lines))

        if on_foot:
            max_speed = config.FOOT_WALK_SPEED * config.FOOT_SPRINT_MULT
        elif mode == "atmosphere":
            max_speed = config.ATMO_MAX_SPEED * 1.22
        else:
            max_speed = config.SPACE_MAX_SPEED * config.SPACE_BOOST_MULT
        speed_fraction = clamp(speed / max(max_speed, 1e-3), 0.0, 1.0)
        bar = self._bar_width
        self.speed_fill.setScale(max(0.001, bar * speed_fraction), 1.0,
                                 0.010)
        throttle_fraction = clamp(
            (throttle + config.SPACE_REVERSE)
            / (1.0 + config.SPACE_REVERSE), 0.0, 1.0)
        self.throttle_fill.setScale(
            max(0.001, bar * throttle_fraction), 1.0, 0.010)
        if throttle < -0.01:
            self.throttle_fill.setColor(1.0, 0.45, 0.30, 0.9)
        elif boost:
            self.throttle_fill.setColor(0.65, 0.82, 1.0, 0.95)
        else:
            self.throttle_fill.setColor(0.55, 1.0, 0.62, 0.9)

        mining_progress = float(telemetry.get("mining_progress", 0.0))
        mining_target = telemetry.get("mining_target")
        if mining_target:
            self.mining_fill.setScale(
                max(0.001, bar * mining_progress), 1.0, 0.010)
            self.mining_fill.setColor(0.95, 0.78, 0.20, 0.95)
        else:
            self.mining_fill.setScale(0.001, 1.0, 0.010)

        # Jetpack fuel display (only shown on foot).
        jetpack_fuel = float(telemetry.get("jetpack_fuel", 0.0))
        if telemetry.get("on_foot", False):
            jp_text = OnscreenText(
                text=f"JP: {jetpack_fuel:.0f}/{config.JETPACK_FUEL_MAX:.0f}",
                parent=self.root, pos=(-1.75, -0.80), scale=0.032,
                fg=(0.1, 0.7, 0.9, 1.0), align=TextNode.ALeft,
                shadow=(0, 0, 0, 0.75), mayChange=True)
            self._jp_text = jp_text
        fuel = float(telemetry.get("fuel", 0.0))
        pulse_fuel = float(telemetry.get("pulse_fuel", 0.0))
        self.fuel_fill.setScale(max(0.001, bar * (fuel / 100.0)), 1.0, 0.010)
        self.pulse_fill.setScale(max(0.001, bar * (pulse_fuel / 100.0)), 1.0, 0.010)
        if telemetry.get("pulse_active"):
            self.pulse_fill.setColor(0.6, 0.8, 1.0, 0.95)
        else:
            self.pulse_fill.setColor(0.55, 1.0, 0.62, 0.9)

        self._update_message(dt)

    def _update_message(self, dt):
        if self._message_left > 0.0:
            self._message_left = max(0.0, self._message_left - dt)
            elapsed = self._message_total - self._message_left
            fade_in = clamp(elapsed / 0.18, 0.0, 1.0)
            fade_out = clamp(self._message_left / 0.75, 0.0, 1.0)
            alpha = min(fade_in, fade_out)
        else:
            alpha = 0.0
        self.message.setFg(Vec4(*self._message_color[:3], alpha))

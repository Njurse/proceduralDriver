"""Pause overlay menu: Continue / Inventory / Ship / Options / Exit."""

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, TextNode, TransparencyAttrib

from core.settings import default_settings, sanitize_settings

_ACCENT = (0.55, 0.95, 1.00, 1.0)
_DIM = (0.46, 0.58, 0.72, 1.0)
_SELECTED = (0.78, 0.90, 0.98, 1.0)

# Main menu actions
PAUSE_CONTINUE = "continue"
PAUSE_INVENTORY = "inventory"
PAUSE_SHIP = "ship"
PAUSE_OPTIONS = "options"
PAUSE_EXIT = "exit"

# Options submenu items
OPT_MASTER_VOLUME = "master_volume"
OPT_EFFECTS_VOLUME = "effects_volume"
OPT_SENSITIVITY = "control_sensitivity"
OPT_MOUSE_SENSITIVITY = "mouse_sensitivity"
OPT_INVERT_Y = "invert_y"
OPT_DEADZONE = "gamepad_deadzone"
OPT_BACK = "back"

_MAIN_LABELS = {
    PAUSE_CONTINUE: "Continue",
    PAUSE_INVENTORY: "Inventory",
    PAUSE_SHIP: "Ship",
    PAUSE_OPTIONS: "Options",
    PAUSE_EXIT: "Exit",
}


def _clamp(value, low, high):
    return max(low, min(high, value))


class PauseOverlay:
    """Pause menu rendered over the 2D overlay layer."""

    def __init__(self, base):
        self.base = base
        self.visible = False
        self.screen = "main"  # "main" or "options"
        self.main_items = [PAUSE_CONTINUE, PAUSE_INVENTORY, PAUSE_SHIP,
                           PAUSE_OPTIONS, PAUSE_EXIT]
        self.option_items = [
            OPT_MASTER_VOLUME, OPT_EFFECTS_VOLUME, OPT_SENSITIVITY,
            OPT_MOUSE_SENSITIVITY, OPT_INVERT_Y, OPT_DEADZONE, OPT_BACK,
        ]
        self.main_index = 0
        self.option_index = 0
        self._row_count = max(len(self.main_items), len(self.option_items))

        # Runtime-adjustable settings (shared with the persistence layer).
        self._settings = default_settings()

        self.root = self.base.aspect2d.attachNewNode("pause_root")

        maker = CardMaker("pause_dim")
        maker.setFrame(-2.0, 2.0, -1.4, 1.4)
        self.dim = self.root.attachNewNode(maker.generate())
        self.dim.setColor(0.008, 0.012, 0.028, 0.80)
        self.dim.setTransparency(TransparencyAttrib.MAlpha)
        self.dim.setDepthTest(False)
        self.dim.setBin("background", 70)

        maker = CardMaker("pause_card")
        maker.setFrame(-0.78, 0.78, -0.52, 0.52)
        self.card = self.root.attachNewNode(maker.generate())
        self.card.setColor(0.05, 0.07, 0.12, 0.90)
        self.card.setTransparency(TransparencyAttrib.MAlpha)
        self.card.setDepthTest(False)
        self.card.setBin("background", 71)

        def text(pos, scale, align, color):
            return OnscreenText(
                text="", parent=self.root, pos=pos, scale=scale, fg=color,
                align=align, shadow=(0, 0, 0, 0.75), mayChange=True)

        self.title = text((0.0, 0.38), 0.085, TextNode.ACenter, _ACCENT)
        self.rows = [
            text((0.0, 0.18 - i * 0.085), 0.052, TextNode.ACenter, _DIM)
            for i in range(self._row_count)
        ]
        self.prompt = text((0.0, -0.44), 0.034, TextNode.ACenter, _DIM)

        self.layout(base.getAspectRatio())
        self.set_visible(False)

    # -- layout -------------------------------------------------------------
    def layout(self, aspect):
        """Scale the dimmer to cover ultrawide/narrow windows."""
        half_width = max(float(aspect), 0.5)
        # The dim card is 4 units wide (-2..2); stretch it to the full width.
        self.dim.setScale(max(1.0, half_width / 2.0), 1.0, 1.0)

    # -- settings helpers ---------------------------------------------------
    def settings(self):
        return dict(self._settings)

    def apply_settings(self, settings):
        self._settings = sanitize_settings(settings)

    def _option_value_text(self, item):
        if item == OPT_MASTER_VOLUME:
            return f"Master Volume: {int(round(self._settings['master_volume'] * 100))}%"
        if item == OPT_EFFECTS_VOLUME:
            return f"Effects Volume: {int(round(self._settings['effects_volume'] * 100))}%"
        if item == OPT_SENSITIVITY:
            return f"Stick Sensitivity: {self._settings['control_sensitivity']:.1f}x"
        if item == OPT_MOUSE_SENSITIVITY:
            return f"Mouse Sensitivity: {self._settings['mouse_sensitivity']:.2f}x"
        if item == OPT_INVERT_Y:
            return f"Invert Look Y: {'On' if self._settings['invert_y'] else 'Off'}"
        if item == OPT_DEADZONE:
            return f"Stick Deadzone: {int(round(self._settings['gamepad_deadzone'] * 100))}%"
        return "Back"

    # -- display ------------------------------------------------------------
    def _refresh(self):
        if self.screen == "main":
            self.title.setText("PAUSED")
            for i, row in enumerate(self.rows):
                if i < len(self.main_items):
                    item = self.main_items[i]
                    row.setText(_MAIN_LABELS.get(item, item))
                    row.setFg(_SELECTED if i == self.main_index else _DIM)
                    row.show()
                else:
                    row.hide()
            self.prompt.setText(
                "Up/Down or W/S: navigate   Enter/A: select   P/Esc/Start: resume")
        else:
            self.title.setText("OPTIONS")
            for i, row in enumerate(self.rows):
                if i >= len(self.option_items):
                    row.hide()
                    continue
                item = self.option_items[i]
                row.setText(self._option_value_text(item))
                row.setFg(_SELECTED if i == self.option_index else _DIM)
                row.show()
            self.prompt.setText(
                "Up/Down: select   Left/Right or A/D: adjust   "
                "Enter: select   Esc/B/Start: back")

    # -- state --------------------------------------------------------------
    def set_visible(self, visible):
        self.visible = bool(visible)
        if self.visible:
            self.screen = "main"
            self.main_index = 0
            self.option_index = 0
            self._refresh()
            self.root.show()
        else:
            self.root.hide()

    def move(self, delta):
        if not self.visible:
            return
        if self.screen == "main":
            self.main_index = (self.main_index + int(delta)) % len(self.main_items)
        else:
            self.option_index = (self.option_index + int(delta)) % len(self.option_items)
        self._refresh()

    def adjust(self, delta):
        """Adjust selected option value; returns True if a setting changed."""
        if not self.visible or self.screen != "options":
            return False
        item = self.option_items[self.option_index]
        step = 1 if delta > 0 else -1
        settings = self._settings
        if item == OPT_MASTER_VOLUME:
            settings["master_volume"] = _clamp(
                round((settings["master_volume"] + step * 0.05) * 20.0) / 20.0,
                0.0, 1.0)
        elif item == OPT_EFFECTS_VOLUME:
            settings["effects_volume"] = _clamp(
                round((settings["effects_volume"] + step * 0.05) * 20.0) / 20.0,
                0.0, 1.0)
        elif item == OPT_SENSITIVITY:
            settings["control_sensitivity"] = _clamp(
                round((settings["control_sensitivity"] + step * 0.25) * 4.0) / 4.0,
                0.1, 10.0)
        elif item == OPT_MOUSE_SENSITIVITY:
            settings["mouse_sensitivity"] = _clamp(
                round((settings["mouse_sensitivity"] + step * 0.10) * 20.0) / 20.0,
                0.1, 5.0)
        elif item == OPT_INVERT_Y:
            settings["invert_y"] = not settings["invert_y"]
        elif item == OPT_DEADZONE:
            settings["gamepad_deadzone"] = _clamp(
                round((settings["gamepad_deadzone"] + step * 0.02) * 100.0) / 100.0,
                0.05, 0.4)
        else:
            return False
        self._refresh()
        return True

    def back(self):
        """Handle back/cancel; returns True if pause should fully close."""
        if not self.visible:
            return False
        if self.screen == "options":
            self.screen = "main"
            self._refresh()
            return False
        return True

    def activate(self):
        """Activate current selection; returns action string or None."""
        if not self.visible:
            return None
        if self.screen == "main":
            item = self.main_items[self.main_index]
            if item == PAUSE_OPTIONS:
                self.screen = "options"
                self.option_index = 0
                self._refresh()
                return None
            if item == PAUSE_INVENTORY:
                return PAUSE_INVENTORY
            if item == PAUSE_SHIP:
                return PAUSE_SHIP
            return item
        # options screen
        item = self.option_items[self.option_index]
        if item == OPT_BACK:
            self.screen = "main"
            self._refresh()
        return None

    @property
    def in_options(self):
        return self.visible and self.screen == "options"

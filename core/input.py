"""Unified action layer merging keyboard, mouse and gamepad input."""

import logging

from panda3d.core import KeyboardButton, MouseButton, WindowProperties

import config
from core.gamepad import Gamepad
from core.mathutil import clamp

log = logging.getLogger("galaxy_explorer.input")

_KEYS = {
    "forward": KeyboardButton.asciiKey("w"),
    "back": KeyboardButton.asciiKey("s"),
    "left": KeyboardButton.asciiKey("a"),
    "right": KeyboardButton.asciiKey("d"),
    "pitch_up": KeyboardButton.up(),
    "pitch_down": KeyboardButton.down(),
    "yaw_left": KeyboardButton.left(),
    "yaw_right": KeyboardButton.right(),
    "boost": KeyboardButton.shift(),
    "brake": KeyboardButton.space(),
    "interact": KeyboardButton.asciiKey("e"),
    "map": KeyboardButton.asciiKey("m"),
    "map_alt": KeyboardButton.tab(),
    "release": KeyboardButton.escape(),
    "confirm": KeyboardButton.enter(),
    "pause": KeyboardButton.asciiKey("p"),
    "land": KeyboardButton.asciiKey("f"),
    "land_alt": KeyboardButton.asciiKey("x"),
    "mine": KeyboardButton.asciiKey("r"),
    "inventory": KeyboardButton.asciiKey("i"),
    "help": KeyboardButton.f1(),
}


def _deadzone(value, threshold):
    if abs(value) < threshold:
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    return sign * (abs(value) - threshold) / max(1.0 - threshold, 1e-6)


class InputState:
    """Current frame of merged input; edge flags last exactly one frame."""

    __slots__ = (
        "throttle_axis", "pitch_axis", "yaw_axis", "lateral_axis",
        "pad_roll_axis", "mouse_dx", "mouse_dy",
        "boost", "brake", "gamepad_connected", "sprint", "jump",
        "jetpack", "interact", "toggle_map", "release_mouse", "confirm", "cancel",
        "select_prev", "select_next", "select_up", "select_down", "pause",
        "land", "mine_held", "toggle_inventory", "toggle_help",
    )

    def __init__(self):
        self.throttle_axis = 0.0
        self.pitch_axis = 0.0
        self.yaw_axis = 0.0
        self.lateral_axis = 0.0
        self.pad_roll_axis = 0.0
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.boost = False
        self.brake = False
        self.gamepad_connected = False
        self.sprint = False
        self.jump = False
        self.jetpack = False
        self.interact = False
        self.toggle_map = False
        self.release_mouse = False
        self.confirm = False
        self.cancel = False
        self.select_prev = False
        self.select_next = False
        self.select_up = False
        self.select_down = False
        self.pause = False
        self.land = False
        self.mine_held = False
        self.toggle_inventory = False
        self.toggle_help = False


class InputManager:
    def __init__(self, base):
        self.base = base
        self.pad = Gamepad()
        self.state = InputState()
        self.mouse_look = False
        self._prev = {}
        # Runtime-adjustable settings; apply_settings() is called by game.py.
        self.mouse_sensitivity = float(
            getattr(config, "MOUSE_SENSITIVITY", 1.0))
        self.invert_y = bool(getattr(config, "INVERT_Y", False))
        self.deadzone = float(getattr(config, "GAMEPAD_DEADZONE", 0.16))

    def apply_settings(self, settings):
        """Update look sensitivity, axis inversion and gamepad deadzone."""
        if not isinstance(settings, dict):
            return
        try:
            self.mouse_sensitivity = clamp(
                float(settings.get("mouse_sensitivity",
                                   self.mouse_sensitivity)), 0.1, 5.0)
            self.deadzone = clamp(
                float(settings.get("gamepad_deadzone", self.deadzone)),
                0.05, 0.4)
        except (TypeError, ValueError):
            pass
        self.invert_y = bool(settings.get("invert_y", self.invert_y))

    # -- helpers ------------------------------------------------------------
    def _down(self, action):
        watcher = getattr(self.base, "mouseWatcherNode", None)
        if watcher is None:
            return False
        button = _KEYS.get(action)
        return bool(button is not None and watcher.is_button_down(button))

    def _edge(self, name, down):
        was_down = self._prev.get(name, False)
        self._prev[name] = down
        return down and not was_down

    def set_mouse_look(self, enabled):
        """Toggle relative pointer mode for first-person style look."""
        if enabled == self.mouse_look:
            return
        self.mouse_look = enabled
        props = WindowProperties()
        props.setCursorHidden(enabled)
        props.setMouseMode(
            WindowProperties.M_relative if enabled
            else WindowProperties.M_absolute)
        try:
            if self.base.win is not None:
                self.base.win.requestProperties(props)
        except Exception as exc:
            log.debug("[input] cursor mode change failed: %s", exc)

    def _mouse_delta(self):
        watcher = getattr(self.base, "mouseWatcherNode", None)
        if not self.mouse_look or watcher is None:
            return 0.0, 0.0
        try:
            if not watcher.hasMouse():
                return 0.0, 0.0
            # In relative mode Panda reports per-frame pointer movement.
            dx = watcher.getMouseX()
            dy = watcher.getMouseY()
        except Exception as exc:
            log.debug("[input] mouse delta unavailable: %s", exc)
            return 0.0, 0.0
        limit = config.MOUSE_DELTA_LIMIT
        dx = clamp(dx, -limit, limit) * self.mouse_sensitivity
        dy = clamp(dy, -limit, limit) * self.mouse_sensitivity
        if self.invert_y:
            dy = -dy
        return dx, dy

    # -- main poll ----------------------------------------------------------
    def poll(self):
        state = self.state
        state.throttle_axis = 0.0
        state.pitch_axis = 0.0
        state.yaw_axis = 0.0
        state.lateral_axis = 0.0
        state.pad_roll_axis = 0.0
        for name in ("interact", "toggle_map", "release_mouse", "confirm",
                     "cancel", "select_prev", "select_next",
                     "select_up", "select_down", "pause", "land",
                     "toggle_help"):
            setattr(state, name, False)
        state.mine_held = False
        state.toggle_inventory = False
        state.jump = False
        state.sprint = False
        state.jetpack = False

        kb_throttle = float(self._down("forward")) - float(self._down("back"))
        kb_pitch = float(self._down("pitch_up")) - float(self._down("pitch_down"))
        kb_yaw = float(self._down("yaw_right")) - float(self._down("yaw_left"))
        kb_lateral = float(self._down("right")) - float(self._down("left"))
        boost = self._down("boost")
        brake = self._down("brake")
        interact = self._down("interact")
        map_pressed = self._down("map") or self._down("map_alt")
        release = self._down("release")
        confirm = self._down("confirm")
        pause_pressed = self._down("pause")
        land_pressed = self._down("land") or self._down("land_alt")
        mine_held = self._down("mine")
        inventory_pressed = self._down("inventory")
        help_pressed = self._down("help")
        nav_left = self._down("yaw_left") or self._down("left")
        nav_right = self._down("yaw_right") or self._down("right")
        nav_up = self._down("pitch_up") or self._down("forward")
        nav_down = self._down("pitch_down") or self._down("back")

        self.pad.poll()
        state.gamepad_connected = self.pad.connected
        if self.pad.connected:
            # Connected controller takes control precedence for flight.
            throttle = self.pad.right_trigger - self.pad.left_trigger
            # Consistent angle mapping:
            #   left stick X -> roll (right = bank right)
            #   left stick Y -> pitch (up = nose up)
            #   right stick X + bumpers -> yaw (right/RB = turn right)
            # Roll comes only from the stick; bumpers never fight it, and they
            # share the same sign convention as keyboard/mouse yaw.
            pitch = _deadzone(self.pad.left_y, self.deadzone)
            if self.invert_y:
                pitch = -pitch
            bumper_yaw = (1.0 if self.pad.right_shoulder else 0.0) - (
                1.0 if self.pad.left_shoulder else 0.0)
            yaw = clamp(_deadzone(self.pad.right_x, self.deadzone) + bumper_yaw,
                        -1.0, 1.0)
            lateral = 0.0
            state.pad_roll_axis = _deadzone(self.pad.left_x, self.deadzone)
            boost = self.pad.boost
            brake = self.pad.brake
            interact = self.pad.interact
            map_pressed = self.pad.map_button
            confirm = confirm or self.pad.interact
            pause_pressed = pause_pressed or self.pad.start_button
            nav_left = self.pad.dpad_left
            nav_right = self.pad.dpad_right
            nav_up = self.pad.dpad_up
            nav_down = self.pad.dpad_down
            land_pressed = land_pressed or self.pad.x_button
            mine_held = mine_held or getattr(self.pad, "a_button", False)
            mouse_dx, mouse_dy = 0.0, 0.0
        else:
            throttle = kb_throttle
            pitch = kb_pitch
            yaw = kb_yaw
            lateral = kb_lateral
            state.pad_roll_axis = 0.0
            mouse_dx, mouse_dy = self._mouse_delta()

        state.throttle_axis = clamp(throttle, -1.0, 1.0)
        state.pitch_axis = clamp(pitch, -1.0, 1.0)
        state.yaw_axis = clamp(yaw, -1.0, 1.0)
        state.lateral_axis = clamp(lateral, -1.0, 1.0)
        state.boost = boost
        state.brake = brake

        # Mouse left button held also mines.
        watcher = getattr(self.base, "mouseWatcherNode", None)
        if (watcher is not None
                and hasattr(watcher, "is_button_down")
                and watcher.is_button_down(MouseButton.one())):
            mine_held = True
        state.mouse_dx, state.mouse_dy = mouse_dx, mouse_dy
        state.mine_held = bool(mine_held)

        # Jump reuses the spacebar (brake in flight) and gamepad X button.
        space_down = self._down("brake")
        gamepad_jump = False
        if self.pad.connected:
            gamepad_jump = bool(getattr(self.pad, "x_button", False))
        jump_pressed = space_down or gamepad_jump
        state.sprint = bool(boost)
        # Jetpack on foot: space held while not in flight mode.
        # In space/atmosphere flight, space = brake; when on foot, space = jetpack.
        # We set jetpack True here; the foot controller decides if it's valid.
        state.jetpack = space_down

        state.interact = self._edge("interact", interact)
        state.toggle_map = self._edge("map", map_pressed)
        state.release_mouse = self._edge("release", release)
        state.confirm = self._edge("confirm", confirm) or state.interact
        cancel = release or map_pressed
        if self.pad.connected and getattr(self.pad, "brake", False):
            # B doubles as "back" in menus; in flight it is just the brake.
            cancel = True
        state.cancel = self._edge("cancel", cancel)
        state.select_prev = self._edge("select_prev", nav_left)
        state.select_next = self._edge("select_next", nav_right)
        state.select_up = self._edge("select_up", nav_up)
        state.select_down = self._edge("select_down", nav_down)
        state.pause = self._edge("pause", pause_pressed)
        state.land = self._edge("land", land_pressed)
        state.jump = self._edge("jump", jump_pressed)
        state.toggle_inventory = self._edge("inventory", inventory_pressed)
        state.toggle_help = self._edge("help", help_pressed)
        return state

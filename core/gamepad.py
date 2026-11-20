"""Gamepad polling through pygame-ce and native XInput.  Used only for input."""

import logging
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

from core import xinput
from core.mathutil import clamp

log = logging.getLogger("gamepad")

_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1
_AXIS_RIGHT_X = 2
_AXIS_RIGHT_Y = 3  # positive = stick pulled down (SDL convention)
_AXIS_LT = 4
_AXIS_RT = 5

_BUTTON_A = 0
_BUTTON_B = 1
_BUTTON_X = 2
_BUTTON_Y = 3
_BUTTON_LB = 4
_BUTTON_RB = 5
_BUTTON_SELECT = 6
_BUTTON_START = 7
_BUTTON_RIGHT_THUMB = 9  # R3 (stick click)
_BUTTON_DPAD_UP = 11
_BUTTON_DPAD_DOWN = 12
_BUTTON_DPAD_LEFT = 13
_BUTTON_DPAD_RIGHT = 14

class Gamepad:
    """Auto-detects XInput controllers on Windows, falls back to pygame-ce."""

    def __init__(self):
        self.pygame = None
        self.joystick = None
        self.backend = None  # "xinput" or "pygame"
        self.xinput_user = None
        self.connected = False
        self.name = ""
        self.axes = []
        self.buttons = []
        self._rest = {_AXIS_LT: -1.0, _AXIS_RT: -1.0}
        self._next_scan = 0.0

    # -- setup --------------------------------------------------------------
    def _ensure_pygame(self):
        if self.pygame is not None:
            return self.pygame
        try:
            import pygame
            # A display is not required for joysticks; only try to bring one up
            # when the platform already provides a working video driver.
            try:
                if not pygame.display.get_init():
                    pygame.display.init()
            except Exception as exc:
                log.debug("[gamepad] pygame display init failed: %s", exc)
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            self.pygame = pygame
        except Exception as exc:
            log.debug("[gamepad] pygame unavailable: %s", exc)
            self.pygame = False
        return self.pygame or None

    def _scan_xinput(self):
        found = xinput.scan()
        if found is None:
            return False
        user, _state = found
        self.backend = "xinput"
        self.xinput_user = user
        self.connected = True
        self.name = f"XInput Controller {user}"
        # XInput triggers are reported as 0..255 -> normalized to 0..1.
        self._rest[_AXIS_LT] = 0.0
        self._rest[_AXIS_RT] = 0.0
        self._read_xinput()
        return True

    def _scan(self):
        # Prefer native XInput auto-detection on Windows.
        try:
            if self._scan_xinput():
                return
        except Exception as exc:
            log.debug("[gamepad] XInput scan failed: %s", exc)

        pygame = self._ensure_pygame()
        if pygame is None:
            if self.connected:
                self._disconnect()
            return
        try:
            count = pygame.joystick.get_count()
        except Exception as exc:
            log.debug("[gamepad] joystick count failed: %s", exc)
            count = 0
        target = None
        for index in range(max(int(count), 0)):
            try:
                joystick = pygame.joystick.Joystick(index)
                joystick.init()
                target = joystick
                break
            except Exception as exc:
                log.debug("[gamepad] joystick %d init failed: %s",
                          index, exc)
                continue
        if target is None:
            if self.connected:
                self._disconnect()
            return
        try:
            self.joystick = target
            self.backend = "pygame"
            self.connected = True
            self.name = target.get_name()
            self._read_pygame()
            # Calibrate trigger rest values (drivers disagree: -1 or 0).
            self._rest[_AXIS_LT] = self._raw_axis(_AXIS_LT)
            self._rest[_AXIS_RT] = self._raw_axis(_AXIS_RT)
        except Exception as exc:
            log.warning("[gamepad] joystick setup failed: %s", exc)
            self._disconnect()

    def _disconnect(self):
        try:
            if self.joystick is not None:
                self.joystick.quit()
        except Exception as exc:
            log.debug("[gamepad] joystick quit failed: %s", exc)
        self.joystick = None
        self.backend = None
        self.xinput_user = None
        self.connected = False
        self.name = ""
        self.axes = []
        self.buttons = []

    def _read_xinput(self):
        pad = None if self.xinput_user is None else xinput.read(self.xinput_user)
        if pad is None:
            self._disconnect()
            return
        axes = [0.0] * 8
        # Match existing mapping: 0 LX, 1 LY, 2 RX, 3 RY, 4 LT, 5 RT
        axes[_AXIS_LEFT_X] = xinput.norm_thumb(pad.sThumbLX)
        axes[_AXIS_LEFT_Y] = xinput.norm_thumb(pad.sThumbLY)
        axes[_AXIS_RIGHT_X] = xinput.norm_thumb(pad.sThumbRX)
        axes[_AXIS_RIGHT_Y] = -xinput.norm_thumb(pad.sThumbRY)  # XInput Y is up-positive
        axes[_AXIS_LT] = pad.bLeftTrigger / 255.0
        axes[_AXIS_RT] = pad.bRightTrigger / 255.0
        buttons = [False] * 16
        w = int(pad.wButtons)
        buttons[_BUTTON_A] = bool(w & xinput.A)
        buttons[_BUTTON_B] = bool(w & xinput.B)
        buttons[_BUTTON_X] = bool(w & xinput.X)
        buttons[_BUTTON_Y] = bool(w & xinput.Y)
        buttons[_BUTTON_LB] = bool(w & xinput.LEFT_SHOULDER)
        buttons[_BUTTON_RB] = bool(w & xinput.RIGHT_SHOULDER)
        buttons[_BUTTON_SELECT] = bool(w & xinput.BACK)
        buttons[_BUTTON_START] = bool(w & xinput.START)
        buttons[_BUTTON_RIGHT_THUMB] = bool(w & xinput.RIGHT_THUMB)
        buttons[_BUTTON_DPAD_UP] = bool(w & xinput.DPAD_UP)
        buttons[_BUTTON_DPAD_DOWN] = bool(w & xinput.DPAD_DOWN)
        buttons[_BUTTON_DPAD_LEFT] = bool(w & xinput.DPAD_LEFT)
        buttons[_BUTTON_DPAD_RIGHT] = bool(w & xinput.DPAD_RIGHT)
        self.axes = axes
        self.buttons = buttons

    def _read_pygame(self):
        if self.joystick is None:
            return
        axes = []
        for index in range(8):
            try:
                axes.append(float(self.joystick.get_axis(index)))
            except Exception as exc:
                log.debug("[gamepad] axis %d read failed: %s", index, exc)
                self._disconnect()
                return
        buttons = []
        for index in range(16):
            try:
                buttons.append(bool(self.joystick.get_button(index)))
            except Exception as exc:
                log.debug("[gamepad] button %d read failed: %s", index, exc)
                self._disconnect()
                return
        self.axes = axes
        self.buttons = buttons

    def _read(self):
        if self.backend == "xinput":
            self._read_xinput()
        elif self.backend == "pygame":
            self._read_pygame()

    # -- polling ------------------------------------------------------------
    def poll(self):
        # Pump pygame events when available to keep its device state fresh.
        pygame = self._ensure_pygame()
        if pygame is not None:
            try:
                pygame.event.pump()
            except Exception as exc:
                log.debug("[gamepad] event pump failed: %s", exc)
        if not self.connected:
            now = time.monotonic()
            if now < self._next_scan:
                return
            self._next_scan = now + 2.5
            self._scan()
            return
        try:
            self._read()
        except Exception as exc:
            log.debug("[gamepad] poll failed: %s", exc)
            self._disconnect()
        if not self.connected:
            self._next_scan = time.monotonic() + 2.5

    # -- accessors ----------------------------------------------------------
    def _raw_axis(self, index):
        if 0 <= index < len(self.axes):
            return float(self.axes[index])
        return -1.0 if index in self._rest else 0.0

    def button(self, index):
        if 0 <= index < len(self.buttons):
            return bool(self.buttons[index])
        return False

    def stick(self, index):
        value = self._raw_axis(index)
        if abs(value) < 0.08:
            return 0.0
        return clamp(value, -1.0, 1.0)

    def trigger(self, index):
        raw = self._raw_axis(index)
        rest = self._rest.get(index, -1.0)
        span = max(1.0 - rest, 1e-4)
        return clamp((raw - rest) / span, 0.0, 1.0)

    @property
    def left_x(self):
        return self.stick(_AXIS_LEFT_X)

    @property
    def left_y(self):
        return self.stick(_AXIS_LEFT_Y)

    @property
    def right_x(self):
        return self.stick(_AXIS_RIGHT_X)

    @property
    def right_y(self):
        return self.stick(_AXIS_RIGHT_Y)

    @property
    def right_thumb(self):
        return self.button(_BUTTON_RIGHT_THUMB)

    @property
    def left_trigger(self):
        return self.trigger(_AXIS_LT)

    @property
    def right_trigger(self):
        return self.trigger(_AXIS_RT)

    @property
    def left_shoulder(self):
        return self.button(_BUTTON_LB)

    @property
    def right_shoulder(self):
        return self.button(_BUTTON_RB)

    @property
    def x_button(self):
        return self.button(_BUTTON_X)

    @property
    def a_button(self):
        return self.button(_BUTTON_A)

    @property
    def boost(self):
        return self.button(_BUTTON_Y)

    @property
    def brake(self):
        return self.button(_BUTTON_B)

    @property
    def interact(self):
        return self.button(_BUTTON_A)

    @property
    def map_button(self):
        return self.button(_BUTTON_SELECT)

    @property
    def start_button(self):
        return self.button(_BUTTON_START)

    @property
    def dpad_up(self):
        return self.button(_BUTTON_DPAD_UP)

    @property
    def dpad_down(self):
        return self.button(_BUTTON_DPAD_DOWN)

    @property
    def dpad_left(self):
        return self.button(_BUTTON_DPAD_LEFT)

    @property
    def dpad_right(self):
        return self.button(_BUTTON_DPAD_RIGHT)

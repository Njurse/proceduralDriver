"""Gamepad polling through pygame-ce and native XInput.  Used only for input."""

import logging
import os
import sys
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from core.mathutil import clamp

log = logging.getLogger("galaxy_explorer.gamepad")

_AXIS_LEFT_X = 0
_AXIS_LEFT_Y = 1
_AXIS_RIGHT_X = 2
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
_BUTTON_DPAD_UP = 11
_BUTTON_DPAD_DOWN = 12
_BUTTON_DPAD_LEFT = 13
_BUTTON_DPAD_RIGHT = 14

# -- XInput auto-detection (Windows) ----------------------------------------
_XINPUT = None
if sys.platform == "win32":
    try:
        import ctypes
        from ctypes import wintypes

        class _XINPUT_GAMEPAD(ctypes.Structure):
            _fields_ = [
                ("wButtons", wintypes.WORD),
                ("bLeftTrigger", wintypes.BYTE),
                ("bRightTrigger", wintypes.BYTE),
                ("sThumbLX", wintypes.SHORT),
                ("sThumbLY", wintypes.SHORT),
                ("sThumbRX", wintypes.SHORT),
                ("sThumbRY", wintypes.SHORT),
            ]

        class _XINPUT_STATE(ctypes.Structure):
            _fields_ = [
                ("dwPacketNumber", wintypes.DWORD),
                ("Gamepad", _XINPUT_GAMEPAD),
            ]

        _ERROR_SUCCESS = 0
        _DPAD_UP = 0x0001
        _DPAD_DOWN = 0x0002
        _DPAD_LEFT = 0x0004
        _DPAD_RIGHT = 0x0008
        _START = 0x0010
        _BACK = 0x0020
        _LEFT_THUMB = 0x0040
        _RIGHT_THUMB = 0x0080
        _LEFT_SHOULDER = 0x0100
        _RIGHT_SHOULDER = 0x0200
        _A = 0x1000
        _B = 0x2000
        _X = 0x4000
        _Y = 0x8000

        for _dll in ("xinput1_4", "xinput1_3", "xinput9_1_0", "xinput1_2", "xinput1_1"):
            try:
                _lib = getattr(ctypes.windll, _dll)
                _get_state = _lib.XInputGetState
                _get_state.argtypes = [wintypes.DWORD, ctypes.POINTER(_XINPUT_STATE)]
                _get_state.restype = wintypes.DWORD
                _XINPUT = (_get_state, _XINPUT_STATE)
                break
            except Exception as exc:
                log.debug("[gamepad] XInput DLL %s unavailable: %s",
                          _dll, exc)
                continue
    except Exception as exc:
        log.debug("[gamepad] XInput setup failed: %s", exc)
        _XINPUT = None


def _xinput_scan():
    """Return (user_index, state) for the first connected XInput controller."""
    if _XINPUT is None:
        return None
    get_state, state_type = _XINPUT
    for user in range(4):
        state = state_type()
        try:
            if get_state(user, ctypes.byref(state)) == _ERROR_SUCCESS:
                return user, state
        except Exception as exc:
            log.debug("[gamepad] XInput user %d query failed: %s", user, exc)
            continue
    return None


def _norm_thumb(value):
    # XInput thumbs are signed 16-bit; normalize to [-1, 1].
    if value < 0:
        return max(-1.0, value / 32768.0)
    return min(1.0, value / 32767.0)


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
            if not pygame.display.get_init():
                pygame.display.init()
            if not pygame.joystick.get_init():
                pygame.joystick.init()
            self.pygame = pygame
        except Exception as exc:
            log.debug("[gamepad] pygame unavailable: %s", exc)
            self.pygame = False
        return self.pygame or None

    def _scan_xinput(self):
        found = _xinput_scan()
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
        if self.xinput_user is None or _XINPUT is None:
            self._disconnect()
            return
        get_state, state_type = _XINPUT
        state = state_type()
        try:
            if get_state(self.xinput_user, ctypes.byref(state)) != _ERROR_SUCCESS:
                self._disconnect()
                return
        except Exception as exc:
            log.debug("[gamepad] XInput read failed: %s", exc)
            self._disconnect()
            return
        pad = state.Gamepad
        axes = [0.0] * 8
        # Match existing mapping: 0 LX, 1 LY, 2 RX, 4 LT, 5 RT
        axes[_AXIS_LEFT_X] = _norm_thumb(pad.sThumbLX)
        axes[_AXIS_LEFT_Y] = _norm_thumb(pad.sThumbLY)
        axes[_AXIS_RIGHT_X] = _norm_thumb(pad.sThumbRX)
        axes[_AXIS_LT] = pad.bLeftTrigger / 255.0
        axes[_AXIS_RT] = pad.bRightTrigger / 255.0
        buttons = [False] * 16
        w = int(pad.wButtons)
        buttons[_BUTTON_A] = bool(w & _A)
        buttons[_BUTTON_B] = bool(w & _B)
        buttons[_BUTTON_X] = bool(w & _X)
        buttons[_BUTTON_Y] = bool(w & _Y)
        buttons[_BUTTON_LB] = bool(w & _LEFT_SHOULDER)
        buttons[_BUTTON_RB] = bool(w & _RIGHT_SHOULDER)
        buttons[_BUTTON_SELECT] = bool(w & _BACK)
        buttons[_BUTTON_START] = bool(w & _START)
        buttons[_BUTTON_DPAD_UP] = bool(w & _DPAD_UP)
        buttons[_BUTTON_DPAD_DOWN] = bool(w & _DPAD_DOWN)
        buttons[_BUTTON_DPAD_LEFT] = bool(w & _DPAD_LEFT)
        buttons[_BUTTON_DPAD_RIGHT] = bool(w & _DPAD_RIGHT)
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

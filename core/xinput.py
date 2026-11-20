"""Windows XInput through ctypes: controller state for core.gamepad.

Loads the first XInput DLL that exists; elsewhere (or if none loads) every
query returns None and core.gamepad falls back to pygame-ce.
"""

import logging
import sys

log = logging.getLogger("gamepad")

ERROR_SUCCESS = 0
# XINPUT_GAMEPAD.wButtons bits
DPAD_UP = 0x0001
DPAD_DOWN = 0x0002
DPAD_LEFT = 0x0004
DPAD_RIGHT = 0x0008
START = 0x0010
BACK = 0x0020
LEFT_THUMB = 0x0040
RIGHT_THUMB = 0x0080
LEFT_SHOULDER = 0x0100
RIGHT_SHOULDER = 0x0200
A = 0x1000
B = 0x2000
X = 0x4000
Y = 0x8000

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


def scan():
    """Return (user_index, state) for the first connected XInput controller."""
    if _XINPUT is None:
        return None
    get_state, state_type = _XINPUT
    for user in range(4):
        state = state_type()
        try:
            if get_state(user, ctypes.byref(state)) == ERROR_SUCCESS:
                return user, state
        except Exception as exc:
            log.debug("[gamepad] XInput user %d query failed: %s", user, exc)
            continue
    return None


def read(user):
    """XINPUT_GAMEPAD for ``user``, or None if it is gone or the read failed."""
    if _XINPUT is None:
        return None
    get_state, state_type = _XINPUT
    state = state_type()
    try:
        if get_state(user, ctypes.byref(state)) != ERROR_SUCCESS:
            return None
    except Exception as exc:
        log.debug("[gamepad] XInput read failed: %s", exc)
        return None
    return state.Gamepad


def norm_thumb(value):
    # XInput thumbs are signed 16-bit; normalize to [-1, 1].
    if value < 0:
        return max(-1.0, value / 32768.0)
    return min(1.0, value / 32767.0)

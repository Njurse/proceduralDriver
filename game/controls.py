"""Keys and gamepad: driving flags, camera keys, and the pause menu's
keyboard/gamepad navigation (a DriveSim mixin)."""

import logging

log = logging.getLogger(__name__)


class _NoPad:
    """Stand-in Gamepad that is never connected."""

    connected = False

    def poll(self):
        pass


class Controls:
    """Keyboard and gamepad bindings, and driving the pause menu with them."""

    def _bind_keys(self):
        self.key_throttle = False
        self.key_brake = False
        self.key_left = False
        self.key_right = False
        self.key_hb = False
        self.key_fast = False
        self.key_cam_up = self.key_cam_down = False
        self.key_cam_left = self.key_cam_right = self.key_cam_latch = False
        for name, flag in [("w", "key_throttle"), ("s", "key_brake"),
                           ("a", "key_left"), ("d", "key_right"),
                           ("space", "key_hb"), ("lshift", "key_fast"),
                           ("i", "key_cam_up"), ("k", "key_cam_down"),
                           ("j", "key_cam_left"), ("l", "key_cam_right"),
                           ("o", "key_cam_latch")]:
            self.accept(name, self._set_key, [flag, True])
            self.accept(name + "-up", self._set_key, [flag, False])
        self.accept("escape", self._escape)
        for key, fn in [("arrow_up", lambda: self._menu_move(-1)),
                        ("arrow_down", lambda: self._menu_move(1)),
                        ("arrow_left", lambda: self._menu_adjust(-1)),
                        ("arrow_right", lambda: self._menu_adjust(1)),
                        ("enter", self._menu_activate), ("backspace", self._menu_backspace),
                        ("m", self.minimap.toggle_zoom)]:
            self.accept(key, fn)
        for d in "0123456789":
            self.accept(d, self._menu_type, [d])

    def _bind_window_events(self):
        # Shut the audio down cleanly when the window's X is used.
        self.exitFunc = self._on_exit
        try:
            if self.win is not None:
                self.win.setCloseRequestEvent("window-close")
                self.accept("window-close", self._quit)
        except Exception as exc:
            log.debug("[window] close handler unavailable: %s", exc)

    def _set_key(self, flag, value):
        setattr(self, flag, value)
        if value and self.paused:   # W/S/A/D/space work the menu while paused
            {"key_throttle": lambda: self._menu_move(-1),
             "key_brake": lambda: self._menu_move(1),
             "key_left": lambda: self._menu_adjust(-1),
             "key_right": lambda: self._menu_adjust(1),
             "key_hb": self._menu_activate}.get(flag, lambda: None)()

    def _escape(self):
        if not self.paused:
            self.set_paused(True)
        elif self.pause_menu.back():
            self.set_paused(False)

    def _menu_move(self, d):
        if self.paused:
            self.pause_menu.move(d)

    def _menu_adjust(self, d):
        if self.paused:
            self.pause_menu.adjust(d)

    def _menu_type(self, ch):
        if self.paused:
            self.pause_menu.type_char(ch)

    def _menu_backspace(self):
        if self.paused:
            self.pause_menu.backspace()

    def _menu_activate(self):
        if self.paused:
            self.menu_action(self.pause_menu.activate())

    def _pad_menu(self):
        """Gamepad edges: Start pauses, Back zooms the map; in the menu the
        d-pad / left stick move, A selects, B goes back."""
        pad = self.pad
        if not pad.connected:
            return
        now = {"start": pad.start_button, "back": pad.map_button, "a": pad.a_button,
               "b": pad.brake, "up": pad.dpad_up, "down": pad.dpad_down,
               "left": pad.dpad_left, "right": pad.dpad_right}
        hit = {k for k, v in now.items() if v and not self._pad_prev.get(k)}
        self._pad_prev = now
        # XInput reports the stick up-positive, SDL down-positive
        y = pad.left_y * (-1.0 if pad.backend == "xinput" else 1.0)
        stick = (y > 0.6) - (y < -0.6)
        if stick and stick != self._stick_prev:
            hit.add("down" if stick > 0 else "up")
        self._stick_prev = stick
        if "back" in hit:
            self.minimap.toggle_zoom()
        if "start" in hit:
            self.set_paused(not self.paused)
            return
        if not self.paused:
            return
        if "up" in hit:
            self.pause_menu.move(-1)
        if "down" in hit:
            self.pause_menu.move(1)
        if "left" in hit:
            self.pause_menu.adjust(-1)
        if "right" in hit:
            self.pause_menu.adjust(1)
        if "a" in hit:
            self.menu_action(self.pause_menu.activate())
        elif "b" in hit and self.pause_menu.back():
            self.set_paused(False)

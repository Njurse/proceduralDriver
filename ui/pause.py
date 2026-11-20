"""Pause menu (Esc / Start): Continue, Restart, Adjust seed, Export world
image, Exit.

Restored from the space game's core/pause.py (deleted in d46e8d3): the same
dimmed screen, centre card, title, rows and prompt line in the same colours,
driven by move / adjust / activate / back so keyboard and gamepad share it.
The menu only reports what was chosen; main.DriveSim carries it out.
"""

import random

from direct.gui.OnscreenText import OnscreenText
from panda3d.core import CardMaker, TextNode, TransparencyAttrib

_ACCENT = (0.55, 0.95, 1.00, 1.0)
_DIM = (0.46, 0.58, 0.72, 1.0)
_SELECTED = (0.78, 0.90, 0.98, 1.0)

CONTINUE, RESTART, SEED, EXPORT, EXIT = "continue", "restart", "seed", "export", "exit"
APPLY_SEED = "apply_seed"
_MAIN = [(CONTINUE, "Continue"), (RESTART, "Restart"), (SEED, "Adjust seed"),
         (EXPORT, "Export world image"), (EXIT, "Exit")]
_SEED_ROWS = ["entry", "random", "apply", "back"]
MAX_SEED = (1 << 31) - 1
_ROWS = 5


class PauseMenu:
    def __init__(self, base, current_seed, discovered_count):
        """``current_seed`` and ``discovered_count`` are callables, read
        whenever the menu opens or asks for confirmation."""
        self.current_seed = current_seed
        self.discovered_count = discovered_count
        self.visible = False
        self.screen = "main"
        self.index = 0
        self.entry = ""
        self.pending_seed = None
        self.status = ""

        self.root = base.aspect2d.attachNewNode("pause_root")
        maker = CardMaker("pause_dim")
        maker.setFrame(-4.0, 4.0, -1.4, 1.4)
        dim = self.root.attachNewNode(maker.generate())
        dim.setColor(0.008, 0.012, 0.028, 0.70)
        maker = CardMaker("pause_card")
        maker.setFrame(-0.82, 0.82, -0.56, 0.56)
        card = self.root.attachNewNode(maker.generate())
        card.setColor(0.05, 0.07, 0.12, 0.90)
        for node in (dim, card):
            node.setTransparency(TransparencyAttrib.MAlpha)
            node.setDepthTest(False)
        self.root.setBin("fixed", 100)
        self.root.setLightOff(1)

        def text(pos, scale, color):
            return OnscreenText(text="", parent=self.root, pos=pos, scale=scale, fg=color,
                                align=TextNode.ACenter, shadow=(0, 0, 0, 0.75),
                                mayChange=True)

        self.title = text((0.0, 0.40), 0.085, _ACCENT)
        self.rows = [text((0.0, 0.20 - i * 0.09), 0.055, _DIM) for i in range(_ROWS)]
        self.status_text = text((0.0, -0.33), 0.038, _ACCENT)
        self.prompt = text((0.0, -0.47), 0.032, _DIM)
        self.set_visible(False)

    # ------------------------------------------------------------- state
    def set_visible(self, visible):
        self.visible = bool(visible)
        if self.visible:
            self.screen, self.index, self.status = "main", 0, ""
            self._refresh()
            self.root.show()
        else:
            self.root.hide()

    def set_status(self, text):
        self.status = text
        self._refresh()

    def _items(self):
        if self.screen == "main":
            return [label for _, label in _MAIN]
        if self.screen == "seed":
            return [f"Seed: {self.entry}_", "Random seed", "Apply", "Back"]
        n = self.discovered_count()
        return [f"Yes: seed {self.pending_seed}, forget {n} chunks", "No, keep this world"]

    def _refresh(self):
        titles = {"main": "PAUSED", "seed": f"WORLD SEED (now {self.current_seed()})",
                  "confirm": "START A NEW WORLD?"}
        prompts = {
            "main": "Up/Down: choose   Enter/A: select   Esc/Start: resume",
            "seed": "Type digits or Left/Right   Enter/A: select   Esc/B: back",
            "confirm": "The old save is kept as a copy   Enter/A: select   Esc/B: back",
        }
        self.title.setText(titles[self.screen])
        items = self._items()
        for i, row in enumerate(self.rows):
            if i < len(items):
                row.setText(items[i])
                row.setFg(_SELECTED if i == self.index else _DIM)
                row.show()
            else:
                row.hide()
        self.status_text.setText(self.status)
        self.prompt.setText(prompts[self.screen])

    # ------------------------------------------------------------- input
    def move(self, delta):
        if self.visible:
            self.index = (self.index + int(delta)) % len(self._items())
            self._refresh()

    def adjust(self, delta):
        """Left/Right on the seed entry steps the number."""
        if self.visible and self.screen == "seed" and self.index == 0:
            value = int(self.entry or 0) + int(delta)
            self.entry = str(min(max(value, 0), MAX_SEED))
            self._refresh()

    def type_char(self, ch):
        if self.visible and self.screen == "seed" and ch.isdigit():
            entry = (self.entry + ch).lstrip("0") or "0"
            if int(entry) <= MAX_SEED:
                self.entry = entry
            self.index = 0
            self._refresh()

    def backspace(self):
        if self.visible and self.screen == "seed":
            self.entry = self.entry[:-1]
            self._refresh()

    def back(self):
        """Esc / B: up one screen.  Returns True when the menu should close."""
        if not self.visible:
            return False
        if self.screen == "main":
            return True
        self.screen, self.index = ("seed", 2) if self.screen == "confirm" else ("main", 2)
        self._refresh()
        return False

    def activate(self):
        """Enter / A.  Returns an action: CONTINUE, RESTART, EXPORT, EXIT,
        (APPLY_SEED, seed) or None."""
        if not self.visible:
            return None
        if self.screen == "main":
            action = _MAIN[self.index][0]
            if action != SEED:
                return action
            self.screen, self.index = "seed", 0
            self.entry = str(self.current_seed())
        elif self.screen == "seed":
            row = _SEED_ROWS[self.index]
            if row == "random":
                self.entry = str(random.randrange(MAX_SEED))
            elif row == "apply" and self.entry:
                seed = int(self.entry)
                if seed == self.current_seed():
                    self.status = "That is the current seed"
                else:
                    self.pending_seed = seed
                    self.screen, self.index = "confirm", 1   # default to No
            elif row == "back":
                self.screen, self.index = "main", 2
        else:
            if self.index == 0:
                seed, self.pending_seed = self.pending_seed, None
                self.screen, self.index = "main", 0
                return (APPLY_SEED, seed)
            self.screen, self.index = "seed", 2
        self._refresh()
        return None

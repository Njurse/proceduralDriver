"""Grid-based inventory overlay rendered on aspect2d."""

from panda3d.core import CardMaker, TextNode, TransparencyAttrib

import config
from core.materials import get

_ACCENT = (0.55, 0.95, 1.00, 1.0)
_DIM = (0.46, 0.58, 0.72, 1.0)
_CELL = (0.11, 0.11)


class InventoryOverlay:
    """Renders the player's inventory as a grid of cells."""

    def __init__(self, base):
        self.base = base
        self.visible = False
        self.selected = 0

        self.root = base.aspect2d.attachNewNode("inventory_root")

        maker = CardMaker("inv_dim")
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        self.dim = self.root.attachNewNode(maker.generate())
        self.dim.setColor(0.008, 0.012, 0.028, 0.78)
        self.dim.setTransparency(TransparencyAttrib.MAlpha)
        self.dim.setDepthTest(False)
        self.dim.setBin("background", 70)

        maker = CardMaker("inv_card")
        maker.setFrame(-0.78, 0.78, -0.52, 0.52)
        self.card = self.root.attachNewNode(maker.generate())
        self.card.setColor(0.05, 0.07, 0.12, 0.90)
        self.card.setTransparency(TransparencyAttrib.MAlpha)
        self.card.setDepthTest(False)
        self.card.setBin("background", 71)

        def text(pos, scale, align, color):
            from direct.gui.OnscreenText import OnscreenText
            return OnscreenText(text="", parent=self.root, pos=pos, scale=scale,
                                fg=color, align=align,
                                shadow=(0, 0, 0, 0.75), mayChange=True)

        self.title = text((0.0, 0.44), 0.06, TextNode.ACenter, _ACCENT)
        self.subtitle = text((0.0, 0.37), 0.034, TextNode.ACenter, _DIM)
        self.cols = config.INVENTORY_COLS
        self.rows = config.INVENTORY_ROWS
        self.cells = []
        self.cell_labels = []
        cols, rows = self.cols, self.rows
        for r in range(rows):
            for c in range(cols):
                x = -0.66 + c * 0.22
                z = 0.22 - r * 0.22
                maker = CardMaker(f"cell_{r}_{c}")
                maker.setFrame(-0.095, 0.095, -0.095, 0.095)
                cell = self.root.attachNewNode(maker.generate())
                cell.setPos(x, 0, z)
                cell.setColor(0.10, 0.13, 0.20, 0.85)
                cell.setTransparency(TransparencyAttrib.MAlpha)
                cell.setDepthTest(False)
                cell.setBin("background", 72)
                self.cells.append(cell)
                lbl = text((x, z - 0.02), 0.028, TextNode.ACenter, (1, 1, 1, 1))
                self.cell_labels.append(lbl)
        self.info = text((0.0, -0.42), 0.036, TextNode.ACenter, _DIM)
        self.prompt = text((0.0, -0.47), 0.030, TextNode.ACenter, _DIM)

        self.title.setText("INVENTORY")
        self.prompt.setText("Arrows/WASD - navigate   Esc/I/B - close")
        self.layout(base.getAspectRatio())
        self.set_visible(False)

    def layout(self, aspect):
        """Stretch the dimmer so it covers any window width."""
        half_width = max(float(aspect), 0.5)
        self.dim.setScale(max(1.0, half_width), 1.0, 1.0)

    def set_visible(self, visible, inventory=None):
        self.visible = bool(visible)
        if self.visible:
            self.root.show()
            self.refresh(inventory)
        else:
            self.root.hide()

    def refresh(self, inventory):
        if inventory is None:
            return
        total = inventory.total_items()
        self.subtitle.setText(f"Items: {total}   Value: {inventory.value()}")
        for i, cell in enumerate(self.cells):
            mid = inventory.slots[i] if i < len(inventory.slots) else None
            if mid is None:
                cell.setColor(0.10, 0.13, 0.20, 0.85)
                self.cell_labels[i].setText("")
            else:
                m = get(mid)
                c = m["color"]
                cell.setColor(c[0] * 0.7, c[1] * 0.7, c[2] * 0.7, 0.95)
                n = inventory.counts.get(mid, 0)
                self.cell_labels[i].setText(f"{n}")
            if i == self.selected:
                cell.setColor(0.30, 0.55, 0.85, 0.95)
        # Info line for selected slot.
        mid = inventory.slots[self.selected] if self.selected < len(inventory.slots) else None
        if mid is None:
            self.info.setText("Empty slot")
        else:
            m = get(mid)
            self.info.setText(f"{m['name']}  ({m['tier']})  x{inventory.counts.get(mid, 0)}")

    def move(self, dx, dy):
        cols, rows = self.cols, self.rows
        c = self.selected % cols
        r = self.selected // cols
        c = (c + dx) % cols
        r = (r + dy) % rows
        self.selected = r * cols + c

"""Grid-based inventory for mined materials."""

import config
from core.materials import get


class Inventory:
    """A fixed grid of material slots with stack counts."""

    def __init__(self, cols=None, rows=None):
        if cols is None:
            cols = config.INVENTORY_COLS
        if rows is None:
            rows = config.INVENTORY_ROWS
        self.cols = cols
        self.rows = rows
        self.slots = [None] * (cols * rows)   # material id or None
        self.counts = {}                       # material id -> total count
        self.capacity = cols * rows

    def total_items(self):
        return sum(self.counts.values())

    def add(self, material_id, amount=1):
        if material_id not in self.counts:
            # Need a free slot for a new material.
            if None not in self.slots:
                return False
            idx = self.slots.index(None)
            self.slots[idx] = material_id
        self.counts[material_id] = self.counts.get(material_id, 0) + amount
        return True

    def remove(self, material_id, amount=1):
        if self.counts.get(material_id, 0) < amount:
            return False
        self.counts[material_id] -= amount
        if self.counts[material_id] <= 0:
            del self.counts[material_id]
            if material_id in self.slots:
                self.slots[self.slots.index(material_id)] = None
        return True

    def value(self):
        total = 0
        for mid, n in self.counts.items():
            total += get(mid)["value"] * n
        return total

    def to_dict(self):
        return {"slots": list(self.slots), "counts": dict(self.counts),
                "cols": self.cols, "rows": self.rows}

    @classmethod
    def from_dict(cls, data):
        inv = cls(int(data.get("cols", 6)), int(data.get("rows", 4)))
        slots = data.get("slots", [])
        for i in range(len(inv.slots)):
            inv.slots[i] = slots[i] if i < len(slots) else None
        inv.counts = dict(data.get("counts", {}))
        return inv

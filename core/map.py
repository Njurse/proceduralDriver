"""Top-down galaxy map overlay with search, filtering, zoom/pan and warping."""

import logging
import math

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
from core.textures import soft_dot_texture

log = logging.getLogger("galaxy_explorer.map")

_ACCENT = (0.55, 0.95, 1.00, 1.0)
_DIM = (0.46, 0.58, 0.72, 1.0)


class MapOverlay:
    def __init__(self, base, systems):
        self.base = base
        self.systems = list(systems)
        self.selected = 0
        self.current_index = None
        self.visited = set()
        self.discovered = set()
        self._ship_pos = None
        self._scale = 0.80 / config.GALAXY_RADIUS
        self._dot = soft_dot_texture(64)

        # View state ---------------------------------------------------------
        self.filter_text = ""
        self.search_mode = False
        self.visited_only = False
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        maker = CardMaker("map_dimmer")
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        self.dimmer = base.render2d.attachNewNode(maker.generate())
        self.dimmer.setColor(0.008, 0.012, 0.028, 0.92)
        self.dimmer.setTransparency(TransparencyAttrib.MAlpha)
        self.dimmer.setDepthTest(False)
        self.dimmer.setBin("background", 50)

        self.root = base.aspect2d.attachNewNode("map_root")
        self._build_backdrop()
        self._build_grid()
        self._build_markers()
        self._build_rings()
        self._build_ship_marker()
        self._build_text()
        self._bind_keys()
        self.layout(base.getAspectRatio())

        self.dimmer.hide()
        self.root.hide()

    def layout(self, aspect):
        half_width = max(float(aspect), 0.5)
        margin = clamp(half_width * 0.035, 0.06, 0.16)
        self.title.setPos(-half_width + margin, 0.90)
        self.info.setPos(half_width - margin, 0.52)
        self.search_display.setPos(0.0, 0.90)
        self.hint.setPos(0.0, -0.90)

    # -- construction -------------------------------------------------------
    def _project(self, position):
        return (position.x * self._scale * self.zoom + self.pan_x,
                position.y * self._scale * self.zoom + self.pan_y)

    def _circle(self, radius, color, thickness=1.0):
        lines = LineSegs("map_circle")
        lines.setThickness(thickness)
        lines.setColor(Vec4(*color))
        steps = 64
        for index in range(steps + 1):
            angle = math.tau * index / steps
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            if index == 0:
                lines.moveTo(x, 0.0, y)
            else:
                lines.drawTo(x, 0.0, y)
        node = self.root.attachNewNode(lines.create())
        node.setDepthTest(False)
        return node

    def _soft_card(self, name, scale, color):
        maker = CardMaker(name)
        maker.setFrame(-1.0, 1.0, -1.0, 1.0)
        card = self.root.attachNewNode(maker.generate())
        card.setTexture(self._dot)
        card.setTransparency(TransparencyAttrib.MAlpha)
        card.setDepthTest(False)
        card.setScale(scale)
        card.setColor(*color)
        return card

    def _build_backdrop(self):
        self.core_glow = self._soft_card(
            "map_core_glow", 1.05, (0.22, 0.36, 0.78, 0.30))
        self.core_glow.setPos(0.0, 0.0, 0.0)
        self.core_glow.setBin("background", 51)
        self.core_hot = self._soft_card(
            "map_core_hot", 0.30, (1.0, 0.84, 0.58, 0.45))
        self.core_hot.setPos(0.0, 0.0, 0.0)
        self.core_hot.setBin("background", 52)

    def _build_grid(self):
        for index in range(1, 6):
            self._circle(0.80 * index / 5.0, (0.28, 0.38, 0.58, 0.20))

    def _build_markers(self):
        self.markers = []
        self.labels = []
        for index, system in enumerate(self.systems):
            x, y = self._project(system.position)
            maker = CardMaker(f"marker_{index}")
            maker.setFrame(-1.0, 1.0, -1.0, 1.0)
            card = self.root.attachNewNode(maker.generate())
            card.setTexture(self._dot)
            card.setTransparency(TransparencyAttrib.MAlpha)
            card.setDepthTest(False)
            card.setPos(x, 0.0, y)
            card.setScale(0.050)
            color = system.star_color
            card.setColor(color[0], color[1], color[2], 0.95)
            self.markers.append(card)

            label = OnscreenText(
                text=system.name, parent=self.root, pos=(x, y - 0.074),
                scale=0.030, fg=_DIM, align=TextNode.ACenter,
                shadow=(0, 0, 0, 0.8), mayChange=True)
            self.labels.append(label)

    def _build_rings(self):
        self.selection_ring = self._circle(0.068, (1.0, 1.0, 1.0, 0.95), 2.4)
        self.current_ring = self._circle(0.042, (0.45, 0.98, 1.0, 0.9), 2.0)

    def _build_ship_marker(self):
        self.ship_marker = self._soft_card(
            "map_ship", 0.034, (0.72, 1.0, 0.98, 0.95))
        self.ship_marker.setBin("fixed", 60)
        self.ship_marker.hide()

    def _build_text(self):
        self.title = OnscreenText(
            text="GALAXY MAP", parent=self.root, pos=(-1.72, 0.90),
            scale=0.060, fg=_ACCENT, align=TextNode.ALeft,
            shadow=(0, 0, 0, 0.8))
        self.info = OnscreenText(
            text="", parent=self.root, pos=(1.72, 0.52), scale=0.036,
            fg=_DIM, align=TextNode.ARight, shadow=(0, 0, 0, 0.8),
            mayChange=True)
        self.search_display = OnscreenText(
            text="", parent=self.root, pos=(0.0, 0.90), scale=0.040,
            fg=_ACCENT, align=TextNode.ACenter, shadow=(0, 0, 0, 0.8),
            mayChange=True)
        self.hint = OnscreenText(
            text="", parent=self.root, pos=(0.0, -0.90), scale=0.032,
            fg=_DIM, align=TextNode.ACenter, shadow=(0, 0, 0, 0.8),
            mayChange=True)

    def _bind_keys(self):
        """Search / zoom / pan keys are self-contained while the map is open."""
        for char in "abcdefghijklmnopqrstuvwxyz0123456789":
            self.base.accept(char, self._on_char, [char])
        self.base.accept("backspace", self._on_backspace)
        self.base.accept("slash", self._on_search_toggle)
        self.base.accept("/", self._on_search_toggle)
        self.base.accept("enter", self._on_search_enter)
        self.base.accept("escape", self._on_search_escape)
        self.base.accept("v", self._on_visited_toggle)
        self.base.accept("wheel_up", self.zoom_by, [1.15])
        self.base.accept("wheel_down", self.zoom_by, [1.0 / 1.15])
        self.base.accept("equal", self.zoom_by, [1.15])
        self.base.accept("=", self.zoom_by, [1.15])
        self.base.accept("minus", self.zoom_by, [1.0 / 1.15])
        self.base.accept("-", self.zoom_by, [1.0 / 1.15])
        self.base.accept("home", self.reset_view)
        self.base.accept("shift-arrow_left", self._pan, [-0.10, 0.0])
        self.base.accept("shift-arrow_right", self._pan, [0.10, 0.0])
        self.base.accept("shift-arrow_up", self._pan, [0.0, 0.10])
        self.base.accept("shift-arrow_down", self._pan, [0.0, -0.10])

    # -- key handlers -------------------------------------------------------
    def _on_char(self, char):
        if not self.visible or not self.search_mode:
            return
        if len(self.filter_text) < 24:
            self.filter_text += char
            self._refresh_filtered()

    def _on_backspace(self):
        if not self.visible or not self.search_mode:
            return
        self.filter_text = self.filter_text[:-1]
        self._refresh_filtered()

    def _on_search_toggle(self):
        if not self.visible:
            return
        self.search_mode = not self.search_mode
        self._refresh()

    def _on_search_enter(self):
        if not self.visible or not self.search_mode:
            return
        self.search_mode = False
        visible = self._visible_indices()
        if visible:
            self.selected = visible[0]
        self._refresh()

    def _on_search_escape(self):
        if not self.visible or not self.search_mode:
            return
        self.search_mode = False
        self.filter_text = ""
        self._refresh()

    def _on_visited_toggle(self):
        if not self.visible or self.search_mode:
            return
        self.visited_only = not self.visited_only
        self._refresh_filtered()

    def zoom_by(self, factor):
        if not self.visible:
            return
        self.zoom = clamp(self.zoom * float(factor), 0.20, 5.0)
        self._refresh()

    def _pan(self, dx, dy):
        if not self.visible:
            return
        self.pan_x = clamp(self.pan_x + dx, -2.0, 2.0)
        self.pan_y = clamp(self.pan_y + dy, -2.0, 2.0)
        self._refresh()

    def reset_view(self):
        if not self.visible:
            return
        self.zoom = 1.0
        self.pan_x = self.pan_y = 0.0
        self._refresh()

    # -- state --------------------------------------------------------------
    @property
    def visible(self):
        return not self.root.isHidden()

    def open(self, selected, current_index, ship_pos=None, visited=None,
             discovered=None):
        if not self.systems:
            return
        if selected is None:
            selected = current_index or 0
        self.selected = int(selected) % len(self.systems)
        self.current_index = current_index
        self._ship_pos = ship_pos
        self.visited = set(visited or ())
        self.discovered = set(discovered or ())
        self.filter_text = ""
        self.search_mode = False
        self._ensure_selection_visible()
        self.dimmer.show()
        self.root.show()
        self._refresh()

    def close(self):
        self.dimmer.hide()
        self.root.hide()
        self.search_mode = False

    def handle_escape(self):
        """Consume Esc while searching; return True if search was cancelled."""
        if self.visible and self.search_mode:
            self._on_search_escape()
            return True
        return False

    def update_ship(self, position):
        """Keep the ship marker glued to the flight position while open."""
        self._ship_pos = position
        if not self.systems:
            return
        x, y = self._project(position)
        self.ship_marker.setPos(x, 0.0, y)
        self.ship_marker.show()

    def _visible_indices(self):
        query = self.filter_text.strip().lower()
        result = []
        for index, system in enumerate(self.systems):
            if (self.visited_only and index != self.current_index
                    and index not in self.visited):
                continue
            if query and query not in system.name.lower():
                continue
            result.append(index)
        return result

    def _ensure_selection_visible(self):
        visible = self._visible_indices()
        if visible and self.selected not in visible:
            self.selected = visible[0]

    def _refresh_filtered(self):
        """Re-check selection after the visible set changes, then redraw."""
        self._ensure_selection_visible()
        self._refresh()

    def move(self, delta):
        if not self.systems:
            return
        visible = self._visible_indices()
        if not visible:
            return
        try:
            position = visible.index(self.selected)
        except ValueError:
            position = 0
        self.selected = visible[(position + int(delta)) % len(visible)]
        self._refresh()

    @property
    def selected_index(self):
        return self.selected

    # -- refresh ------------------------------------------------------------
    def _refresh(self):
        if not self.systems:
            return
        visible = set(self._visible_indices())
        query = self.filter_text.strip().lower()
        show_labels = bool(query) or len(visible) <= 12

        # When zoomed out, collapse nearby systems into a single marker.
        clustered = {}
        if self.zoom < config.MAP_CLUSTER_ZOOM:
            for index in visible:
                x, y = self._project(self.systems[index].position)
                key = (round(x / 0.05), round(y / 0.05))
                bucket = clustered.setdefault(key, [])
                bucket.append(index)

        for index, system in enumerate(self.systems):
            marker = self.markers[index]
            label = self.labels[index]
            if index not in visible:
                marker.hide()
                label.hide()
                continue
            x, y = self._project(system.position)
            marker.setPos(x, 0.0, y)
            label.setPos(x, y - 0.074)

            is_important = (index == self.selected or index == self.current_index
                            or index in self.visited)
            if is_important:
                pass
            elif clustered:
                key = (round(x / 0.05), round(y / 0.05))
                bucket = clustered.get(key, [])
                # Only one representative per screen cell when zoomed out.
                if index != bucket[0]:
                    marker.hide()
                    label.hide()
                    continue

            base_color = system.star_color
            if index == self.selected:
                marker.setColor(1.0, 1.0, 1.0, 1.0)
                marker.setScale(0.072)
                label.setFg(Vec4(1.0, 1.0, 1.0, 1.0))
                label.setScale(0.036)
            elif index == self.current_index:
                marker.setColor(0.55, 0.98, 1.0, 1.0)
                marker.setScale(0.058)
                label.setFg(Vec4(0.55, 0.98, 1.0, 0.95))
                label.setScale(0.032)
            else:
                marker.setColor(base_color[0], base_color[1],
                                base_color[2], 0.85)
                marker.setScale(0.048)
                label.setFg(Vec4(*_DIM[:3], 0.72))
                label.setScale(0.029)
            marker.show()
            if show_labels or is_important:
                label.show()
            else:
                label.hide()

        system = self.systems[self.selected]
        x, y = self._project(system.position)
        self.selection_ring.setPos(x, 0.0, y)
        self.selection_ring.show()
        if self.current_index is not None:
            cx, cy = self._project(
                self.systems[self.current_index].position)
            self.current_ring.setPos(cx, 0.0, cy)
            self.current_ring.show()
        else:
            self.current_ring.hide()
        if self._ship_pos is not None:
            sx, sy = self._project(self._ship_pos)
            self.ship_marker.setPos(sx, 0.0, sy)
            self.ship_marker.show()
        else:
            self.ship_marker.hide()

        planet_count = len(system.planets)
        key = system.name
        if self.selected == self.current_index:
            status = "CURRENT SYSTEM"
        elif key in self.visited:
            status = "VISITED"
        else:
            status = "UNEXPLORED"
        lines = [system.name.upper(), system.star_type,
                 f"{planet_count} planets", status,
                 f"ZOOM {self.zoom:.2f}x"]
        if self._ship_pos is not None:
            distance = (system.position - self._ship_pos).length()
            lines.append(f"{distance:.0f} u away")
        self.info.setText("\n".join(lines))

        if self.search_mode:
            self.search_display.setText(f"SEARCH: {self.filter_text}_")
        elif self.filter_text:
            self.search_display.setText(f"FILTER: {self.filter_text}")
        else:
            self.search_display.setText("")
        self.hint.setText(
            "A/D or arrows select   / search   V visited only   "
            "wheel or +/- zoom   Shift+arrows pan   Home reset   "
            "Enter warp   M/Esc close")

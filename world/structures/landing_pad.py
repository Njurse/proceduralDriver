"""Landing pad convention: pad disc, painted ring, pulsing guidance cone.

A pad lives in a parent's local frame (for example inside a freighter
hangar) and reports its world pose through the live scene-graph transform,
so the docking autopilot can target it while the parent holds position.
"""

import math

from panda3d.core import LineSegs

import config
from core.geometry import additive_blend, make_cone, make_uv_sphere

_CONE_COLOR = (0.25, 1.00, 0.42)


class LandingPad:
    """A pad the player's ship can lock onto, with a green guidance beam.

    The beam is a translucent additive cone (apex at the pad, opening
    upward) that pulses while the player is nearby and hides once docking
    starts.  ``panda3d.core`` has no ``Cone`` primitive, so the cone is
    built by ``core.geometry.make_cone``.
    """

    def __init__(self, parent, local_position):
        self.root = parent.attachNewNode("landing_pad")
        self.root.setPos(local_position)
        self.radius = config.PAD_RADIUS

        # Pad surface: a flattened dark disc.
        disc = make_uv_sphere(self.radius, 10, 22, "pad_disc")
        disc.setScale(1.0, 1.0, 0.06)
        disc.setColor(0.09, 0.12, 0.16, 1.0)
        disc.setLightOff(1)
        disc.setMaterialOff(1)
        disc.reparentTo(self.root)

        # Green landing ring painted just above the pad surface.
        ring = LineSegs("pad_ring")
        ring.setThickness(2.5)
        ring.setColor(0.30, 1.0, 0.45, 1.0)
        steps = 40
        lift = self.radius * 0.06 + 0.05
        for index in range(steps + 1):
            angle = math.tau * index / steps
            point = (self.radius * 0.86 * math.cos(angle),
                     self.radius * 0.86 * math.sin(angle), lift)
            if index == 0:
                ring.moveTo(*point)
            else:
                ring.drawTo(*point)
        ring_node = self.root.attachNewNode(ring.create())
        ring_node.setLightOff(1)
        ring_node.setDepthWrite(False)
        ring_node.setBin("transparent", 24)

        # Guidance beam: additive cone widening upward from the pad centre.
        self.cone = make_cone(config.PAD_CONE_RADIUS, config.PAD_CONE_HEIGHT,
                              name="pad_cone")
        additive_blend(self.cone)
        self.cone.setDepthWrite(False)
        self.cone.setBin("transparent", 25)
        self.cone.reparentTo(self.root)
        self._pulse = 0.0
        self.set_cone_active(False)

    def set_cone_active(self, active):
        """Show/hide the pulsing guidance beam."""
        if active:
            self.cone.show()
        else:
            self.cone.hide()

    def update(self, dt):
        """Pulse the guidance beam's colour and width."""
        self._pulse += dt
        wave = 0.5 + 0.5 * math.sin(math.tau * 1.1 * self._pulse)
        glow = 0.35 + 0.65 * wave
        self.cone.setColor(_CONE_COLOR[0] * glow, _CONE_COLOR[1] * glow,
                           _CONE_COLOR[2] * glow, 0.10 + 0.22 * wave)
        self.cone.setScale(0.82 + 0.18 * wave)

    def world_position(self, reference):
        """World-space pad centre (``reference`` is usually ``base.render``)."""
        return self.root.getPos(reference)

    def world_quat(self, reference):
        """World-space pad orientation (nose points along the pad's +Y)."""
        return self.root.getQuat(reference)

    def destroy(self):
        try:
            self.root.removeNode()
        except Exception:
            pass

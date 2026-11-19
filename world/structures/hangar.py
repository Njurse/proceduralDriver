"""Procedural freighter hangar: interior room, glowing entry, docking autopilot.

The hangar is built as a child of a freighter's ship root, so it inherits the
freighter's world transform and scale.  The local frame runs along the
freighter's Y axis with the glowing blue entry opening toward +Y (the bow):
fly into the entry rectangle and you are inside the hangar room, where a
landing pad waits with its pulsing green guidance cone.

The docking/takeoff autopilots operate on a ``SpaceFlight`` controller in
world space using the freighter's live transform, so the freighter must hold
position (``TrafficShip.docked``) while a ship is docking or launching.
"""

import math

from panda3d.core import LineSegs, Quat, Vec3

import config
from core.flight import _quat_nlerp
from core.geometry import additive_blend, make_quad
from core.mathutil import clamp, damp_vec
from world.structures.landing_pad import LandingPad

_HULL = (0.16, 0.19, 0.24, 1.0)
_FLOOR = (0.12, 0.14, 0.18, 1.0)


def _align_quat_to_vector(quat, direction, max_angle=None):
    """Rotate quat toward direction, limited to max_angle radians this step.

    Local copy of the traffic helper (kept here to avoid a circular import
    between ``world.traffic`` and this module).
    """
    if direction.length() < 1e-6:
        return quat
    d = Vec3(direction)
    d.normalize()
    fwd = quat.getForward()
    if fwd.length() < 1e-6:
        fwd = Vec3(0, 1, 0)
    dot = clamp(fwd.dot(d), -1.0, 1.0)
    angle = math.acos(dot)
    if max_angle is not None:
        angle = min(angle, max(0.0, max_angle))
    if angle < 1e-9:
        return quat
    axis = fwd.cross(d)
    if axis.length() < 1e-6:
        axis = quat.getRight()
    axis.normalize()
    delta = Quat()
    delta.setFromAxisAngleRad(angle, axis)
    result = delta * quat
    result.normalize()
    return result


class Hangar:
    """A dockable hangar bay attached to a freighter (or any ship root)."""

    def __init__(self, base, ship_root, owner):
        self.base = base
        self.owner = owner          # TrafficShip that carries this hangar
        self.root = ship_root.attachNewNode("hangar")

        w = config.HANGAR_WIDTH
        h = config.HANGAR_HEIGHT
        l = config.HANGAR_LENGTH
        self.half_w = w * 0.5
        self.half_h = h * 0.5
        self.half_l = l * 0.5

        # Mount the bay pod under the hull (top edge overlapping by 2 units)
        # so the glowing entry rectangle is visible from outside the hull.
        drop = config.FREIGHTER_HULL_HALF_H + self.half_h - 2.0
        self.root.setPos(0.0, 0.0, -drop)

        self._build_room(w, h, l)
        self._build_entry_glow(w, h, l)
        self.pad = LandingPad(self.root, Vec3(
            config.PAD_LOCAL[0], config.PAD_LOCAL[1], -self.half_h))

        self.dock_phase = None      # None|descent|settling|aligning|docked
        self.takeoff_phase = None   # None|rise|exit
        self.settle_timer = 0.0
        self.rotate_timer = 0.0
        self.dock_start_quat = None
        self.exit_speed = 0.0
        self._time = 0.0

    # -- construction --------------------------------------------------------
    def _build_room(self, w, h, l):
        hw, hl = self.half_w, self.half_l
        floor_z, ceil_z = -self.half_h, self.half_h

        def quad(corners, color, name):
            node = make_quad(corners, color, name)
            node.reparentTo(self.root)
            return node

        # Floor and ceiling (horizontal X-Y quads).
        quad([(-hw, -hl, floor_z), (hw, -hl, floor_z),
              (hw, hl, floor_z), (-hw, hl, floor_z)], _FLOOR, "hangar_floor")
        quad([(-hw, -hl, ceil_z), (hw, -hl, ceil_z),
              (hw, hl, ceil_z), (-hw, hl, ceil_z)], _HULL, "hangar_ceiling")
        # Back wall (y = -hl); the entry end (+hl) stays open.
        quad([(-hw, -hl, floor_z), (hw, -hl, floor_z),
              (hw, -hl, ceil_z), (-hw, -hl, ceil_z)], _HULL, "hangar_back")
        # Side walls (x = +/-hw).
        quad([(-hw, -hl, floor_z), (-hw, hl, floor_z),
              (-hw, hl, ceil_z), (-hw, -hl, ceil_z)], _HULL, "hangar_left")
        quad([(hw, -hl, floor_z), (hw, hl, floor_z),
              (hw, hl, ceil_z), (hw, -hl, ceil_z)], _HULL, "hangar_right")
        # Ceiling light strips for a hangar-bay feel.
        strip_w = w * 0.16
        strip_z = ceil_z - 0.35
        for offset in (-w * 0.24, w * 0.24):
            strip = quad([
                (offset - strip_w, -hl * 0.55, strip_z),
                (offset + strip_w, -hl * 0.55, strip_z),
                (offset + strip_w, hl * 0.55, strip_z),
                (offset - strip_w, hl * 0.55, strip_z)],
                (0.72, 0.84, 1.0, 1.0), "hangar_strip")
            additive_blend(strip)
            strip.setDepthWrite(False)
            strip.setBin("transparent", 26)

        # Empty marker node at the entry plane for world-space queries.
        self.entry_node = self.root.attachNewNode("hangar_entry")
        self.entry_node.setPos(0.0, hl, 0.0)

    def _build_entry_glow(self, w, h, l):
        """The humming glowing blue rectangle framing the hangar opening."""
        hw, fh = self.half_w, self.half_h
        y = self.half_l

        frame = LineSegs("hangar_entry_frame")
        frame.setThickness(5.0)
        frame.moveTo(-hw, y, -fh)
        frame.drawTo(hw, y, -fh)
        frame.drawTo(hw, y, fh)
        frame.drawTo(-hw, y, fh)
        frame.drawTo(-hw, y, -fh)
        self.entry_frame = self.root.attachNewNode(frame.create())
        self.entry_frame.setLightOff(1)
        self.entry_frame.setDepthWrite(False)
        self.entry_frame.setBin("transparent", 27)

        inset = 1.2
        self.entry_fill = make_quad(
            [(-hw + inset, y, -fh + inset), (hw - inset, y, -fh + inset),
             (hw - inset, y, fh - inset), (-hw + inset, y, fh - inset)],
            (0.30, 0.55, 1.0, 0.16), "hangar_entry_fill")
        additive_blend(self.entry_fill)
        self.entry_fill.setDepthWrite(False)
        self.entry_fill.setBin("transparent", 28)
        self.entry_fill.reparentTo(self.root)

    # -- coordinate helpers --------------------------------------------------
    def to_local(self, world_point):
        """World point -> hangar-local coordinates (scale/rotation aware)."""
        origin = self.root.getPos(self.base.render)
        quat = self.root.getQuat(self.base.render)
        scale = max(self.root.getScale().x, 1e-6)
        return quat.conjugate().xform(Vec3(world_point) - origin) / scale

    def is_inside(self, world_point):
        """Whether the world point is within the hangar room volume."""
        p = self.to_local(world_point)
        return (abs(p.x) <= self.half_w
                and abs(p.z) <= self.half_h
                and -self.half_l <= p.y <= self.half_l)

    def entry_world(self):
        """World position of the centre of the glowing entry rectangle."""
        return self.entry_node.getPos(self.base.render)

    def exit_direction(self):
        """World-space direction pointing out through the entry."""
        return self.root.getQuat(self.base.render).getForward()

    def entry_distance(self, world_point):
        return (Vec3(world_point) - self.entry_world()).length()

    def hum_level(self, world_point):
        """0..1 proximity level for the hangar hum loop."""
        if self.is_inside(world_point):
            return 1.0
        level = 1.0 - self.entry_distance(world_point) / config.HANGAR_HUM_RANGE
        return max(0.0, level) ** 1.4

    # -- per-frame -----------------------------------------------------------
    def update(self, dt, player_world):
        """Pulse the entry glow and pad cone; apply distance LOD."""
        self._time += dt
        pulse = 0.5 + 0.5 * math.sin(math.tau * 1.6 * self._time)
        self.entry_frame.setColor(0.34 + 0.40 * pulse, 0.64 + 0.26 * pulse,
                                  1.0, 1.0)
        self.entry_fill.setColor(0.30, 0.55, 1.0, 0.12 + 0.20 * pulse)

        docking = self.dock_phase is not None or self.takeoff_phase is not None
        pad_pos = self.pad.world_position(self.base.render)
        near_pad = ((Vec3(player_world) - pad_pos).length()
                    <= config.PAD_PROMPT_RANGE * 1.5)
        self.pad.set_cone_active((self.is_inside(player_world) or near_pad)
                                 and not docking)
        if not docking:
            self.pad.update(dt)

        freighter_pos = self.owner.ship.root.getPos(self.base.render)
        if ((Vec3(player_world) - freighter_pos).length()
                > config.HANGAR_LOD_DIST):
            self.root.hide()
        else:
            self.root.show()

    def prompt_state(self, world_point):
        """Context prompt for the player position, or None when out of range."""
        if self.is_inside(world_point):
            pad_pos = self.pad.world_position(self.base.render)
            if ((Vec3(world_point) - pad_pos).length()
                    <= config.PAD_PROMPT_RANGE):
                return "X/F  -  land on the pad"
            return "Hangar bay  -  approach the landing pad"
        if self.entry_distance(world_point) <= config.HANGAR_ENTRY_RANGE:
            return "Fly into the hangar bay glow"
        return None

    # -- docking autopilot ---------------------------------------------------
    def begin_docking(self):
        """Lock the ship onto the pad coordinates and start the descent."""
        self.dock_phase = "descent"
        self.settle_timer = 0.0
        self.rotate_timer = 0.0
        self.dock_start_quat = None
        self.takeoff_phase = None

    def dock_update(self, controller, dt):
        """Autopilot onto the pad; returns the current dock phase.

        Phases: ``descent`` (slide onto the pad coordinates and settle),
        ``settling`` (hold still for half a second), ``aligning`` (smoothly
        lerp the ship's orientation onto the pad orientation), ``docked``.
        """
        pad_pos = self.pad.world_position(self.base.render)
        pad_up = self.pad.world_quat(self.base.render).getUp()
        target = pad_pos + pad_up * config.LAND_CLEARANCE
        controller.throttle = 0.0

        if self.dock_phase == "descent":
            controller.position = damp_vec(controller.position, target,
                                           config.PAD_DESCENT_RATE, dt)
            controller.velocity = Vec3(0.0)
            controller.speed = 0.0
            if (controller.position - target).length() < 0.8:
                self.dock_phase = "settling"
                self.settle_timer = 0.0
        elif self.dock_phase == "settling":
            controller.position = damp_vec(controller.position, target, 8.0, dt)
            controller.velocity = Vec3(0.0)
            controller.speed = 0.0
            self.settle_timer += dt
            if self.settle_timer >= config.PAD_SETTLE_SECONDS:
                self.dock_phase = "aligning"
                self.rotate_timer = 0.0
        elif self.dock_phase == "aligning":
            self.rotate_timer += dt
            if self.dock_start_quat is None:
                self.dock_start_quat = Quat(controller.quat)
            pad_quat = self.pad.world_quat(self.base.render)
            # Deterministic smoothstep from the ship's landing approach
            # orientation onto the pad orientation: exact at the end.
            t = clamp(self.rotate_timer / config.PAD_ROTATE_SECONDS, 0.0, 1.0)
            ease = t * t * (3.0 - 2.0 * t)
            controller.quat = _quat_nlerp(self.dock_start_quat, pad_quat, ease)
            controller.velocity = Vec3(0.0)
            controller.speed = 0.0
            if self.rotate_timer >= config.PAD_ROTATE_SECONDS:
                self.dock_phase = "docked"
        return self.dock_phase

    # -- takeoff autopilot ---------------------------------------------------
    def begin_takeoff(self):
        """Raise the ship off the pad and fly it out through the entry."""
        self.takeoff_phase = "rise"
        self.exit_speed = 0.0
        self.dock_phase = None

    def takeoff_update(self, controller, dt):
        """Autopilot out of the hangar; returns True when control is handed back.

        Phase ``rise`` climbs vertically off the pad; phase ``exit`` accelerates
        along the exit direction through the glowing entry.  Once the ship is
        ``HANGAR_EXIT_CLEAR`` beyond the entry plane, control returns.
        """
        pad_pos = self.pad.world_position(self.base.render)
        pad_up = self.pad.world_quat(self.base.render).getUp()

        if self.takeoff_phase == "rise":
            hover = pad_pos + pad_up * (config.PAD_HOVER_HEIGHT
                                        + config.TAKEOFF_RISE_HEIGHT)
            controller.position = damp_vec(controller.position, hover, 2.0, dt)
            controller.velocity = pad_up * config.TAKEOFF_CLIMB_RATE
            controller.speed = controller.velocity.length()
            controller.throttle = 0.4
            if (controller.position - hover).length() < 2.0:
                self.takeoff_phase = "exit"
            return False

        entry = self.entry_world()
        exit_dir = self.exit_direction()
        self.exit_speed = min(
            self.exit_speed + config.TAKEOFF_EXIT_SPEED
            * config.TAKEOFF_ACCEL * dt, config.TAKEOFF_EXIT_SPEED)
        controller.velocity = exit_dir * self.exit_speed
        controller.position += controller.velocity * dt
        controller.speed = self.exit_speed
        controller.throttle = 0.5
        controller.quat = _align_quat_to_vector(controller.quat, exit_dir,
                                                1.2 * dt)
        progress = (controller.position - entry).dot(exit_dir)
        if progress >= config.HANGAR_EXIT_CLEAR:
            controller.throttle = 0.5
            return True
        return False

    def destroy(self):
        try:
            self.root.removeNode()
        except Exception:
            pass

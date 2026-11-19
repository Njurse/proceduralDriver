"""Space and atmospheric flight controllers.

Both controllers expose ``position``, ``quat``, ``throttle`` and ``speed`` so
the game can swap modes without touching the renderable ship.
"""

import math

from panda3d.core import Quat, Vec3

import config
from core.mathutil import clamp, damp, damp_vec

_X = Vec3(1.0, 0.0, 0.0)
_Z = Vec3(0.0, 0.0, 1.0)


def _local_rotation(quat, angle, axis):
    if abs(angle) < 1e-9:
        return quat
    delta = Quat()
    delta.setFromAxisAngleRad(angle, axis)
    result = quat * delta
    result.normalize()
    return result


def _body_rotation(quat, delta_body, max_pitch_deg=None):
    """Apply a body-frame angular displacement as one rotation.

    ``delta_body`` is ``(pitch about +X, roll about +Y, yaw about -Z)`` in
    radians.  The delta is composed in the ship's local frame, so every control
    axis stays locked to the hull no matter which way the ship is pointing.
    Integrating a single axis-angle instead of sequential Euler rotations also
    removes axis coupling and gimbal lock.

    ``max_pitch_deg`` optionally stops the nose from passing vertical, which is
    useful for the plane-style atmosphere model.  ``Quat.getHpr()`` reports
    degrees in Panda3D, so the limit is compared in degrees.
    """
    delta = Vec3(delta_body)
    if delta.length() < 1e-9:
        return quat
    if max_pitch_deg is not None:
        pitch = quat.getHpr().y
        too_high = pitch > max_pitch_deg and delta.x > 0.0
        too_low = pitch < -max_pitch_deg and delta.x < 0.0
        if too_high or too_low:
            delta.x = 0.0
        if delta.length() < 1e-9:
            return quat
    magnitude = delta.length()
    step = Quat()
    step.setFromAxisAngleRad(magnitude, delta / magnitude)
    result = quat * step
    result.normalize()
    return result


def _analogue_gain(inp, sensitivity):
    """Steering gain for gamepad sticks (50x base, scaled by the runtime slider).

    Keyboard/digital input keeps the unscaled rates so only analogue control is
    boosted.
    """
    if getattr(inp, "gamepad_connected", False):
        return config.ANALOGUE_SENSITIVITY * max(float(sensitivity), 0.0)
    return 1.0


def _quat_dot(a, b):
    return (a.getR() * b.getR() + a.getI() * b.getI()
            + a.getJ() * b.getJ() + a.getK() * b.getK())


def _quat_nlerp(current, target, t):
    """Shortest-path normalized quaternion blend (lerp-style)."""
    t = max(0.0, min(1.0, t))
    sign = -1.0 if _quat_dot(current, target) < 0.0 else 1.0
    result = Quat(
        current.getR() + (sign * target.getR() - current.getR()) * t,
        current.getI() + (sign * target.getI() - current.getI()) * t,
        current.getJ() + (sign * target.getJ() - current.getJ()) * t,
        current.getK() + (sign * target.getK() - current.getK()) * t,
    )
    result.normalize()
    return result


def _surface_normal(sampler, x, y, eps=2.0):
    """Approximate the heightfield normal at (x, y) via central differences."""
    dx = (float(sampler(x + eps, y)) - float(sampler(x - eps, y))) / (2.0 * eps)
    dy = (float(sampler(x, y + eps)) - float(sampler(x, y - eps))) / (2.0 * eps)
    normal = Vec3(-dx, -dy, 1.0)
    normal.normalize()
    return normal


def _align_up(quat, target_up):
    """Return a quat rotated minimally so its up axis matches ``target_up``."""
    target = Vec3(target_up)
    target.normalize()
    up = quat.getUp()
    dot = clamp(up.dot(target), -1.0, 1.0)
    if dot > 0.99999:
        return Quat(quat)
    axis = up.cross(target)
    if axis.length() < 1e-6:
        # Opposite directions: rotate 180 degrees about the ship's right axis.
        axis = quat.getRight()
    axis.normalize()
    delta = Quat()
    delta.setFromAxisAngleRad(math.acos(dot), axis)
    result = delta * quat
    result.normalize()
    return result


class SpaceFlight:
    """Arcade spaceflight with persistent throttle and inertia."""

    def __init__(self):
        self.position = Vec3(0.0, -config.GALAXY_START_DIST, 0.0)
        self.quat = Quat()
        self.throttle = 0.0
        self.velocity = Vec3(0.0)
        self.speed = 0.0
        self._pitch_cmd = 0.0
        self._yaw_cmd = 0.0
        self._roll_cmd = 0.0

    def reset(self, position=None, hpr=None, throttle=0.0):
        if position is not None:
            self.position = Vec3(position)
        self.quat = Quat()
        if hpr is not None:
            self.quat.setHpr(hpr)
        self.throttle = clamp(throttle, -config.SPACE_REVERSE, 1.0)
        self.velocity = self.quat.getForward() * (
            self.throttle * config.SPACE_MAX_SPEED)
        self.speed = self.velocity.length()
        self._pitch_cmd = self._yaw_cmd = self._roll_cmd = 0.0

    def update(self, dt, inp, mouse_pitch=0.0, mouse_yaw=0.0,
               sensitivity=1.0):
        throttle_axis = inp.throttle_axis
        if inp.brake and throttle_axis >= 0.0:
            throttle_axis = -1.0
        self.throttle = clamp(
            self.throttle + throttle_axis * config.SPACE_THROTTLE_RATE * dt,
            -config.SPACE_REVERSE, 1.0)

        self._pitch_cmd = damp(self._pitch_cmd, inp.pitch_axis, 12.0, dt)
        self._yaw_cmd = damp(self._yaw_cmd, inp.yaw_axis, 12.0, dt)
        self._roll_cmd = damp(
            self._roll_cmd, clamp(inp.lateral_axis + inp.pad_roll_axis,
                                  -1.0, 1.0), 12.0, dt)
        # Angular deadband so released controls settle instead of wobbling.
        if abs(self._pitch_cmd) < 0.02:
            self._pitch_cmd = 0.0
        if abs(self._yaw_cmd) < 0.02:
            self._yaw_cmd = 0.0
        if abs(self._roll_cmd) < 0.02:
            self._roll_cmd = 0.0

        gain = _analogue_gain(inp, sensitivity)
        # Every control (keyboard, stick and mouse) contributes to one
        # body-frame angular displacement.  Mouse movement is a direct angle
        # step while analogue input is a rate multiplied by dt.
        delta = Vec3(
            self._pitch_cmd * config.SPACE_PITCH_RATE * gain * dt
            + mouse_pitch * config.MOUSE_GAIN_SPACE,
            self._roll_cmd * config.SPACE_ROLL_RATE * gain * dt,
            -(self._yaw_cmd * config.SPACE_YAW_RATE * gain * dt
              + mouse_yaw * config.MOUSE_GAIN_SPACE),
        )
        # Space is free 6DOF: no pitch clamp, so the ship can loop and roll
        # without ever hitting a control singularity.
        self.quat = _body_rotation(self.quat, delta)

        boosting = inp.boost and self.throttle > 0.05
        multiplier = config.SPACE_BOOST_MULT if boosting else 1.0
        target = self.quat.getForward() * (
            self.throttle * config.SPACE_MAX_SPEED * multiplier)
        smooth = config.SPACE_SPEED_SMOOTH * (1.6 if boosting else 1.0)
        self.velocity = damp_vec(self.velocity, target, smooth, dt)
        if inp.brake:
            self.velocity *= math.exp(-2.2 * dt)
        self.position += self.velocity * dt
        self.speed = self.velocity.length()

        # Soft galaxy boundary so the player never flies into empty infinity.
        radius = math.hypot(self.position.x, self.position.y)
        limit = config.GALAXY_BOUNDARY
        if radius > limit:
            scale = limit / radius
            self.position.x *= scale
            self.position.y *= scale
            self.velocity *= 0.4
        return boosting


class AtmosphereFlight:
    """Plane-style controller: pitch, banked turns and rudder, no strafing."""

    def __init__(self):
        self.position = Vec3(0.0, -320.0, config.ATMO_ENTRY_ALTITUDE)
        self.quat = Quat()
        self.speed = config.ATMO_CRUISE
        self.throttle = 0.55
        self.bank = 0.0
        self._pitch_cmd = 0.0
        self._yaw_cmd = 0.0
        self._roll_cmd = 0.0
        self._boosting = False
        # Landing/liftoff assist state (triggered by the X button / F key).
        self.assist = None       # None, "land", or "takeoff"
        self.landed = False
        self.land_point = None

    def reset(self, position, hpr=None, throttle=0.55):
        self.position = Vec3(position)
        self.quat = Quat()
        if hpr is not None:
            self.quat.setHpr(hpr)
        self.throttle = clamp(throttle, 0.0, 1.0)
        self.speed = config.ATMO_CRUISE
        self.bank = 0.0
        self._pitch_cmd = self._yaw_cmd = self._roll_cmd = 0.0
        self._boosting = False
        self.assist = None
        self.landed = False
        self.land_point = None

    def update(self, dt, inp, mouse_pitch=0.0, mouse_roll=0.0,
               height_sampler=None, sensitivity=1.0):
        # Landing / liftoff assist takes full control of the angles so manual
        # steering cannot fight it (no conflicting rotations in the same frame).
        if height_sampler is None and self.assist is not None:
            self.assist = None
            self.landed = False
        if height_sampler is not None:
            throttle_held = float(getattr(inp, "throttle_axis", 0.0)) > 0.05
            if getattr(inp, "land", False):
                self._handle_land_input(inp, height_sampler)
            elif throttle_held and (self.landed or self.assist == "land"):
                # Holding throttle alone reverses an active/completed landing.
                self._begin_takeoff()
        if self.assist == "land":
            self._update_landing(dt, height_sampler)
            return False
        if self.assist == "takeoff":
            self._update_takeoff(dt, height_sampler)
            return False

        self.throttle = clamp(
            self.throttle + inp.throttle_axis * config.ATMO_THROTTLE_RATE * dt,
            0.0, 1.0)
        if inp.brake:
            self.throttle = 0.0

        # Pitch = Up/Down (or stick Y); yaw = Left/Right (or stick X);
        # roll = A/D (or stick X).  No strafing: left/right is roll.
        self._pitch_cmd = damp(self._pitch_cmd, inp.pitch_axis, 10.0, dt)
        self._yaw_cmd = damp(self._yaw_cmd, inp.yaw_axis, 10.0, dt)
        self._roll_cmd = damp(
            self._roll_cmd,
            clamp(inp.lateral_axis + inp.pad_roll_axis, -1.0, 1.0),
            10.0, dt)
        # Angular deadband so released controls settle instead of wobbling.
        if abs(self._pitch_cmd) < 0.02:
            self._pitch_cmd = 0.0
        if abs(self._yaw_cmd) < 0.02:
            self._yaw_cmd = 0.0
        if abs(self._roll_cmd) < 0.02:
            self._roll_cmd = 0.0

        right = self.quat.getRight()
        up = self.quat.getUp()
        self.bank = math.atan2(-right.z, up.z)

        gain = _analogue_gain(inp, sensitivity)
        # Body-frame angular displacement: manual pitch/roll/yaw plus the
        # banked-turn yaw and gentle auto-level, all integrated together.
        delta = Vec3(
            self._pitch_cmd * config.ATMO_PITCH_RATE * gain * dt
            + mouse_pitch * config.MOUSE_GAIN_ATMO,
            self._roll_cmd * config.ATMO_ROLL_RATE * gain * dt
            + mouse_roll * config.MOUSE_GAIN_ATMO,
            -(self._yaw_cmd * config.ATMO_YAW_RATE * gain * dt
              + config.ATMO_BANK_TURN * self.bank * dt),
        )
        if abs(self._roll_cmd) < 0.12 and abs(self.bank) < 1.1:
            delta.y += -self.bank * config.ATMO_AUTO_LEVEL * dt
        # Atmosphere is plane-like: keep the nose below vertical (78 degrees).
        self.quat = _body_rotation(self.quat, delta, max_pitch_deg=78.0)

        forward = self.quat.getForward()
        boosting = inp.boost
        target_speed = (config.ATMO_MIN_SPEED
                        + (config.ATMO_MAX_SPEED - config.ATMO_MIN_SPEED)
                        * self.throttle)
        if boosting:
            target_speed *= 1.22
        self.speed = damp(self.speed, target_speed, config.ATMO_SPEED_SMOOTH,
                          dt)
        self.speed -= forward.z * 16.0 * dt
        self.speed = clamp(self.speed, 12.0, config.ATMO_MAX_SPEED * 1.3)

        self.position += forward * self.speed * dt
        slow = 1.0 - clamp(self.speed / config.ATMO_MAX_SPEED, 0.0, 1.0)
        self.position.z -= config.ATMO_SINK * slow * 38.0 * dt

        if height_sampler is not None:
            ground = float(height_sampler(self.position.x, self.position.y))
            floor = ground + config.ATMO_TERRAIN_CLEARANCE
            if self.position.z < floor:
                penetration = floor - self.position.z
                self.position.z = floor
                # Gentle nose-up nudge; this is a body-frame angle in radians.
                lift = min(0.010, 0.00021 * penetration + 0.00007)
                self.quat = _local_rotation(self.quat, lift, _X)
                self.speed *= 0.82
        ceiling = config.ATMO_CEILING * 1.15
        if self.position.z > ceiling:
            self.position.z = ceiling
        return boosting

    # -- landing / liftoff assist -------------------------------------------
    def _handle_land_input(self, inp, sampler):
        ground = float(sampler(self.position.x, self.position.y))
        altitude = self.position.z - ground
        throttle_held = float(getattr(inp, "throttle_axis", 0.0)) > 0.05

        if self.assist == "takeoff":
            return
        if self.assist == "land" or self.landed:
            if throttle_held:
                self._begin_takeoff()
            return
        if throttle_held:
            if altitude <= config.LAND_MAX_ALTITUDE * 1.5:
                self._begin_takeoff()
            return
        if altitude <= config.LAND_MAX_ALTITUDE:
            self._begin_landing()

    def _begin_landing(self):
        self.assist = "land"
        self.landed = False
        self.land_point = Vec3(self.position.x, self.position.y, 0.0)
        self.speed = 0.0
        self.throttle = 0.0
        self._pitch_cmd = self._yaw_cmd = self._roll_cmd = 0.0

    def _begin_takeoff(self):
        self.assist = "takeoff"
        self.landed = False
        self.land_point = None
        self.throttle = max(self.throttle, 0.45)
        self.speed = max(self.speed, config.ATMO_MIN_SPEED)
        self._pitch_cmd = self._yaw_cmd = self._roll_cmd = 0.0

    def _update_landing(self, dt, sampler):
        if sampler is None or self.land_point is None:
            self.assist = None
            return
        px, py = self.land_point.x, self.land_point.y
        ground = float(sampler(px, py))
        target_z = ground + config.LAND_CLEARANCE
        rate = config.LAND_DESCENT_SPEED
        self.position.x = damp(self.position.x, px, rate, dt)
        self.position.y = damp(self.position.y, py, rate, dt)
        self.position.z = damp(self.position.z, target_z, rate, dt)
        normal = _surface_normal(sampler, px, py)
        target_quat = _align_up(self.quat, normal)
        t = 1.0 - math.exp(-config.LAND_ALIGN_SPEED * dt)
        self.quat = _quat_nlerp(self.quat, target_quat, t)
        self.speed = damp(self.speed, 0.0, config.LAND_BRAKE_SPEED, dt)
        if abs(self.position.z - target_z) < 0.4:
            self.landed = True

    def _update_takeoff(self, dt, sampler):
        ground = float(sampler(self.position.x, self.position.y))
        altitude = self.position.z - ground
        # Blend from the landing normal toward level flight as we climb.
        progress = clamp(altitude / max(config.TAKEOFF_ALTITUDE, 1.0), 0.0, 1.0)
        normal = _surface_normal(sampler, self.position.x, self.position.y)
        target_up = normal * (1.0 - progress) + _Z * progress
        target_up.normalize()
        target_quat = _align_up(self.quat, target_up)
        t = 1.0 - math.exp(-config.LAND_ALIGN_SPEED * dt)
        self.quat = _quat_nlerp(self.quat, target_quat, t)
        self.position += target_up * config.TAKEOFF_CLIMB_SPEED * dt
        self.speed = damp(self.speed, config.ATMO_CRUISE,
                          config.TAKEOFF_ACCEL, dt)
        if altitude >= config.TAKEOFF_ALTITUDE:
            self.assist = None
            self.landed = False

    @property
    def altitude(self):
        return self.position.z

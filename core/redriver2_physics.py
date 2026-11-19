"""REDRIVER2 vehicle dynamics port (simplified but drivable).

Mirrors the structures/flow from PORTING_NOTES.md section 4.  Exact per-wheel
force parity with AddWheelForcesDriver1 is NOT yet achieved (C source not in
repo); provisional slip values are used (marked TODO).  Tunable via config.py.
"""

import math
try:
    import config
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
from core.fixedpoint import (ONE, FIXED, FIXEDH, RSIN, RCOS, clamp, Vec3)
from core.track import surface_at
try:
    from core.physics_helpers import (steering_update, auto_brake_update,
                                     thrust_from_input, friction_scales,
                                     STEER_MAX)
except Exception:
    from physics_helpers import (steering_update, auto_brake_update,
                                 thrust_from_input, friction_scales,
                                 STEER_MAX)


class HandlingType:
    def __init__(self):
        self.friction_scale_ratio = 32
        self.aggressive_braking = 1
        self.four_wheel_drive = 0
        self.auto_brake_on = 1


class Wheel:
    def __init__(self):
        self.sus_compression = 0
        self.locked = 0
        self.on_grass = 0
        self.surface = 0
        self.world_pos = (0, 0, 0)
        self.slip_lateral = 0.0
        self.slip_longitudinal = 0.0
        self.wheel_speed = 0
        self.texture_fast = False


class Vehicle:
    def __init__(self, cosmetics, handling_type=None):
        self.cos = cosmetics
        self.handling = handling_type or HandlingType()
        self.position = Vec3(0, 0, 0)
        self.orientation = 0
        self.angular_velocity = Vec3(0, 0, 0)
        self.linear_velocity = Vec3(0, 0, 0)
        self.wheel_angle = 0
        self.thrust = 0
        self.handbrake = 0
        self.wheelspin = 0
        self.auto_brake = 0
        self.may_be_colliding = 0
        self.wheels = [Wheel() for _ in range(4)]
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steer_input = 0.0
        self.fast_steer = False
        self.speed = 0.0

    def reset(self, x: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> None:  # InitCarPhysics
        self.position = Vec3(int(x * ONE), 0, int(z * ONE))
        self.orientation = int((yaw / (2 * math.pi)) * ONE) & (ONE - 1)
        self.angular_velocity = Vec3(0, 0, 0)
        self.linear_velocity = Vec3(0, 0, 0)
        self.wheel_angle = 0
        self.thrust = 0
        self.handbrake = 0
        self.wheelspin = 0
        self.auto_brake = 0
        for w in self.wheels:
            w.sus_compression = 0
            w.locked = 0
            w.slip_lateral = 0.0
            w.slip_longitudinal = 0.0
            w.wheel_speed = 0
            w.texture_fast = False

    def _yaw_rad(self):
        return (self.orientation / ONE) * 2.0 * math.pi

    # -- convenience accessors (float, intuitive units) ---------------------
    @property
    def yaw_rad(self):
        return self._yaw_rad()

    @property
    def yaw_deg(self):
        return self.yaw_rad * 180.0 / math.pi

    @property
    def position_m(self):
        return (self.position.x / ONE, 0.0, self.position.z / ONE)

    @property
    def velocity_m(self):
        return (self.linear_velocity.x / ONE, 0.0, self.linear_velocity.z / ONE)

    def process_input(self, inp) -> None:  # ProcessCarPad
        self.steer_input = float(getattr(inp, "steer", 0.0))
        self.fast_steer = bool(getattr(inp, "fast_steer", False))
        self.throttle_input = float(getattr(inp, "throttle", 0.0))
        self.brake_input = float(getattr(inp, "brake", 0.0))
        self.handbrake = 1 if getattr(inp, "handbrake", False) else 0
        self.wheelspin = 1 if (self.throttle_input > 0.5 and getattr(inp, "wheelspin", False)) else 0

        self.wheel_angle = steering_update(self.wheel_angle, self.steer_input,
                                          self.fast_steer)
        self.auto_brake = auto_brake_update(self.auto_brake, self.brake_input,
                                           bool(self.handling.auto_brake_on))
        self.thrust = thrust_from_input(self.cos.power_ratio, self.throttle_input,
                                       self.brake_input, bool(self.wheelspin))

    def _get_friction_scales(self):
        any_grass = any(w.on_grass for w in self.wheels)
        return friction_scales(self.cos, self.handling, self.thrust,
                              bool(self.handbrake), bool(self.wheelspin),
                              self.auto_brake, any_grass)

    def step(self, dt: float) -> None:  # StepOneCar
        # Phase 1: forces
        self._compute_forces(dt)
        # Phase 2: integration
        self._integrate(dt)
        # Phase 3: wheel data (provisional; TODO wire AddWheelForcesDriver1)
        self._update_wheels()

    def _compute_forces(self, dt):
        yaw = self._yaw_rad()
        self._fx = math.cos(yaw)
        self._fz = math.sin(yaw)
        self._vx = self.linear_velocity.x / ONE
        self._vz = self.linear_velocity.z / ONE
        self._speed = self._vx * self._fx + self._vz * self._fz
        self.speed = self._speed

        front_scale, rear_scale = self._get_friction_scales()

        ax = 0.0
        az = 0.0
        thrust_f = self.thrust / ONE
        ax += self._fx * thrust_f * 2.0
        az += self._fz * thrust_f * 2.0

        steer_norm = (self.wheel_angle / 511.0) if abs(self.wheel_angle) > 0 else 0.0
        self._yaw_torque = steer_norm * (min(abs(self._speed), 30.0) * 0.08 + 0.6)

        drag = 0.3 + (0.002 * self._speed * self._speed)
        ax -= self._vx * drag
        az -= self._vz * drag

        self._rx = -self._fz
        self._rz = self._fx
        self._side = self._vx * self._rx + self._vz * self._rz
        grip = (front_scale + rear_scale) / (2 * ONE) * 1.2
        ax -= self._rx * self._side * grip
        az -= self._rz * self._side * grip

        if self.handbrake:
            ax *= 0.6
            az *= 0.6

        self._ax = ax
        self._az = az

    def _integrate(self, dt):
        vx2 = self._vx + self._ax * dt
        vz2 = self._vz + self._az * dt
        self.linear_velocity.x = int(vx2 * ONE)
        self.linear_velocity.z = int(vz2 * ONE)

        d_yaw = self._yaw_torque * dt
        self.orientation = (self.orientation + int((d_yaw / (2 * math.pi)) * ONE)) & (ONE - 1)

        px = self.position.x / ONE + vx2 * dt
        pz = self.position.z / ONE + vz2 * dt
        self.position.x = int(px * ONE)
        self.position.z = int(pz * ONE)

    def _update_wheels(self):
        for i, w in enumerate(self.wheels):
            w.wheel_speed = int(abs(self._speed) * ONE * 0.5)
            w.slip_lateral = abs(self._side) * 0.5
            w.slip_longitudinal = abs(self.thrust / ONE) * 0.001
            w.locked = 1 if (self.handbrake and i >= 2) else 0
            w.texture_fast = abs(w.wheel_speed) > config.WHEEL_FAST_TEXTURE_SPEED

    def _rebuild_matrix(self):
        pass


class PhysicsWorld:
    """Steps all vehicles and resolves collisions against the track."""

    def __init__(self, track=None, substeps=1):
        self.vehicles = []
        self.track = track
        self.substeps = max(1, int(substeps))

    def add(self, v):
        self.vehicles.append(v)

    def step(self, dt: float) -> None:  # GlobalTimeStep
        sub_dt = dt / self.substeps
        for _ in range(self.substeps):
            for v in self.vehicles:
                v.step(sub_dt)
                if self.track is not None:
                    self._resolve_collisions(v)

    def _resolve_collisions(self, v) -> None:  # bcollided2d / bFindCollisionPoint
        corr = self.track.collide_car(
            v.position.x / ONE, v.position.z / ONE, 2.0, 1.0, v._yaw_rad())
        for nx, nz, depth in corr:
            v.position.x += int(nx * depth * ONE * 0.5)
            v.position.z += int(nz * depth * ONE * 0.5)
            vn = (v.linear_velocity.x / ONE) * nx + (v.linear_velocity.z / ONE) * nz
            if vn < 0:
                v.linear_velocity.x -= int(vn * nx * ONE * 0.5)
                v.linear_velocity.z -= int(vn * nz * ONE * 0.5)


if __name__ == "__main__":
    from core.track import Track
    from core.car_profile import load_cosmetics
    cos = load_cosmetics()
    v = Vehicle(cos)
    v.reset(0, 0, 0)
    w = PhysicsWorld(Track())
    w.add(v)
    inp = type("I", (), {"steer": 0.0, "throttle": 1.0, "brake": 0.0,
                         "handbrake": False, "fast_steer": False, "wheelspin": False})()
    v.process_input(inp)
    for _ in range(60):
        w.step(1 / 60.0)
    print("speed", round(v.speed, 2), "pos", round(v.position.x / ONE, 2), round(v.position.z / ONE, 2))
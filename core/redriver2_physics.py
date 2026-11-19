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

    def reset(self, x=0.0, z=0.0, yaw=0.0):
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

    def process_input(self, inp):
        self.steer_input = float(getattr(inp, "steer", 0.0))
        self.fast_steer = bool(getattr(inp, "fast_steer", False))
        self.throttle_input = float(getattr(inp, "throttle", 0.0))
        self.brake_input = float(getattr(inp, "brake", 0.0))
        self.handbrake = 1 if getattr(inp, "handbrake", False) else 0
        self.wheelspin = 1 if (self.throttle_input > 0.5 and getattr(inp, "wheelspin", False)) else 0

        inc = 64 if self.fast_steer else 32
        target = int(self.steer_input * 511)
        if self.wheel_angle < target:
            self.wheel_angle = min(target, self.wheel_angle + inc)
        elif self.wheel_angle > target:
            self.wheel_angle = max(target, self.wheel_angle - inc)

        if abs(self.steer_input) < 0.05:
            if self.wheel_angle > 0:
                self.wheel_angle = max(0, self.wheel_angle - 16)
            elif self.wheel_angle < 0:
                self.wheel_angle = min(0, self.wheel_angle + 16)

        if self.handling.auto_brake_on:
            if self.brake_input > 0.1:
                self.auto_brake = min(90, self.auto_brake + 6)
            else:
                self.auto_brake = max(0, self.auto_brake - 3)

        pr = self.cos.power_ratio
        if self.brake_input > 0.1:
            self.thrust = -FIXEDH(pr * int(self.brake_input * 4096))
        elif self.throttle_input > 0.0:
            base = FIXEDH(pr * 4915)
            if self.wheelspin:
                base = base * 13 // 10
            self.thrust = base * int(self.throttle_input * 1000) // 1000
        else:
            self.thrust = 0

    def _get_friction_scales(self):
        pr = self.cos.power_ratio
        traction = self.cos.traction
        front_scale = traction
        rear_scale = traction
        if self.thrust < 0:
            rear_scale = FIXEDH(rear_scale * 11 // 10)
        if self.handbrake:
            rear_scale = FIXEDH(rear_scale * 4 // 10)
        if self.wheelspin:
            rear_scale = FIXEDH(rear_scale * 7 // 10)
        for w in self.wheels:
            if w.on_grass:
                front_scale = FIXEDH(front_scale * 6 // 10)
                rear_scale = FIXEDH(rear_scale * 6 // 10)
        if self.auto_brake > 0:
            rear_scale = FIXEDH(rear_scale * (100 - self.auto_brake) // 100)
        ratio = self.handling.friction_scale_ratio
        front_scale = FIXEDH(front_scale * ratio // 32)
        rear_scale = FIXEDH(rear_scale * ratio // 32)
        return front_scale, rear_scale

    def step(self, dt):
        yaw = self._yaw_rad()
        fx = math.cos(yaw)
        fz = math.sin(yaw)
        vx = self.linear_velocity.x / ONE
        vz = self.linear_velocity.z / ONE
        speed = vx * fx + vz * fz
        self.speed = speed

        front_scale, rear_scale = self._get_friction_scales()

        ax = 0.0
        az = 0.0
        thrust_f = self.thrust / ONE
        ax += fx * thrust_f * 2.0
        az += fz * thrust_f * 2.0

        steer_norm = self.wheel_angle / 511.0 if abs(self.wheel_angle) > 0 else 0.0
        yaw_torque = steer_norm * (min(abs(speed), 30.0) * 0.08 + 0.6)

        drag = 0.3 + (0.002 * speed * speed)
        ax -= vx * drag
        az -= vz * drag

        rx = -fz
        rz = fx
        side = vx * rx + vz * rz
        grip = (front_scale + rear_scale) / (2 * ONE) * 1.2
        ax -= rx * side * grip
        az -= rz * side * grip

        if self.handbrake:
            ax *= 0.6
            az *= 0.6

        # integrate (velocity in units/s)
        vx2 = vx + ax * dt
        vz2 = vz + az * dt
        self.linear_velocity.x = int(vx2 * ONE)
        self.linear_velocity.z = int(vz2 * ONE)

        d_yaw = yaw_torque * dt
        self.orientation = (self.orientation + int((d_yaw / (2 * math.pi)) * ONE)) & (ONE - 1)

        px = self.position.x / ONE + vx2 * dt
        pz = self.position.z / ONE + vz2 * dt
        self.position.x = int(px * ONE)
        self.position.z = int(pz * ONE)

        for i, w in enumerate(self.wheels):
            w.wheel_speed = int(abs(speed) * ONE * 0.5)
            w.slip_lateral = abs(side) * 0.5
            w.slip_longitudinal = abs(self.thrust / ONE) * 0.001
            w.locked = 1 if (self.handbrake and i >= 2) else 0
            w.texture_fast = abs(w.wheel_speed) > config.WHEEL_FAST_TEXTURE_SPEED

    def _rebuild_matrix(self):
        pass


class PhysicsWorld:
    def __init__(self, track=None):
        self.vehicles = []
        self.track = track

    def add(self, v):
        self.vehicles.append(v)

    def step(self, dt):
        for v in self.vehicles:
            v.step(dt)
            if self.track is not None:
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
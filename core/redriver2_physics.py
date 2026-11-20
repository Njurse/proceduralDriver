"""REDRIVER2 vehicle dynamics port (work in progress).

Pad and grip rules are the Driver 2 ones (core.physics_helpers:
ProcessCarPad, GetFrictionScalesDriver1).  The force/integration step below
is still a provisional float model; AddWheelForcesDriver1, StepOneCar and
GlobalTimeStep are the next ports.  Tunable via config.py.
"""

import math

import config
from core.fixedpoint import ONE, Vec3
from core.physics_helpers import (
    CAR_PAD_ACCEL,
    CAR_PAD_BRAKE,
    CAR_PAD_FASTSTEER,
    CAR_PAD_HANDBRAKE,
    CAR_PAD_LEFT,
    CAR_PAD_RIGHT,
    CAR_PAD_WHEELSPIN,
    FS_TOTAL,
    STEER_CAP_FAST,
    PadState,
    get_friction_scales_driver1,
    process_car_pad,
)

STEER_DEADZONE = 0.05
BRAKE_THRESHOLD = 0.1


class Wheel:
    def __init__(self):
        self.sus_compression = 0
        self.locked = False
        self.on_grass = 0
        self.surface = 0
        self.world_pos = (0, 0, 0)
        self.slip_lateral = 0.0
        self.slip_longitudinal = 0.0
        self.wheel_speed = 0
        self.texture_fast = False


def pad_from_input(inp) -> int:
    """Map a float input snapshot (keyboard/gamepad) to CAR_PAD_* bits."""
    steer = float(getattr(inp, "steer", 0.0))
    pad = 0
    if steer > STEER_DEADZONE:
        pad |= CAR_PAD_RIGHT
    elif steer < -STEER_DEADZONE:
        pad |= CAR_PAD_LEFT
    if getattr(inp, "fast_steer", False):
        pad |= CAR_PAD_FASTSTEER
    if float(getattr(inp, "throttle", 0.0)) > 0.0:
        pad |= CAR_PAD_ACCEL
    if float(getattr(inp, "brake", 0.0)) > BRAKE_THRESHOLD:
        pad |= CAR_PAD_BRAKE
    if getattr(inp, "handbrake", False):
        pad |= CAR_PAD_HANDBRAKE
    if getattr(inp, "wheelspin", False):
        pad |= CAR_PAD_WHEELSPIN
    return pad


class Vehicle:
    def __init__(self, cosmetics, hnd_type: int = 0):
        self.cos = cosmetics
        self.hnd_type = hnd_type
        self.wheels = [Wheel() for _ in range(4)]
        self.pad = 0
        self.analog_steer = None
        self.changing_gear = False
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steer_input = 0.0
        self.fast_steer = False
        self.reset()

    def reset(self, x: float = 0.0, z: float = 0.0, yaw: float = 0.0) -> None:  # InitCarPhysics
        self.position = Vec3(int(x * ONE), 0, int(z * ONE))
        self.orientation = int((yaw / (2 * math.pi)) * ONE) & (ONE - 1)
        self.angular_velocity = Vec3(0, 0, 0)
        self.linear_velocity = Vec3(0, 0, 0)
        self.pad_state = PadState()
        self.wheel_speed = 0.0  # m/s along heading (C hd.wheel_speed)
        self.speed = 0.0
        self.may_be_colliding = 0
        for w in self.wheels:
            w.sus_compression = 0
            w.locked = False
            w.slip_lateral = 0.0
            w.slip_longitudinal = 0.0
            w.wheel_speed = 0
            w.texture_fast = False

    # -- C CAR_DATA fields written by ProcessCarPad -------------------------
    @property
    def wheel_angle(self) -> float:
        return self.pad_state.wheel_angle

    @property
    def thrust(self) -> float:
        return self.pad_state.thrust

    @property
    def handbrake(self) -> int:
        return int(self.pad_state.handbrake)

    @property
    def wheelspin(self) -> int:
        return int(self.pad_state.wheelspin)

    @property
    def auto_brake(self) -> float:
        return self.pad_state.auto_brake

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

    def process_input(self, inp) -> None:
        """Latch this frame's controls; ProcessCarPad runs in step(dt)."""
        self.steer_input = float(getattr(inp, "steer", 0.0))
        self.fast_steer = bool(getattr(inp, "fast_steer", False))
        self.throttle_input = float(getattr(inp, "throttle", 0.0))
        self.brake_input = float(getattr(inp, "brake", 0.0))
        self.analog_steer = self.steer_input if getattr(inp, "analog", False) else None
        self.pad = pad_from_input(inp)

    def step(self, dt: float) -> None:  # StepOneCar
        self.pad_state = process_car_pad(
            self.pad_state,
            self.pad,
            dt,
            analog_steer=self.analog_steer,
            power_ratio=self.cos.power_ratio,
            wheel_speed=self.wheel_speed,
            hnd_type=self.hnd_type,
            changing_gear=self.changing_gear,
        )
        self._compute_forces(dt)
        self._integrate(dt)
        self._update_wheels()

    def _apply_friction_scales(self):
        """GetFrictionScalesDriver1, writing back thrust/wheelspin/locks."""
        ps = self.pad_state
        fr = get_friction_scales_driver1(
            thrust=ps.thrust,
            wheelspin=ps.wheelspin,
            handbrake=ps.handbrake,
            auto_brake=ps.auto_brake,
            wheel_speed=self.wheel_speed,
            front_on_grass=bool(self.wheels[0].on_grass),
            hnd_type=self.hnd_type,
            power_ratio=self.cos.power_ratio,
            traction=self.cos.traction,
        )
        self.pad_state = PadState(ps.wheel_angle, ps.auto_brake, fr.thrust, ps.handbrake,
                                  fr.wheelspin)
        for w, locked in zip(self.wheels, fr.locked, strict=True):
            w.locked = locked
        return fr.front, fr.rear

    def _compute_forces(self, dt):
        # Provisional float force model (replaced by AddWheelForcesDriver1).
        yaw = self._yaw_rad()
        self._fx = math.cos(yaw)
        self._fz = math.sin(yaw)
        self._vx = self.linear_velocity.x / ONE
        self._vz = self.linear_velocity.z / ONE
        self._speed = self._vx * self._fx + self._vz * self._fz
        self.speed = self._speed

        front_scale, rear_scale = self._apply_friction_scales()

        thrust = self.thrust
        ax = self._fx * thrust * 2.0
        az = self._fz * thrust * 2.0

        steer_norm = self.wheel_angle / STEER_CAP_FAST
        self._yaw_torque = steer_norm * (min(abs(self._speed), 30.0) * 0.08 + 0.6)

        drag = 0.3 + (0.002 * self._speed * self._speed)
        ax -= self._vx * drag
        az -= self._vz * drag

        self._rx = -self._fz
        self._rz = self._fx
        self._side = self._vx * self._rx + self._vz * self._rz
        # Normalised so straight-line coasting grip matches the old model.
        grip = (front_scale + rear_scale) / (FS_TOTAL / ONE) * 1.2
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
        yaw = self._yaw_rad()
        vx = self.linear_velocity.x / ONE
        vz = self.linear_velocity.z / ONE
        self.wheel_speed = vx * math.cos(yaw) + vz * math.sin(yaw)
        for w in self.wheels:
            w.wheel_speed = int(abs(self._speed) * ONE * 0.5)
            w.slip_lateral = abs(self._side) * 0.5
            w.slip_longitudinal = abs(self.thrust) * 0.001
            w.texture_fast = abs(w.wheel_speed) > config.WHEEL_FAST_TEXTURE_SPEED


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

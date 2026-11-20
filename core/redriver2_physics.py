"""Driver 2 player car: pad rules, forces and GlobalTimeStep integration.

State is float64 SI (metres, m/s, radians) in the C physics frame: +Y up,
body +Z forward, heading / wheel angle positive toward +X (the C's right).
PhysicsWorld runs a fixed tick (TICK_HZ) with an accumulator, so behaviour
does not depend on the render frame rate.
"""

import math
from types import SimpleNamespace

import numpy as np

import config
from core import quat
from core.car_forces import (
    GRAVITY,
    SUS_REST_OFFSET,
    SUS_STIFFNESS,
    CarGeometry,
    car_forces,
    flat_ground,
)
from core.engine import GearboxState, gear_thrust_scale, step_gearbox, wheel_rpm
from core.physics_helpers import (
    CAR_PAD_ACCEL,
    CAR_PAD_BRAKE,
    CAR_PAD_FASTSTEER,
    CAR_PAD_HANDBRAKE,
    CAR_PAD_LEFT,
    CAR_PAD_RIGHT,
    CAR_PAD_WHEELSPIN,
    WHEELSPIN_CUTOFF_SPEED,
    PadState,
    process_car_pad,
)
from core.units import C_FRAME_HZ, dr2_length, dr2_velocity

TICK_HZ = 60.0  # 2 physics ticks per Driver 2 frame
MAX_TICKS_PER_STEP = 8
STEER_DEADZONE = 0.05
BRAKE_THRESHOLD = 0.1

# GlobalTimeStep constants
REST_SPEED = dr2_velocity(2048)  # hd.speed == 0 below this
REST_SNAP = dr2_velocity(1000)  # |vx| + |vy| + |vz| below this: stop dead
VY_DAMP_ABOVE = dr2_velocity(200000)  # fast upward velocity * 3/4 per frame
# Kept as a sanity clamp (C: +-0x800000 per axis = 0.5 rad per frame).
MAX_ANGULAR_SPEED = 0x800000 / 2.0**24 * C_FRAME_HZ


class Wheel:
    """Per-wheel view for render/audio."""

    def __init__(self):
        self.compression = 0.0  # m
        self.locked = False
        self.on_grass = False
        self.world_pos = np.zeros(3)
        self.slip_lateral = 0.0  # m/s sideways at the contact
        self.slip_longitudinal = 0.0  # m/s, locked or spinning
        self.wheel_speed = 0.0  # rad/s
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


def Input():  # noqa: N802 - kept as a factory for the old vehicle3d.Input call sites
    return SimpleNamespace(steer=0.0, throttle=0.0, brake=0.0, handbrake=False,
                           fast_steer=False, wheelspin=False, analog=False)


class Vehicle:
    def __init__(self, cosmetics, hnd_type: int = 0):
        self.cos = cosmetics
        self.geo = CarGeometry.from_cosmetics(cosmetics)
        self.hnd_type = hnd_type
        self.wheels = [Wheel() for _ in range(4)]
        self.pad = 0
        self.analog_steer = None
        self.throttle_input = 0.0
        self.brake_input = 0.0
        self.steer_input = 0.0
        self.fast_steer = False
        self.reset()

    # -- InitCarPhysics -----------------------------------------------------
    def rest_height(self) -> float:
        """Body origin height that puts the car at suspension equilibrium."""
        rest = -GRAVITY / (4.0 * SUS_STIFFNESS)
        return -float(np.mean(self.geo.wheel_disp[:, 1])) + SUS_REST_OFFSET - rest

    def reset(self, x: float = 0.0, z: float = 0.0, heading: float = 0.0,
              ground=flat_ground) -> None:
        self.ground = ground
        base = ground(x, z)[0]
        self.pos = np.array([x, base + self.rest_height(), z])
        self.vel = np.zeros(3)
        self.avel = np.zeros(3)
        self.quat = quat.from_yaw(heading)
        self.rot = quat.to_matrix(self.quat)
        self.compression = np.full(4, -GRAVITY / (4.0 * SUS_STIFFNESS))
        self.pad_state = PadState()
        self.gearbox = GearboxState()
        self.wheel_speed = 0.0  # hd.wheel_speed, m/s along the heading
        for w in self.wheels:
            w.__init__()

    # -- views ----------------------------------------------------------------
    @property
    def heading(self) -> float:
        return math.atan2(self.rot[0, 2], self.rot[2, 2])

    @property
    def forward(self) -> np.ndarray:
        return self.rot[:, 2]

    @property
    def forward_speed(self) -> float:
        h = self.heading
        return self.vel[0] * math.sin(h) + self.vel[2] * math.cos(h)

    @property
    def speed(self) -> float:
        """Signed speed along the heading, m/s."""
        return self.forward_speed

    @property
    def changing_gear(self) -> bool:
        return self.gearbox.changing_gear

    @property
    def gear(self) -> int:
        return self.gearbox.gear

    @property
    def engine_rpm(self) -> float:
        """Clutch-in rpm for the HUD (the sound model adds slip and idle)."""
        return max(config.IDLE_RPM, wheel_rpm(self.forward_speed, self.gear))

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

    # -- control ----------------------------------------------------------------
    def process_input(self, inp) -> None:
        """Latch this frame's controls; ProcessCarPad runs every tick in step()."""
        self.steer_input = float(getattr(inp, "steer", 0.0))
        self.fast_steer = bool(getattr(inp, "fast_steer", False))
        self.throttle_input = float(getattr(inp, "throttle", 0.0))
        self.brake_input = float(getattr(inp, "brake", 0.0))
        self.analog_steer = self.steer_input if getattr(inp, "analog", False) else None
        self.pad = pad_from_input(inp)

    # -- one physics tick -------------------------------------------------------
    def step(self, dt: float) -> None:
        self.pad_state = process_car_pad(
            self.pad_state, self.pad, dt,
            analog_steer=self.analog_steer,
            power_ratio=self.cos.power_ratio,
            wheel_speed=self.wheel_speed,
            hnd_type=self.hnd_type,
            changing_gear=self.gearbox.changing_gear,
        )
        f = car_forces(self, self.pad_state, self.geo, dt, self.ground, self.hnd_type,
                       thrust_scale=gear_thrust_scale(self.gearbox.gear))
        ps = self.pad_state
        self.pad_state = PadState(ps.wheel_angle, -1.0 if f.reset_auto_brake else ps.auto_brake,
                                  f.thrust, ps.handbrake, f.wheelspin)
        self._integrate(f.acc, f.alpha, dt)
        self.wheel_speed = f.wheel_speed
        self.gearbox = step_gearbox(
            self.gearbox, dt, wheel_speed=f.wheel_speed,
            rear_grounded=self.compression[1] > 0.0 or self.compression[3] > 0.0)
        self._update_wheels(f.wheels)

    def _integrate(self, acc, alpha, dt) -> None:
        """GlobalTimeStep for one car.

        DEVIATION: semi-implicit Euler at TICK_HZ with an exact quaternion
        rotation instead of the C's 4 x RK2 fixed-point subframes.
        """
        was_resting = math.hypot(self.vel[0], self.vel[2]) < REST_SPEED
        self.vel = self.vel + acc * dt
        self.avel = self.avel + alpha * dt
        if self.vel[1] > VY_DAMP_ABOVE:
            self.vel[1] *= 0.75 ** (dt * C_FRAME_HZ)
        if was_resting and float(np.abs(self.vel).sum()) < REST_SNAP:
            self.vel = np.zeros(3)
            self.avel = np.zeros(3)
        self.avel = np.clip(self.avel, -MAX_ANGULAR_SPEED, MAX_ANGULAR_SPEED)
        self.pos = self.pos + self.vel * dt
        self.quat = quat.integrate(self.quat, self.avel, dt)
        self.rot = quat.to_matrix(self.quat)

    def _update_wheels(self, outs) -> None:
        radius = self.geo.wheel_radius
        fwd = self.forward_speed
        for i, (w, o) in enumerate(zip(self.wheels, outs, strict=True)):
            w.compression = o.compression
            w.locked = o.locked
            w.on_grass = o.on_grass
            w.world_pos = o.world_pos
            w.slip_lateral = o.slide
            rolling = self.wheel_speed
            if w.locked:
                rolling = 0.0
                w.slip_longitudinal = abs(fwd)
            elif i & 1 and self.wheelspin:
                rolling = max(abs(fwd), WHEELSPIN_CUTOFF_SPEED)
                w.slip_longitudinal = rolling - abs(fwd)
            else:
                w.slip_longitudinal = 0.0
            w.wheel_speed = rolling / radius
            w.texture_fast = abs(w.wheel_speed) * 100.0 > config.WHEEL_FAST_TEXTURE_SPEED

    # -- render helpers ---------------------------------------------------------
    def wheel_visual_offset(self, i: int) -> np.ndarray:
        """Body-space wheel centre for drawing (cars.c DrawCarWheels).

        The C draws the wheel 17 units inboard of wheelDisp and wheelSize - 14
        + compression above it, so the tyre sits on the ground.
        """
        disp = self.geo.wheel_disp[i].copy()
        inset = dr2_length(17)
        disp[0] -= math.copysign(inset, disp[0])
        disp[1] += self.geo.wheel_radius - SUS_REST_OFFSET + self.wheels[i].compression
        return disp


class PhysicsWorld:
    """Steps all vehicles at a fixed tick and resolves track collisions."""

    def __init__(self, track=None, tick_hz: float = TICK_HZ):
        self.vehicles = []
        self.track = track
        self.tick = 1.0 / tick_hz
        self._accumulator = 0.0

    def add(self, v):
        self.vehicles.append(v)

    def step(self, dt: float) -> int:
        """Advance by dt seconds of wall time; returns the ticks run."""
        self._accumulator = min(self._accumulator + dt, self.tick * MAX_TICKS_PER_STEP)
        ticks = 0
        while self._accumulator >= self.tick - 1e-9:
            for v in self.vehicles:
                v.step(self.tick)
                if self.track is not None:
                    self._resolve_collisions(v)
            self._accumulator -= self.tick
            ticks += 1
        return ticks

    def _resolve_collisions(self, v) -> None:
        hx = dr2_length(v.cos.col_box[0])
        hz = dr2_length(v.cos.col_box[2])
        corr = self.track.collide_car(v.pos[0], v.pos[2], hx, hz, -v.heading)
        for nx, nz, depth in corr:
            v.pos[0] += nx * depth * 0.5
            v.pos[2] += nz * depth * 0.5
            vn = v.vel[0] * nx + v.vel[2] * nz
            if vn < 0:
                v.vel[0] -= vn * nx * 0.5
                v.vel[2] -= vn * nz * 0.5

"""Driver 2 pad and grip rules as float64 functions.

Ports of REDRIVER2 ``Game/C/handling.c:ProcessCarPad`` and
``Game/C/wheelforces.c:GetFrictionScalesDriver1``.  The gameplay rules (steer
caps and rates, centering, autoBrake, friction cases, wheel lock, wheelspin,
gear-change thrust cut) follow the C; the PSX precision machinery does not
(no >> shifts, & masks, integer division or short/char storage).  Places that
deviate from the C on purpose are marked ``DEVIATION``.

Units (C value -> here):
  * time: seconds.  The C ran these rules once per 30 Hz frame, so per-frame
    increments become rates of ``value * C_FRAME_HZ`` per second.
  * wheel angle: radians (C: 4096 units per turn).
  * wheel speed: m/s along the car heading (C: ``hd.wheel_speed``, forward
    velocity in 4096ths of a world unit per frame).
  * thrust and friction scales: the C value / 4096 (``ONE`` = 1.0).
"""

import math
from dataclasses import dataclass

ONE = 4096.0
C_FRAME_HZ = 30.0
# World scale the renderer already uses (config.HAVANA_SHIP_SCALE = 0.01 m/unit).
C_UNITS_PER_METRE = 100.0


def c_angle(units: float) -> float:
    """C angle units (4096 per turn) to radians."""
    return units * math.tau / ONE


def c_wheel_speed(ws: float) -> float:
    """C ``hd.wheel_speed`` to m/s (wheelforces.c AddWheelForcesDriver1 end)."""
    return ws / ONE * C_FRAME_HZ / C_UNITS_PER_METRE


def to_c_wheel_speed(mps: float) -> float:
    """m/s to C ``hd.wheel_speed`` units (inverse of c_wheel_speed)."""
    return mps * C_UNITS_PER_METRE / C_FRAME_HZ * ONE


# Game/C/pad.h ECarPads (MPAD_* bit values).
CAR_PAD_FASTSTEER = 0x4
CAR_PAD_HANDBRAKE = 0x10
CAR_PAD_WHEELSPIN = 0x20
CAR_PAD_ACCEL = 0x40
CAR_PAD_BRAKE = 0x80
CAR_PAD_RIGHT = 0x2000
CAR_PAD_LEFT = 0x8000
CAR_PAD_STEER = CAR_PAD_LEFT | CAR_PAD_RIGHT

# handling.c ProcessCarPad steering constants.
STEER_CAP_REGULAR = c_angle(352)
STEER_CAP_FAST = c_angle(511)
STEER_RATE_REGULAR = c_angle(32) * C_FRAME_HZ  # rad/s (C: 32 units per frame)
STEER_RATE_FAST = c_angle(64) * C_FRAME_HZ
STEER_CENTER_RATE = c_angle(64) * C_FRAME_HZ
# Analogue: C computes s * (s*s/60) * 0x88888889 >> 32 >> 5 = s^3 / 3600 (fast)
# and s * (s*s/80) * 0x66666667 >> 32 >> 5 = s^3 / 6400 (regular), s = stick
# in -128..127.  Here the stick is a float in [-1, 1] scaled by 127.
ANALOG_STICK_SCALE = 127.0
ANALOG_DIVISOR_FAST = 3600.0
ANALOG_DIVISOR_REGULAR = 6400.0
ANALOG_CENTER_OFFSET = c_angle(64)
ANALOG_AUTOBRAKE_LIMIT = c_angle(270)
AUTO_BRAKE_CAP = 90.0  # C frames of continuous steering
_CENTER_EPS = 1e-12

# handling.c ProcessCarPad thrust constants (C value / ONE).
THRUST_ACCEL = 4915.0 / ONE
THRUST_GEAR_CHANGE = 1.0 / ONE  # C sets thrust = 1: a cut, but still "> 0"
WHEELSPIN_CUTOFF_SPEED = c_wheel_speed(452952)

# wheelforces.c GetFrictionScalesDriver1 constants (C units, 4096 = 1.0).
FRONT_FS_BRAKING = 1453
FRONT_FS_COASTING = 937
FRONT_FS_ACCEL = 820
FS_TOTAL = 0x780  # rearFS = 0x780 - frontFS
AUTO_BRAKE_FRONT_CAP = 13
AUTO_BRAKE_FRONT_GAIN = 45  # C: (2*ab + ab) * 15
FRONT_FS_WHEELSPIN = 600
FRONT_FS_HANDBRAKE_MOVING = 656
FRONT_FS_HANDBRAKE_STOPPED = -375
FRONT_FS_REVERSING = -400
WHEEL_LOCK_SPEED = c_wheel_speed(41943)
THRUST_WHEELSPIN = 5000.0 / ONE


@dataclass(frozen=True)
class HandlingType:
    """wheelforces.c HANDLING_TYPE."""

    friction_scale_ratio: int
    aggressive_braking: bool
    four_wheel_drive: bool
    auto_brake_on: bool


# wheelforces.c handlingType[7]
HANDLING_TYPES = (
    HandlingType(32, True, False, True),
    HandlingType(29, False, False, False),
    HandlingType(45, True, True, False),
    HandlingType(55, True, True, False),
    HandlingType(68, True, True, False),
    HandlingType(32, True, False, True),
    HandlingType(29, False, False, False),
)


@dataclass(frozen=True)
class PadState:
    """The CAR_DATA fields ProcessCarPad writes."""

    wheel_angle: float = 0.0  # radians, + = right
    auto_brake: float = 0.0  # C frames of continuous steering
    thrust: float = 0.0  # C thrust / ONE
    handbrake: bool = False
    wheelspin: bool = False


def _steer_digital(angle: float, pad: int, dt: float) -> float:
    """handling.c ProcessCarPad, use_analogue == 0 branch (steering only)."""
    if pad & CAR_PAD_FASTSTEER:
        rate, cap = STEER_RATE_FAST, STEER_CAP_FAST
    else:
        rate, cap = STEER_RATE_REGULAR, STEER_CAP_REGULAR
    # The C clamps only on the side it moved toward, so dropping fast steer
    # while held snaps a 511 lock back to 352 (kept: it shapes the feel).
    if pad & CAR_PAD_RIGHT:
        angle = min(angle + rate * dt, cap)
    if pad & CAR_PAD_LEFT:
        angle = max(angle - rate * dt, -cap)
    return angle


def _steer_analog(stick: float, fast: bool) -> float:
    """handling.c ProcessCarPad, use_analogue != 0 branch (cubic curve).

    DEVIATION: no integer division, no & 0xfffc quantisation.
    """
    s = stick * ANALOG_STICK_SCALE
    divisor = ANALOG_DIVISOR_FAST if fast else ANALOG_DIVISOR_REGULAR
    return c_angle(s * s * s / divisor)


def _center(angle: float, step: float) -> float:
    """handling.c ProcessCarPad 'center steering': move toward 0 by step.

    The C snaps to 0 inside the step; the epsilon keeps rate * dt float
    rounding from leaving a 1e-17 residue.
    """
    if abs(angle) <= step + _CENTER_EPS:
        return 0.0
    return angle - math.copysign(step, angle)


def _brake_thrust(power_ratio: int, wheel_speed: float) -> float:
    """handling.c ProcessCarPad CAR_PAD_BRAKE: brake, or reverse when slow.

    rws = wheel_speed * 1500 / 1024 (in ONE units); moving forward or slowly
    backward gives a flat -5000, faster reversing tapers to 0 at rws = -278.
    """
    rws = to_c_wheel_speed(wheel_speed) * 1500.0 / 1024.0 / ONE
    rws = -5000.0 if -rws < 23.0 else (rws + 278.0) * -4778.0 / 256.0
    return power_ratio * rws / ONE / ONE


def process_car_pad(
    state: PadState,
    pad: int,
    dt: float,
    *,
    analog_steer: float | None = None,
    power_ratio: int = 4096,
    wheel_speed: float = 0.0,
    hnd_type: int = 0,
    changing_gear: bool = False,
) -> PadState:
    """handling.c ProcessCarPad for a player car, advanced by dt seconds.

    ``pad`` holds CAR_PAD_* bits; ``analog_steer`` (stick in [-1, 1]) selects
    the analogue branch like the C ``use_analogue``.  Not ported: car exit,
    horn, damage lock, JERICHO hooks, hndType 5 and target-car rubber banding.
    """
    frames = dt * C_FRAME_HZ
    auto_brake = min(state.auto_brake, AUTO_BRAKE_CAP)
    wheelspin = state.wheelspin

    if pad & CAR_PAD_HANDBRAKE:
        handbrake = True
    else:
        handbrake = False
        wheelspin = bool(pad & CAR_PAD_WHEELSPIN)
        if wheelspin and wheel_speed > WHEELSPIN_CUTOFF_SPEED:
            wheelspin = False  # burnout ends, carry on accelerating
            pad |= CAR_PAD_ACCEL

    if analog_steer is None:
        angle = _steer_digital(state.wheel_angle, pad, dt)
        auto_brake = auto_brake + frames if pad & CAR_PAD_STEER else 0.0
    else:
        angle = _steer_analog(analog_steer, bool(pad & CAR_PAD_FASTSTEER))
        auto_brake = 0.0 if abs(angle) <= ANALOG_AUTOBRAKE_LIMIT else auto_brake + frames

    if not pad & CAR_PAD_STEER:
        # The C subtracts 64 units per frame.  Digital: a centering rate.
        # Analogue: the angle is recomputed every frame, so it acts as a fixed
        # deadzone offset rather than a rate.
        step = STEER_CENTER_RATE * dt if analog_steer is None else ANALOG_CENTER_OFFSET
        angle = _center(angle, step)

    thrust = 0.0
    if pad & CAR_PAD_BRAKE:
        thrust = _brake_thrust(power_ratio, wheel_speed)
    elif pad & CAR_PAD_ACCEL:
        thrust = power_ratio * THRUST_ACCEL / ONE
        if hnd_type == 0 and changing_gear:
            thrust = THRUST_GEAR_CHANGE

    return PadState(angle, auto_brake, thrust, handbrake, wheelspin)


@dataclass(frozen=True)
class FrictionResult:
    """GetFrictionScalesDriver1 outputs, including the car fields it rewrites."""

    front: float  # frontFS / ONE
    rear: float  # rearFS / ONE
    thrust: float
    wheelspin: bool
    locked: tuple[bool, bool, bool, bool]  # wheel[0..3]; 1 and 3 are the rears


def get_friction_scales_driver1(
    *,
    thrust: float,
    wheelspin: bool,
    handbrake: bool,
    auto_brake: float,
    wheel_speed: float,
    front_on_grass: bool = False,
    hnd_type: int = 0,
    power_ratio: int = 4096,
    traction: int = 4096,
) -> FrictionResult:
    """wheelforces.c GetFrictionScalesDriver1 for a player car.

    Not ported: civ-AI stop lock and the hndType 5 / dstate 5 boost.
    DEVIATION: float maths, so no integer division and the C's
    ``traction != 4096`` shortcut is just a multiply by 1.
    """
    hp = HANDLING_TYPES[hnd_type]

    if thrust < 0:
        front = float(FRONT_FS_BRAKING)
    elif thrust > 0:
        front = float(FRONT_FS_ACCEL)
    else:
        front = float(FRONT_FS_COASTING)

    if not wheelspin and hp.auto_brake_on and auto_brake > 0 and wheel_speed > 0:
        front += min(auto_brake, AUTO_BRAKE_FRONT_CAP) * AUTO_BRAKE_FRONT_GAIN

    braking_fast = thrust < 0 and wheel_speed > WHEEL_LOCK_SPEED
    all_locked = braking_fast and hnd_type == 0
    locked = [all_locked] * 4

    if not handbrake:
        if wheelspin:
            front += FRONT_FS_WHEELSPIN
    else:
        if thrust >= 0:
            thrust = 0.0
        front += FRONT_FS_HANDBRAKE_MOVING if wheel_speed > 0 else FRONT_FS_HANDBRAKE_STOPPED
        locked[1] = locked[3] = True
        wheelspin = False

    if wheel_speed < 0 and thrust >= 0 and not handbrake:
        front += FRONT_FS_REVERSING

    rear = FS_TOTAL - front

    if wheelspin:
        thrust = power_ratio * THRUST_WHEELSPIN / ONE

    if thrust < 0 and wheel_speed > WHEEL_LOCK_SPEED and hp.aggressive_braking:
        front *= 10.0 / 8.0
        rear *= 10.0 / 8.0
    else:
        front *= (40.0 - 1.0 if front_on_grass else 36.0 - 1.0) / 32.0

    ratio = hp.friction_scale_ratio / 32.0
    grip = traction / ONE
    return FrictionResult(
        front * ratio * grip / ONE,
        rear * ratio * grip / ONE,
        thrust,
        wheelspin,
        (locked[0], locked[1], locked[2], locked[3]),
    )

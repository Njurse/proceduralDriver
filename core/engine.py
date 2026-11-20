"""Gearbox: which gear the car is in and when an upshift cuts the thrust.

DEVIATION: gamesnd.c picked gears from a fixed wheel-speed table.  This uses
car-style ratios from config (GEAR_RATIOS, FINAL_DRIVE, TYRE_RADIUS_M) so the
gearing can be lengthened by name.  Kept from the C: an upshift raises
changingGear, which ProcessCarPad turns into a thrust cut, for about as long
as the C's rev drop took (SHIFT_TIME_S).
"""

import math
from dataclasses import dataclass

import config

RAD_S_TO_RPM = 60.0 / math.tau


def wheel_rpm(speed: float, gear: int) -> float:
    """Engine rpm with the clutch fully in at ``speed`` m/s (sign ignored)."""
    ratio = config.GEAR_RATIOS[gear] * config.FINAL_DRIVE
    return abs(speed) / config.TYRE_RADIUS_M * RAD_S_TO_RPM * ratio


def gear_top_speed(gear: int, rpm: float = config.SHIFT_UP_RPM) -> float:
    """Road speed (m/s) at ``rpm`` in ``gear``."""
    return rpm / wheel_rpm(1.0, gear)


def gear_thrust_scale(gear: int) -> float:
    """Multiplier on the C's thrust in ``gear`` (config.GEAR_THRUST_SCALES)."""
    return config.GEAR_THRUST_SCALES[gear]


def choose_gear(gear: int, speed: float) -> int:
    """Shift up past SHIFT_UP_RPM, down below SHIFT_DOWN_RPM; reverse is 1st."""
    if speed <= 0.0:
        return 0
    top = len(config.GEAR_RATIOS) - 1
    g = min(max(gear, 0), top)
    while g < top and wheel_rpm(speed, g) > config.SHIFT_UP_RPM:
        g += 1
    while g > 0 and wheel_rpm(speed, g) < config.SHIFT_DOWN_RPM:
        g -= 1
    return g


@dataclass(frozen=True)
class GearboxState:
    gear: int = 0
    shift_timer: float = 0.0  # seconds of thrust cut left

    @property
    def changing_gear(self) -> bool:
        return self.shift_timer > 0.0


def step_gearbox(state: GearboxState, dt: float, *, wheel_speed: float,
                 rear_grounded: bool) -> GearboxState:
    """Advance dt seconds.  The gear holds while the rear wheels are in the air."""
    timer = max(0.0, state.shift_timer - dt)
    gear = choose_gear(state.gear, wheel_speed) if rear_grounded else state.gear
    if gear > state.gear:
        timer = config.SHIFT_TIME_S
    return GearboxState(gear, timer)

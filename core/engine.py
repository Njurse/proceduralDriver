"""Engine revs and gears: gamesnd.c GetEngineRevs / ControlCarRevs in floats.

The C runs ControlCarRevs once per 30 Hz frame right after StepOneCar.  A big
rev drop (an upshift) sets changingGear, which cuts the thrust in the next
ProcessCarPad, so the gearbox is part of the handling, not just the sound.
Revs keep the C's units (about 0..30000).
"""

from dataclasses import dataclass

from core.units import C_FRAME_HZ, to_dr2_velocity

# geard[0]: player / non-civ cars.  (lowidl_ws, low_ws, hi_ws, ratio_ac, ratio_id)
GEARS = (
    (0, 0, 163, 144, 135),
    (86, 133, 260, 90, 85),
    (186, 233, 360, 60, 57),
    (286, 326, 9999, 48, 45),
)
MAX_REV_DROP = 1440.0  # per frame
MAX_REV_RISE = 1600.0
WHEELSPIN_REVS = 20160.0
AIRBORNE_REVS = 30719.0
SPIN_MIN_REVS = 8000.0
REVERSE_REV_CAP = 23000.0
REDLINE_REVS = 24000.0  # about where first gear shifts up (163 * 144)


@dataclass(frozen=True)
class EngineState:
    revs: float = 0.0
    gear: int = 0
    changing_gear: bool = False


def gear_speed(wheel_speed: float) -> float:
    """hd.wheel_speed (m/s) in the gearbox's units (C: wheel_speed >> 11)."""
    return to_dr2_velocity(wheel_speed) / 2048.0


def engine_revs(gear: int, wheel_speed: float, accelerating: bool) -> tuple[int, float]:
    """GetEngineRevs: returns (gear, target revs).

    DEVIATION: the wheel speed is not truncated to whole gearbox units.
    """
    ws = gear_speed(wheel_speed)
    if ws <= 0.0:
        ratio = GEARS[0][3] if accelerating else GEARS[0][4]
        return 0, min(-ws * ratio, REVERSE_REV_CAP)
    g = min(max(gear, 0), 3)
    for _ in range(8):
        low = GEARS[g][1] if accelerating else GEARS[g][0]
        ng = g
        if ws < low and g > 0:
            ng = g - 1
            if GEARS[ng][2] < ws:  # too fast for the lower gear
                ng = g
        elif GEARS[g][2] < ws:
            ng = g + 1
        if ng == g:
            break
        g = min(max(ng, 0), 3)
    ratio = GEARS[g][3] if accelerating else GEARS[g][4]
    return g, ws * ratio


def control_car_revs(state: EngineState, dt: float, *, thrust: float, wheelspin: bool,
                     rear_grounded: bool, wheel_speed: float) -> EngineState:
    """ControlCarRevs, advanced by dt seconds (rev slew rates are per C frame)."""
    frames = dt * C_FRAME_HZ
    old = state.revs
    gear = state.gear
    if not wheelspin and (rear_grounded or thrust == 0.0):
        gear, target = engine_revs(gear, wheel_speed, thrust != 0.0)
    else:
        target = WHEELSPIN_REVS if rear_grounded else AIRBORNE_REVS
        old = max(old, SPIN_MIN_REVS)
        gear = 0
    revs = target
    changing = False
    if old - revs > MAX_REV_DROP * frames:
        changing = True
        revs = old - MAX_REV_DROP * frames
    if revs - old > MAX_REV_RISE * frames:
        revs = old + MAX_REV_RISE * frames
    return EngineState(revs, gear, changing)

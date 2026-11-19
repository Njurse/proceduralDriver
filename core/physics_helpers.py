"""Pure helper functions for the REDRIVER2 handling model.

Extracted from Vehicle so the steering/thrust/friction logic can be unit-tested
in isolation.  All inputs/outputs use the same fixed-point conventions as
core.redriver2_physics (ONE=4096).  Behavior is identical to the original
inline implementation.
"""

try:
    import config
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
from core.fixedpoint import ONE, FIXEDH, clamp

# Steering increments (ProcessCarPad digital).
STEER_INC_REGULAR = 32
STEER_INC_FAST = 64
STEER_CENTER_STEP = 16
STEER_MAX = 511
AUTO_BRAKE_CAP = 90
AUTO_BRAKE_INC = 6
AUTO_BRAKE_DECAY = 3


def steering_update(wheel_angle, steer_input, fast_steer):
    """Return the new wheel_angle after one ProcessCarPad steering pass.

    Digital increments, fast/regular, plus auto-centering when no input.
    """
    inc = STEER_INC_FAST if fast_steer else STEER_INC_REGULAR
    target = int(steer_input * STEER_MAX)
    if wheel_angle < target:
        wheel_angle = min(target, wheel_angle + inc)
    elif wheel_angle > target:
        wheel_angle = max(target, wheel_angle - inc)

    if abs(steer_input) < 0.05:
        if wheel_angle > 0:
            wheel_angle = max(0, wheel_angle - STEER_CENTER_STEP)
        elif wheel_angle < 0:
            wheel_angle = min(0, wheel_angle + STEER_CENTER_STEP)
    return wheel_angle


def auto_brake_update(auto_brake, brake_input, auto_brake_on):
    """Return updated auto_brake (capped 0..90)."""
    if not auto_brake_on:
        return auto_brake
    if brake_input > 0.1:
        return min(AUTO_BRAKE_CAP, auto_brake + AUTO_BRAKE_INC)
    return max(0, auto_brake - AUTO_BRAKE_DECAY)


def thrust_from_input(power_ratio, throttle, brake, wheelspin):
    """Return thrust (fixed-point) from throttle/brake inputs (ProcessCarPad)."""
    if brake > 0.1:
        return -FIXEDH(power_ratio * int(brake * 4096))
    if throttle > 0.0:
        base = FIXEDH(power_ratio * 4915)
        if wheelspin:
            base = base * 13 // 10
        return base * int(throttle * 1000) // 1000
    return 0


def friction_scales(cos, handling, thrust, handbrake, wheelspin, auto_brake,
                    any_grass=False):
    """Return (front_scale, rear_scale) fixed-point (GetFrictionScalesDriver1)."""
    traction = cos.traction
    front_scale = traction
    rear_scale = traction
    if thrust < 0:
        rear_scale = FIXEDH(rear_scale * 11 // 10)
    if handbrake:
        rear_scale = FIXEDH(rear_scale * 4 // 10)
    if wheelspin:
        rear_scale = FIXEDH(rear_scale * 7 // 10)
    if any_grass:
        front_scale = FIXEDH(front_scale * 6 // 10)
        rear_scale = FIXEDH(rear_scale * 6 // 10)
    if auto_brake > 0:
        rear_scale = FIXEDH(rear_scale * (100 - auto_brake) // 100)
    ratio = handling.friction_scale_ratio
    front_scale = FIXEDH(front_scale * ratio // 32)
    rear_scale = FIXEDH(rear_scale * ratio // 32)
    return front_scale, rear_scale


if __name__ == "__main__":
    from core.car_profile import load_cosmetics
    cos = load_cosmetics()
    ht = type("H", (), {"friction_scale_ratio": 32, "auto_brake_on": 1})()
    print("steer", steering_update(0, 0.5, False))
    print("thrust", thrust_from_input(cos.power_ratio, 1.0, 0.0, False))
    print("friction", friction_scales(cos, ht, 100, False, False, 0))
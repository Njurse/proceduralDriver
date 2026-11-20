"""Driver 2 pad/grip rules (handling.c ProcessCarPad, wheelforces.c
GetFrictionScalesDriver1).  Expected values are the C constants; one C frame
is dt = 1/30 s.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.physics_helpers import (  # noqa: E402
    CAR_PAD_ACCEL,
    CAR_PAD_BRAKE,
    CAR_PAD_FASTSTEER,
    CAR_PAD_HANDBRAKE,
    CAR_PAD_LEFT,
    CAR_PAD_RIGHT,
    CAR_PAD_WHEELSPIN,
    PadState,
    c_angle,
    c_wheel_speed,
    get_friction_scales_driver1,
    process_car_pad,
)

FRAME = 1.0 / 30.0
ONE = 4096.0


def pad(pad_bits, state=None, dt=FRAME, **kw):
    return process_car_pad(state or PadState(), pad_bits, dt, **kw)


def frames(pad_bits, n, state=None, **kw):
    state = state or PadState()
    for _ in range(n):
        state = process_car_pad(state, pad_bits, FRAME, **kw)
    return state


# -- steering ---------------------------------------------------------------

def test_regular_steer_rate_and_cap_352():
    assert pad(CAR_PAD_RIGHT).wheel_angle == pytest.approx(c_angle(32))
    assert frames(CAR_PAD_RIGHT, 30).wheel_angle == pytest.approx(c_angle(352))
    assert frames(CAR_PAD_LEFT, 30).wheel_angle == pytest.approx(-c_angle(352))


def test_fast_steer_rate_and_cap_511():
    fast = CAR_PAD_FASTSTEER
    assert pad(fast | CAR_PAD_RIGHT).wheel_angle == pytest.approx(c_angle(64))
    assert frames(fast | CAR_PAD_RIGHT, 30).wheel_angle == pytest.approx(c_angle(511))


def test_releasing_fast_steer_snaps_to_regular_cap():
    s = frames(CAR_PAD_FASTSTEER | CAR_PAD_RIGHT, 30)
    assert pad(CAR_PAD_RIGHT, s).wheel_angle == pytest.approx(c_angle(352))


def test_centering_64_per_frame():
    s = PadState(wheel_angle=c_angle(200))
    assert pad(0, s).wheel_angle == pytest.approx(c_angle(136))
    s = PadState(wheel_angle=-c_angle(200))
    assert pad(0, s).wheel_angle == pytest.approx(-c_angle(136))
    assert pad(0, PadState(wheel_angle=c_angle(64))).wheel_angle == 0.0


def test_steering_is_dt_correct():
    two_half = frames(CAR_PAD_RIGHT, 1)
    half = process_car_pad(PadState(), CAR_PAD_RIGHT, FRAME / 2)
    half = process_car_pad(half, CAR_PAD_RIGHT, FRAME / 2)
    assert half.wheel_angle == pytest.approx(two_half.wheel_angle)


def _c_analog_units(stick, fast):
    """The C integer analogue curve, for comparing against the float port."""
    s = int(stick)
    if fast:
        s *= (s * s) // 60
        a = (s * 0x88888889) >> 32
    else:
        s *= (s * s) // 80
        a = (s * 0x66666667) >> 32
    a = (a >> 5) - (s >> 31) & 0xFFFC
    # then the C's per-frame centering (no D-pad bit held)
    return 0 if a < 65 else a - 64


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("stick", [127, 100, 64])
def test_analog_cubic_curve_matches_c_within_quantisation(stick, fast):
    bits = CAR_PAD_FASTSTEER if fast else 0
    got = pad(bits, analog_steer=stick / 127.0).wheel_angle
    want = c_angle(_c_analog_units(stick, fast))
    assert got == pytest.approx(want, abs=c_angle(8))
    neg = pad(bits, analog_steer=-stick / 127.0).wheel_angle
    assert neg == pytest.approx(-got)


def test_analog_small_stick_is_deadzoned():
    assert pad(0, analog_steer=0.3).wheel_angle == 0.0


# -- autoBrake ----------------------------------------------------------------

def test_auto_brake_counts_frames_while_steering_and_resets():
    assert frames(CAR_PAD_RIGHT, 5).auto_brake == pytest.approx(5)
    s = frames(CAR_PAD_RIGHT, 5)
    assert pad(0, s).auto_brake == 0.0
    assert pad(CAR_PAD_BRAKE, s).auto_brake == 0.0  # not tied to the brake


def test_auto_brake_capped_at_90_before_increment():
    assert pad(CAR_PAD_LEFT, PadState(auto_brake=200)).auto_brake == pytest.approx(91)


def test_auto_brake_analog_threshold_270():
    assert pad(0, PadState(auto_brake=10), analog_steer=0.5).auto_brake == 0.0
    assert pad(0, PadState(auto_brake=10), analog_steer=1.0).auto_brake == pytest.approx(11)


# -- thrust -------------------------------------------------------------------

def test_accel_thrust():
    assert pad(CAR_PAD_ACCEL, power_ratio=4096).thrust == pytest.approx(4915 / ONE)
    assert pad(CAR_PAD_ACCEL, power_ratio=2048).thrust == pytest.approx(4915 / ONE / 2)


def test_brake_is_flat_5000_when_moving_forward_or_slow():
    want = -5000 / ONE
    assert pad(CAR_PAD_BRAKE, wheel_speed=0.0).thrust == pytest.approx(want)
    assert pad(CAR_PAD_BRAKE, wheel_speed=20.0).thrust == pytest.approx(want)


def test_brake_reverse_thrust_tapers_with_reverse_speed():
    # rws = wheel_speed * 1500 / 1024 / ONE = -100 -> (178 * -4778) / 256
    ws = c_wheel_speed(-100 * 1024 * ONE / 1500)
    got = pad(CAR_PAD_BRAKE, wheel_speed=ws).thrust
    assert got == pytest.approx(178 * -4778 / 256 / ONE)
    ws_top = c_wheel_speed(-278 * 1024 * ONE / 1500)
    assert pad(CAR_PAD_BRAKE, wheel_speed=ws_top).thrust == pytest.approx(0.0, abs=1e-12)


def test_brake_wins_over_accel():
    assert pad(CAR_PAD_BRAKE | CAR_PAD_ACCEL).thrust < 0


def test_gear_change_cuts_thrust_to_one_unit_for_hnd_type_0():
    assert pad(CAR_PAD_ACCEL, changing_gear=True).thrust == pytest.approx(1 / ONE)
    s = pad(CAR_PAD_ACCEL, changing_gear=True, hnd_type=1)
    assert s.thrust == pytest.approx(4915 / ONE)


# -- wheelspin / handbrake ---------------------------------------------------

def test_wheelspin_and_cutoff_above_452952():
    assert pad(CAR_PAD_WHEELSPIN).wheelspin is True
    assert pad(CAR_PAD_WHEELSPIN).thrust == 0.0  # wheelspin alone sets no thrust
    fast = c_wheel_speed(452953)
    s = pad(CAR_PAD_WHEELSPIN, wheel_speed=fast)
    assert s.wheelspin is False
    assert s.thrust == pytest.approx(4915 / ONE)  # C ors in CAR_PAD_ACCEL


def test_handbrake_keeps_previous_wheelspin():
    s = pad(CAR_PAD_HANDBRAKE, PadState(wheelspin=True))
    assert s.handbrake is True
    assert s.wheelspin is True
    assert pad(0, s).handbrake is False


# -- GetFrictionScalesDriver1 -------------------------------------------------

def fs(**kw):
    args = dict(thrust=0.0, wheelspin=False, handbrake=False, auto_brake=0.0,
                wheel_speed=5.0)
    args.update(kw)
    return get_friction_scales_driver1(**args)


def test_front_fs_by_thrust_sign():
    for thrust, front in ((1.0, 820), (0.0, 937), (-1.0, 1453)):
        r = fs(thrust=thrust, wheel_speed=0.5)
        assert r.front == pytest.approx(front * 35 / 32 / ONE)
        assert r.rear == pytest.approx((0x780 - front) / ONE)


def test_auto_brake_adds_front_grip_capped_at_13():
    assert fs(auto_brake=4).front == pytest.approx((937 + 4 * 45) * 35 / 32 / ONE)
    assert fs(auto_brake=50).front == pytest.approx((937 + 13 * 45) * 35 / 32 / ONE)
    assert fs(auto_brake=50, wheel_speed=0.0).front == pytest.approx(937 * 35 / 32 / ONE)
    assert fs(auto_brake=50, hnd_type=1).front == pytest.approx(937 * 29 / 32 * 35 / 32 / ONE)


def test_braking_above_41943_locks_all_wheels_and_is_aggressive():
    lock = c_wheel_speed(41943)
    r = fs(thrust=-1.0, wheel_speed=lock + 0.01)
    assert r.locked == (True, True, True, True)
    assert r.front == pytest.approx(1453 * 10 / 8 / ONE)
    assert r.rear == pytest.approx(467 * 10 / 8 / ONE)
    r = fs(thrust=-1.0, wheel_speed=lock - 0.01)
    assert r.locked == (False, False, False, False)


def test_handbrake_locks_rears_zeroes_thrust_and_shifts_grip():
    r = fs(thrust=1.0, handbrake=True, wheelspin=True)
    assert r.locked == (False, True, False, True)
    assert r.thrust == 0.0
    assert r.wheelspin is False
    assert r.front == pytest.approx((820 + 656) * 35 / 32 / ONE)  # thrust > 0 case
    stopped = fs(handbrake=True, wheel_speed=0.0)
    assert stopped.front == pytest.approx((937 - 375) * 35 / 32 / ONE)
    assert fs(thrust=-1.0, handbrake=True).thrust == -1.0  # braking survives


def test_wheelspin_shifts_grip_and_forces_5000_thrust():
    r = fs(thrust=1.0, wheelspin=True)
    assert r.front == pytest.approx((820 + 600) * 35 / 32 / ONE)
    assert r.rear == pytest.approx(500 / ONE)
    assert r.thrust == pytest.approx(5000 / ONE)


def test_rolling_backwards_takes_400_off_front():
    assert fs(wheel_speed=-1.0).front == pytest.approx((937 - 400) * 35 / 32 / ONE)


def test_front_on_grass_uses_39_over_32():
    assert fs(front_on_grass=True).front == pytest.approx(937 * 39 / 32 / ONE)


def test_handling_ratio_and_traction_scale_both_axles():
    r = fs(hnd_type=2, traction=2048)
    assert r.front == pytest.approx(937 * 35 / 32 * 45 / 32 / 2 / ONE)
    assert r.rear == pytest.approx(983 * 45 / 32 / 2 / ONE)


def test_unit_conversions():
    assert c_angle(1024) == pytest.approx(math.pi / 2)
    # 4096ths of a DR2 unit per 30 Hz frame, 150 DR2 units per metre
    assert c_wheel_speed(4096) == pytest.approx(30 / 150)

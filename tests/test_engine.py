"""Gearbox (core.engine)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.engine import (  # noqa: E402
    GearboxState,
    choose_gear,
    gear_top_speed,
    step_gearbox,
    wheel_rpm,
)

DT = 1.0 / 60.0


def test_gears_get_longer_and_top_gear_reaches_top_speed():
    tops = [gear_top_speed(g) for g in range(len(config.GEAR_RATIOS))]
    assert tops == sorted(tops)
    for low, high in zip(tops, tops[1:], strict=False):
        assert 1.15 < high / low < 1.7  # evenly spaced, no huge gaps
    top = config.CAR_TOP_SPEED_KMH / 3.6
    assert wheel_rpm(top, len(tops) - 1) < config.REDLINE_RPM


def test_shift_points_have_hysteresis():
    g1_top = gear_top_speed(0)
    assert choose_gear(0, g1_top * 0.98) == 0
    assert choose_gear(0, g1_top * 1.02) == 1
    assert choose_gear(1, g1_top * 0.9) == 1  # no hunting just below the shift
    assert choose_gear(3, 3.0) == 0
    assert choose_gear(2, -4.0) == 0  # reverse uses first


def test_upshift_cuts_thrust_for_the_shift_time():
    s = GearboxState(gear=0)
    s = step_gearbox(s, DT, wheel_speed=gear_top_speed(0) * 1.02, rear_grounded=True)
    assert s.gear == 1 and s.changing_gear
    for _ in range(round(config.SHIFT_TIME_S / DT) + 1):
        s = step_gearbox(s, DT, wheel_speed=gear_top_speed(0) * 1.02, rear_grounded=True)
    assert not s.changing_gear


def test_gear_holds_in_the_air():
    s = step_gearbox(GearboxState(gear=2), DT, wheel_speed=1.0, rear_grounded=False)
    assert s.gear == 2


def test_rpm_scales_with_speed_and_ratio():
    assert wheel_rpm(20.0, 0) == pytest.approx(2.0 * wheel_rpm(10.0, 0))
    assert wheel_rpm(20.0, 0) > wheel_rpm(20.0, 1)

"""Gamepad steering smoothing (core.steer_filter): ramps by its time constants,
frame-rate independent, passthrough at zero settings."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.steer_filter import SteerFilter, SteerParams, shape  # noqa: E402


def _run(f, raw, seconds, fps):
    for _ in range(round(seconds * fps)):
        v = f.update(raw, 1.0 / fps)
    return v


def test_zero_settings_pass_the_stick_through():
    f = SteerFilter(SteerParams())
    for x in (0.0, 0.3, -0.7, 1.0, -1.0, 0.02):
        assert f.update(x, 1 / 60) == x


def test_step_rises_and_returns_by_their_time_constants():
    f = SteerFilter(SteerParams(smooth_s=0.2, return_s=0.1))
    assert _run(f, 1.0, 0.2, 60) == pytest.approx(1.0 - math.exp(-1.0), abs=1e-6)
    _run(f, 1.0, 3.0, 60)
    assert _run(f, 0.0, 0.1, 60) == pytest.approx(math.exp(-1.0), abs=1e-3)


@pytest.mark.parametrize("fps", [30, 144])
def test_same_response_at_any_frame_rate(fps):
    p = SteerParams(smooth_s=0.15, return_s=0.1)
    a = _run(SteerFilter(p), 0.8, 0.5, 60)
    b = _run(SteerFilter(p), 0.8, 0.5, fps)
    assert a == pytest.approx(b, abs=1e-9)


def test_reversing_through_centre_uses_the_return_time():
    f = SteerFilter(SteerParams(smooth_s=1.0, return_s=0.05))
    _run(f, 1.0, 10.0, 60)
    assert _run(f, -1.0, 0.1, 60) < 0.0       # back through centre at the return time
    slow = SteerFilter(SteerParams(smooth_s=1.0, return_s=1.0))
    _run(slow, 1.0, 10.0, 60)
    assert _run(slow, -1.0, 0.1, 60) > 0.5


def test_deadzone_expo_and_speed_reduction():
    p = SteerParams(deadzone=0.1, expo=1.0)
    assert shape(0.08, p) == 0.0 and shape(1.0, p) == 1.0 and shape(-1.0, p) == -1.0
    assert shape(0.55, p) == pytest.approx(0.125)       # (0.5)^3
    f = SteerFilter(SteerParams(speed_scale=0.25, speed_ref=40.0))
    assert f.update(1.0, 1 / 60, speed=80.0) == pytest.approx(0.75)
    assert f.update(1.0, 1 / 60, speed=0.0) == pytest.approx(1.0)

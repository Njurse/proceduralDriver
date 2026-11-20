"""Behaviour checks for the float Driver 2 car (core.redriver2_physics)."""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.car_profile import load_cosmetics  # noqa: E402
from core.redriver2_physics import TICK_HZ, PhysicsWorld, Vehicle  # noqa: E402

DT = 1.0 / 60.0


def _inp(steer=0.0, throttle=0.0, brake=0.0, handbrake=False):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=brake,
                           handbrake=handbrake, fast_steer=False, wheelspin=False)


def _car():
    v = Vehicle(load_cosmetics())
    world = PhysicsWorld()
    world.add(v)
    return v, world


def _run(v, world, seconds, inp):
    v.process_input(inp)
    for _ in range(round(seconds / DT)):
        world.step(DT)


def _cruise(speed):
    """Accelerate to ``speed`` m/s, then coast for half a second."""
    v, world = _car()
    v.process_input(_inp(throttle=1.0))
    while v.forward_speed < speed:
        world.step(DT)
    _run(v, world, 0.5, _inp())
    return v, world


def _roll_deg(v):
    return math.degrees(math.asin(-v.rot[1, 0]))


def _drift_deg(v):
    """Angle between the heading and the direction of travel."""
    h = v.heading
    fwd = v.vel[0] * math.sin(h) + v.vel[2] * math.cos(h)
    side = v.vel[0] * math.cos(h) - v.vel[2] * math.sin(h)
    return math.degrees(math.atan2(side, fwd))


def test_fixed_tick_is_independent_of_frame_rate():
    v, world = _car()
    assert world.step(1.0 / 30.0) == round(TICK_HZ / 30.0)
    a, wa = _car()
    b, wb = _car()
    a.process_input(_inp(throttle=1.0))
    b.process_input(_inp(throttle=1.0))
    for _ in range(120):
        wa.step(1.0 / 60.0)
    for _ in range(60):
        wb.step(1.0 / 30.0)
    np.testing.assert_allclose(a.pos, b.pos, atol=1e-9)


def test_settles_on_suspension_at_rest():
    v, world = _car()
    v.pos[1] += 0.05  # drop it from slightly too high
    _run(v, world, 4.0, _inp())
    assert np.all(v.vel == 0.0) and np.all(v.avel == 0.0)  # GlobalTimeStep rest snap
    # the C rest snap can freeze the last small bounce a couple of cm high
    assert v.pos[1] == pytest.approx(v.rest_height(), abs=0.03)
    assert 0.6 < v.pos[1] < 0.9  # body origin height, metres
    assert all(w.compression > 0.05 for w in v.wheels)


def test_full_throttle_reaches_plausible_top_speed():
    v, world = _car()
    _run(v, world, 10.0, _inp(throttle=1.0))
    kmh = v.forward_speed * 3.6
    assert 120.0 < kmh < 180.0
    assert abs(v.heading) < math.radians(2.0)  # tracks straight
    assert v.rot[1, 1] > 0.98  # upright


def test_steering_turns_without_yaw_assist():
    for steer, sign in ((1.0, 1.0), (-1.0, -1.0)):
        v, world = _cruise(10.0)
        h0 = v.heading
        _run(v, world, 0.6, _inp(steer=steer))
        turned = (v.heading - h0) * sign
        assert turned > math.radians(10.0)
        assert v.avel[1] * sign > 0.3  # rad/s yaw rate from tyre forces alone
        assert abs(_roll_deg(v)) < 15.0


def _max_drift(handbrake):
    v, world = _cruise(12.0)
    v.process_input(_inp(steer=1.0, handbrake=handbrake))
    drift = rear_slip = 0.0
    for _ in range(45):
        world.step(DT)
        if handbrake:
            assert v.wheels[1].locked and v.wheels[3].locked
            assert not v.wheels[0].locked and not v.wheels[2].locked
        drift = max(drift, abs(_drift_deg(v)))
        rear_slip = max(rear_slip, v.wheels[1].slip_lateral, v.wheels[3].slip_lateral)
    return drift, rear_slip


def test_handbrake_locks_rears_and_slides_the_tail():
    drift_hb, slip_hb = _max_drift(handbrake=True)
    drift_steer, slip_steer = _max_drift(handbrake=False)
    assert drift_hb > drift_steer + 2.0  # degrees of extra body slip angle
    assert slip_hb > slip_steer

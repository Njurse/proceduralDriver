"""Behaviour checks for the float Driver 2 car (core.redriver2_physics)."""

import math
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.car_profile import load_cosmetics  # noqa: E402
from core.redriver2_physics import TICK_HZ, PhysicsWorld, Vehicle  # noqa: E402

DT = 1.0 / 60.0


def _inp(steer=0.0, throttle=0.0, brake=0.0, handbrake=False, fast_steer=False):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=brake,
                           handbrake=handbrake, fast_steer=fast_steer, wheelspin=False)


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
    assert 140.0 < v.forward_speed * 3.6 < 180.0
    _run(v, world, 30.0, _inp(throttle=1.0))
    kmh = v.forward_speed * 3.6
    assert kmh == pytest.approx(config.CAR_TOP_SPEED_KMH, abs=5.0)
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
        # as a fraction of speed: the locked rears also scrub the speed off
        speed = max(math.hypot(v.vel[0], v.vel[2]), 1e-6)
        rear = max(v.wheels[1].slip_lateral, v.wheels[3].slip_lateral) / speed
        rear_slip = max(rear_slip, rear)
    return drift, rear_slip


def test_handbrake_locks_rears_and_slides_the_tail():
    drift_hb, slip_hb = _max_drift(handbrake=True)
    drift_steer, slip_steer = _max_drift(handbrake=False)
    assert drift_hb > drift_steer + 2.0  # degrees of extra body slip angle
    assert slip_hb > slip_steer


def _pitch_deg(v):
    """Nose-up is negative: rot[2, 1] is the forward axis' height."""
    return math.degrees(math.asin(v.rot[2, 1]))


# Stock game handling probe (JERICHO HOOKS.md / dev log, Chicago model 1,
# same power/mass/traction): speed in u/frame at 1, 2, 3 and 4 s of full gas.
# The C-exact model matched these to ~2 u/frame; config.GEAR_THRUST_SCALES
# retunes the launch on purpose, so the curve only has to stay near them.
STOCK_PROBE_SPEEDS = (67, 98, 130, 147)


def test_launch_keeps_the_front_wheels_down():
    v, world = _car()
    v.process_input(_inp(throttle=1.0))
    worst = 0.0
    for _ in range(round(4.0 / DT)):
        world.step(DT)
        assert v.compression[0] > 0.0 or v.compression[2] > 0.0
        worst = min(worst, _pitch_deg(v))
    assert worst > -4.0


def test_launch_stays_near_the_stock_probe():
    v, world = _car()
    v.process_input(_inp(throttle=1.0))
    t60 = None
    speeds = []
    for i in range(1, round(10.0 / DT) + 1):
        world.step(DT)
        if t60 is None and v.forward_speed >= 26.8224:
            t60 = i * DT
        if i % round(1.0 / DT) == 0 and len(speeds) < 4:
            speeds.append(v.forward_speed * 150.0 / 30.0)  # u/frame
    assert 2.9 < t60 < 3.4  # 0-60 mph, seconds
    for got, stock in zip(speeds, STOCK_PROBE_SPEEDS, strict=True):
        assert stock * 0.8 < got < stock * 1.15


def test_upshifts_cut_the_thrust():
    v, world = _car()
    v.process_input(_inp(throttle=1.0))
    cuts = 0
    gears = []
    for _ in range(round(12.0 / DT)):
        world.step(DT)
        cuts += v.changing_gear
        if not gears or gears[-1] != v.gear:
            gears.append(v.gear)
    assert gears == [0, 1, 2, 3]
    assert cuts > 0


@pytest.mark.parametrize("speed", [15.0, 30.0, 50.0])
@pytest.mark.parametrize("controls", [
    dict(throttle=1.0), dict(throttle=0.0), dict(handbrake=True),
    dict(throttle=1.0, fast_steer=True)], ids=["gas", "coast", "handbrake", "fast_steer"])
def test_full_lock_on_flat_ground_does_not_tip(speed, controls):
    v, world = _car()
    v.process_input(_inp(throttle=1.0))
    while v.forward_speed < speed:
        world.step(DT)
    v.process_input(_inp(steer=1.0, **controls))
    for _ in range(round(4.0 / DT)):
        world.step(DT)
        assert abs(_roll_deg(v)) < 15.0
        assert v.rot[1, 1] > 0.95
    assert abs(_drift_deg(v)) > 10.0 or v.forward_speed < 15.0  # slides, scrubs speed


def test_body_model_sits_at_the_cog():
    """cars.c DrawCar draws the body at where + R * cog; FixCarCos shifts the
    cog against the wheels (handling.c UpdateCarPoints)."""
    import dataclasses

    from core.car_forces import CarGeometry

    cos = load_cosmetics()
    wz = (cos.wheel_disp[0][2], cos.wheel_disp[1][2])
    dz = float(int(-(wz[0] + wz[1] - 14) / 2))
    geo = CarGeometry.from_cosmetics(dataclasses.replace(cos, cog=(0, -89, 6)))
    assert geo.model_offset == pytest.approx(np.array([0.0, 89.0, dz - 6.0]) / 150.0)

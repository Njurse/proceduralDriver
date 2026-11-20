"""Engine sound model and loop rendering (core.engine_sound, core.vehicle_audio)."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.engine import gear_top_speed, wheel_rpm  # noqa: E402
from core.engine_sound import EngineSound, SoundInput, crossfade, target_rpm  # noqa: E402
from core.vehicle_audio import Loop, seamless  # noqa: E402

DT = 1.0 / 60.0


def _settle(model, inp, seconds=2.0):
    mix = None
    for _ in range(round(seconds / DT)):
        mix = model.update(DT, inp)
    return mix


def test_idle_floor_and_idle_loop_at_rest():
    mix = _settle(EngineSound(), SoundInput())
    assert mix.rpm == pytest.approx(config.IDLE_RPM)
    assert mix.idle_level > 0.5 and mix.engine_level < 0.01


def test_clutch_couples_rpm_to_wheel_speed():
    speed = 15.0
    assert target_rpm(SoundInput(speed=speed, gear=1, throttle=1.0)) == pytest.approx(
        wheel_rpm(speed, 1))
    # standing start: the throttle revs a free engine
    launch = target_rpm(SoundInput(speed=0.0, throttle=1.0))
    assert launch == pytest.approx(config.FREE_REV_RPM)
    # wheelspin lets the revs flare above the road speed
    spin = target_rpm(SoundInput(speed=8.0, gear=0, throttle=1.0, wheelspin=True))
    assert spin > wheel_rpm(8.0, 0)


def test_rpm_is_capped_at_redline():
    assert target_rpm(SoundInput(speed=80.0, gear=0, throttle=1.0)) == config.REDLINE_RPM


def test_upshift_drops_rpm_quickly_and_goes_off_load():
    model = EngineSound()
    speed = gear_top_speed(0)
    _settle(model, SoundInput(speed=speed, gear=0, throttle=1.0))
    before = model.rpm
    mix = None
    for _ in range(round(0.15 / DT)):
        mix = model.update(DT, SoundInput(speed=speed, gear=1, throttle=1.0,
                                          changing_gear=True))
    after = wheel_rpm(speed, 1)
    assert mix.rpm - after < 0.1 * (before - after)  # >90% of the drop in 0.15 s
    assert mix.load < 0.3


def test_levels_move_smoothly():
    model = EngineSound()
    prev = model.update(DT, SoundInput())
    for i in range(240):
        inp = SoundInput(speed=i * 0.1, gear=0, throttle=1.0 if i < 120 else 0.0)
        mix = model.update(DT, inp)
        # a free rev-up from idle is the fastest move: ~15% of 3650 rpm a frame
        assert abs(mix.rpm - prev.rpm) < 600.0
        assert abs(mix.engine_level - prev.engine_level) < 0.25
        prev = mix


def test_on_load_is_louder_than_off_load():
    on = _settle(EngineSound(), SoundInput(speed=20.0, gear=1, throttle=1.0))
    off = _settle(EngineSound(), SoundInput(speed=20.0, gear=1, throttle=0.0))
    assert on.engine_level > off.engine_level * 1.5


def test_crossfade_keeps_power():
    for rpm in (500.0, 1000.0, 1300.0, 3000.0):
        idle, engine = crossfade(rpm)
        assert idle * idle + engine * engine == pytest.approx(1.0)


def test_skid_hysteresis():
    model = EngineSound()
    assert _settle(model, SoundInput(speed=20.0, slip=0.8), 0.5).skid_level == 0.0
    assert _settle(model, SoundInput(speed=20.0, slip=2.0), 0.5).skid_level > 0.2
    assert _settle(model, SoundInput(speed=20.0, slip=0.8), 0.5).skid_level > 0.1  # held
    assert _settle(model, SoundInput(speed=20.0, slip=0.2), 1.5).skid_level < 0.01


def test_loop_render_is_continuous_across_blocks():
    t = np.arange(4000)
    loop = Loop(seamless(np.sin(t * 2 * math.pi / 100.0).astype(np.float32), 300))
    a = loop.render(512, 1.0, 1.5)
    b = loop.render(512, 1.5, 1.5)
    joined = np.concatenate((a, b))
    assert np.max(np.abs(np.diff(joined))) < 0.2  # no jump at the block edge

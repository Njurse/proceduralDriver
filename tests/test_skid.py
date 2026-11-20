"""When the skid loop plays: core.vehicle_audio.skid_slip on the real car."""

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.car_profile import load_cosmetics  # noqa: E402
from core.engine_sound import EngineSound  # noqa: E402
from core.redriver2_physics import PhysicsWorld, Vehicle  # noqa: E402
from core.vehicle_audio import skid_slip, sound_input  # noqa: E402

DT = 1.0 / 60.0


def _inp(steer=0.0, throttle=0.0, brake=0.0, handbrake=False, analog=False):
    return SimpleNamespace(steer=steer, throttle=throttle, brake=brake, handbrake=handbrake,
                           fast_steer=False, wheelspin=False, analog=analog)


def _drive(*phases):
    """Run (seconds, input) phases; return the loudest skid level of each phase."""
    v = Vehicle(load_cosmetics())
    world = PhysicsWorld()
    world.add(v)
    sound = EngineSound()
    loudest = []
    for seconds, inp in phases:
        v.process_input(inp)
        level = 0.0
        for _ in range(round(seconds / DT)):
            world.step(DT)
            level = max(level, sound.update(DT, sound_input(v)).skid_level)
        loudest.append(level)
    return loudest


def test_no_skid_at_rest():
    assert _drive((2.0, _inp()), (2.0, _inp(handbrake=True))) == [0.0, 0.0]


def test_no_skid_launching_and_cruising_straight():
    assert _drive((4.0, _inp(throttle=1.0)), (2.0, _inp(throttle=0.4))) == [0.0, 0.0]


def test_no_skid_in_a_gentle_turn():
    launch, turn = _drive((2.5, _inp(throttle=1.0)),
                          (3.0, _inp(steer=0.3, throttle=0.4, analog=True)))
    assert launch == 0.0 and turn == 0.0


def test_skid_in_a_hard_handbrake_turn():
    _, turn = _drive((4.0, _inp(throttle=1.0)), (1.5, _inp(steer=1.0, handbrake=True)))
    assert turn > 0.5 * config.SKID_GAIN


def test_skid_when_braking_locks_the_wheels():
    _, braking = _drive((4.0, _inp(throttle=1.0)), (1.0, _inp(brake=1.0)))
    assert braking > 0.5 * config.SKID_GAIN


def _car(speed, slide, compression=0.1, locked=False, wheelspin=False):
    wheel = SimpleNamespace(compression=compression, locked=locked, slip_lateral=slide,
                            slip_longitudinal=slide if locked else 0.0)
    return SimpleNamespace(vel=np.array([speed, 0.0, 0.0]), wheelspin=wheelspin,
                           wheels=[wheel] * 4)


def test_no_skid_in_the_air_or_crawling():
    assert skid_slip(_car(20.0, 10.0)) == 10.0
    assert skid_slip(_car(20.0, 10.0, compression=0.0)) == 0.0  # airborne
    assert skid_slip(_car(config.SKID_SPEED_MIN * 0.9, 10.0)) == 0.0
    assert skid_slip(_car(1.0, 10.0, wheelspin=True)) == 10.0  # burnout on the spot


def test_front_slide_alone_is_not_a_skid():
    car = _car(20.0, 0.0)
    car.wheels = [SimpleNamespace(compression=0.1, locked=False, slip_lateral=5.0 if i % 2 == 0
                                  else 0.0, slip_longitudinal=0.0) for i in range(4)]
    assert skid_slip(car) == 0.0

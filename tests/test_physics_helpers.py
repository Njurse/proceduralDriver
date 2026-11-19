import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.car_profile import load_cosmetics
from core.physics_helpers import (steering_update, auto_brake_update,
                                 thrust_from_input, friction_scales)


def test_steering_increment_regular():
    assert steering_update(0, 1.0, False) == 32


def test_steering_fast():
    assert steering_update(0, 1.0, True) == 64


def test_steering_centering():
    # Both the digital increment (32) and the centering step (16) apply in
    # one pass when no input is given.
    assert steering_update(100, 0.0, False) == 52
    assert steering_update(-100, 0.0, False) == -52


def test_steering_clamp():
    # repeated increments should clamp at max
    v = 0
    for _ in range(100):
        v = steering_update(v, 1.0, False)
    assert v == 511


def test_auto_brake_cap():
    ab = 0
    for _ in range(20):
        ab = auto_brake_update(ab, 1.0, True)
    assert ab == 90


def test_auto_brake_decay():
    assert auto_brake_update(30, 0.0, True) == 27


def test_auto_brake_disabled():
    assert auto_brake_update(30, 1.0, False) == 30


def test_thrust_accel():
    cos = load_cosmetics()
    t = thrust_from_input(cos.power_ratio, 1.0, 0.0, False)
    assert t > 0


def test_thrust_brake():
    cos = load_cosmetics()
    t = thrust_from_input(cos.power_ratio, 0.0, 1.0, False)
    assert t < 0


def test_thrust_wheelspin():
    cos = load_cosmetics()
    t1 = thrust_from_input(cos.power_ratio, 1.0, 0.0, False)
    t2 = thrust_from_input(cos.power_ratio, 1.0, 0.0, True)
    assert t2 > t1


def test_friction_handbrake():
    cos = load_cosmetics()
    ht = type("H", (), {"friction_scale_ratio": 32, "auto_brake_on": 1})()
    f1, r1 = friction_scales(cos, ht, 0, False, False, 0, False)
    f2, r2 = friction_scales(cos, ht, 0, True, False, 0, False)
    assert r2 < r1
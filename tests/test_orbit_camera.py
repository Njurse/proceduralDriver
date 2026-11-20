"""Right-stick orbit camera math (core.orbit_camera)."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.orbit_camera import (  # noqa: E402
    OrbitCamera,
    approach_angle,
    orbit_offset,
    stick_heading,
)

DT = 1.0 / 60.0


# offsets are in the car's yaw frame: x right, z forward
@pytest.mark.parametrize("x, y, right, fwd", [
    (-1.0, 0.0, -10.0, 0.0),  # stick left: camera on the car's left
    (1.0, 0.0, 10.0, 0.0),    # stick right: camera on the car's right
    (0.0, 1.0, 0.0, 10.0),    # stick down: in front, looking back
    (0.0, -1.0, 0.0, -10.0),  # stick up: behind
])
def test_stick_picks_the_side(x, y, right, fwd):
    ox, _, oz = orbit_offset(stick_heading(x, y), 10.0, 3.0)
    assert ox == pytest.approx(right, abs=1e-9)
    assert oz == pytest.approx(fwd, abs=1e-9)


def test_small_deflection_is_ignored():
    assert stick_heading(0.1, -0.1) is None


def test_lerp_takes_the_short_way_across_the_wrap():
    a = approach_angle(math.radians(170.0), math.radians(-170.0), DT)
    assert math.degrees(a) > 170.0 or math.degrees(a) < -170.0


def test_swings_quickly_and_returns_on_release():
    cam = OrbitCamera()
    for _ in range(18):  # 0.3 s
        cam.update(DT, -1.0, 0.0, False)
    assert math.degrees(cam.angle) == pytest.approx(-90.0, abs=5.0)
    for _ in range(30):
        cam.update(DT, 0.0, 0.0, False)
    assert abs(math.degrees(cam.angle)) < 2.0


def test_click_latches_and_second_click_releases():
    cam = OrbitCamera()
    for _ in range(30):
        cam.update(DT, 1.0, 0.0, False)
    cam.update(DT, 1.0, 0.0, True)
    cam.update(DT, 1.0, 0.0, False)
    for _ in range(30):
        cam.update(DT, 0.0, 0.0, False)
    assert math.degrees(cam.angle) == pytest.approx(90.0, abs=1.0)
    cam.update(DT, 0.0, 0.0, True)
    for _ in range(30):
        cam.update(DT, 0.0, 0.0, False)
    assert abs(math.degrees(cam.angle)) < 2.0

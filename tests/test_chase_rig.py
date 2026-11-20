"""The chase camera's follow rig (core.chase_rig): settles, frame-rate
independent, swings and leans the right way, widens with speed."""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.chase_rig import ChaseRig, RigParams  # noqa: E402


def _drive(fps, seconds, speed=lambda t: 30.0, yaw_rate=lambda t: 0.0):
    rig, dt, h, out = ChaseRig(), 1.0 / fps, 0.0, []
    for i in range(round(seconds * fps) + 1):
        t = i * dt
        out.append((t, *rig.update(dt, h, 0.0, speed(t), speed(t))))
        h += yaw_rate(t) * dt
    return out, h


def test_settles_behind_with_no_jitter_on_a_straight():
    out, _ = _drive(60, 4.0)
    tail = out[-60:]
    assert all(abs(y) < 1e-9 for _, y, *_ in tail)
    d = [d for _, _, d, _, _ in tail]
    assert max(d) - min(d) < 1e-3
    p = RigParams()
    assert d[-1] == pytest.approx(p.dist + p.pull_back * 30.0 / p.speed_ref, abs=0.01)


@pytest.mark.parametrize("fps", [30, 144])
def test_same_motion_at_any_frame_rate(fps):
    def turn(t):
        return 0.6 if 1.0 < t < 2.5 else 0.0

    def gas(t):
        return 10.0 + 8.0 * t

    ref, _ = _drive(60, 3.0, gas, turn)
    got, _ = _drive(fps, 3.0, gas, turn)
    for t_ref in (1.5, 2.0, 2.8):
        a = min(ref, key=lambda r: abs(r[0] - t_ref))
        b = min(got, key=lambda r: abs(r[0] - t_ref))
        for i, tol in ((1, 0.03), (2, 0.08), (3, 0.3), (4, 0.25)):
            assert a[i] == pytest.approx(b[i], abs=tol), (fps, t_ref, i)


def test_swings_wide_and_leans_into_a_right_turn():
    out, h = _drive(60, 2.0, yaw_rate=lambda t: 0.8)
    _, yaw, _, _, roll = out[-1]
    assert 0.02 < h - yaw < 0.3       # trails the heading a little
    assert 0.5 < roll <= RigParams().roll_max


def test_speed_widens_the_view_and_pulls_back():
    slow, _ = _drive(60, 5.0, speed=lambda t: 5.0)
    fast, _ = _drive(60, 5.0, speed=lambda t: 50.0)
    assert fast[-1][3] > slow[-1][3] + 5.0
    assert fast[-1][2] > slow[-1][2] + 1.0


def test_hitches_and_resets_stay_bounded():
    rig = ChaseRig()
    rig.update(1 / 60, 0.0, 0.0, 20.0, 20.0)
    yaw, dist, fov, roll = rig.update(5.0, 3.0, 0.0, 20.0, 60.0)
    assert all(math.isfinite(v) for v in (yaw, dist, fov, roll))
    assert dist < RigParams().dist + 4.0 and abs(roll) <= RigParams().roll_max
    rig.reset()
    assert rig.update(1 / 60, 1.0, 0.5, 0.0, 0.0)[0] == pytest.approx(0.5)

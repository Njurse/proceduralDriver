"""gamesnd.c gearbox port (core.engine)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.engine import (  # noqa: E402
    AIRBORNE_REVS,
    MAX_REV_DROP,
    EngineState,
    control_car_revs,
    engine_revs,
)
from core.units import dr2_velocity  # noqa: E402

DT = 1.0 / 30.0


def _ws(gear_units):
    """Wheel speed (m/s) for a speed in gearbox units (wheel_speed >> 11)."""
    return dr2_velocity(gear_units * 2048.0)


# upshifts happen above hi_ws, so start every case in first gear just past it
@pytest.mark.parametrize("ws, gear", [(50, 0), (200, 1), (300, 2), (400, 3)])
def test_gear_follows_wheel_speed(ws, gear):
    g, revs = engine_revs(0, _ws(ws), accelerating=True)
    assert g == gear
    assert revs == pytest.approx(ws * (144, 90, 60, 48)[gear])


def test_upshift_drops_revs_and_flags_a_gear_change():
    s = EngineState(revs=160 * 144, gear=0)
    s = control_car_revs(s, DT, thrust=1.0, wheelspin=False, rear_grounded=True,
                         wheel_speed=_ws(170))
    assert s.gear == 1
    assert s.changing_gear
    assert s.revs == pytest.approx(160 * 144 - MAX_REV_DROP)
    for _ in range(10):
        s = control_car_revs(s, DT, thrust=1.0, wheelspin=False, rear_grounded=True,
                             wheel_speed=_ws(170))
    assert not s.changing_gear
    assert s.revs == pytest.approx(170 * 90)


def test_airborne_rear_wheels_rev_out():
    s = EngineState()
    for _ in range(40):
        s = control_car_revs(s, DT, thrust=1.0, wheelspin=False, rear_grounded=False,
                             wheel_speed=0.0)
    assert s.revs == pytest.approx(AIRBORNE_REVS)
    assert s.gear == 0

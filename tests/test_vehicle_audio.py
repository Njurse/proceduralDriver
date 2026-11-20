"""Revs-driven engine sound math (core.vehicle_audio)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.vehicle_audio import (  # noqa: E402
    IDLE_BANK,
    EngineVolumes,
    bank_weights,
    engine_pitch,
    gain,
    idle_pitch,
    step_volumes,
)

DT = 1.0 / 30.0


def _run(frames, **kw):
    vol = EngineVolumes()
    for _ in range(frames):
        vol = step_volumes(vol, DT, **kw)
    return vol


def test_idle_takes_over_at_low_revs_without_gas():
    vol = _run(60, revs=3000.0, accelerating=False, wheelspin=False)
    assert vol.idlevol == -6000.0
    assert gain(vol.revsvol) == 0.0
    assert gain(vol.idlevol) > 0.0


def test_gas_brings_in_the_engine_and_fades_idle():
    vol = _run(60, revs=12000.0, accelerating=True, wheelspin=False)
    assert vol.revsvol == -5500.0
    assert gain(vol.revsvol) == pytest.approx(1.0)
    assert vol.idlevol < -8000.0


def test_engine_pitch_follows_revs():
    assert engine_pitch(0.0, -5500.0) == pytest.approx((1500 - 5500 / 64) / 4096)
    assert engine_pitch(24000.0, -5500.0) > 1.4
    assert engine_pitch(24000.0 - 1440.0, -5500.0) < engine_pitch(24000.0, -5500.0)
    assert idle_pitch(0.0) == 1.0


@pytest.mark.parametrize("pitch", [0.1, 1.0, 1.1, 1.37, 3.0])
def test_bank_weights_share_one_volume(pitch):
    w = bank_weights(pitch, IDLE_BANK)
    assert sum(w) == pytest.approx(1.0)
    assert sum(1 for x in w if x > 0.0) <= 2

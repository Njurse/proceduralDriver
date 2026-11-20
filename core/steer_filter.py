"""Gamepad steering feel, before the physics: a deadzone, an expo curve,
less lock at speed, and separate rise / return smoothing.

Pure and dt-correct (frame rate does not change the response); with every
setting at 0 it passes the stick straight through.  main.py applies it to
the analog stick only: the keyboard's digital steer is already ramped by the
physics (the C's pad steering).
"""

import math
from dataclasses import dataclass


@dataclass
class SteerParams:
    smooth_s: float = 0.0       # time constant moving away from centre (rise)
    return_s: float = 0.0       # time constant moving back toward centre
    expo: float = 0.0           # 0 linear .. 1 cubic: finer control near centre
    deadzone: float = 0.0       # stick travel ignored round the centre
    speed_scale: float = 0.0    # lock taken off at speed_ref (0.25: a quarter)
    speed_ref: float = 45.0     # m/s


def shape(x, p):
    """The stick position after deadzone, expo and the speed reduction (no smoothing)."""
    a = abs(x)
    if a <= p.deadzone:
        return 0.0
    a = min(1.0, (a - p.deadzone) / (1.0 - p.deadzone))
    a = (1.0 - p.expo) * a + p.expo * a * a * a
    return math.copysign(a, x)


class SteerFilter:
    """Stateful smoothing of one steering axis; ``reset`` on respawn."""

    def __init__(self, params=None):
        self.p = params or SteerParams()
        self.value = 0.0

    def reset(self):
        self.value = 0.0

    def update(self, raw, dt, speed=0.0):
        p = self.p
        target = shape(max(-1.0, min(1.0, raw)), p)
        target *= 1.0 - p.speed_scale * min(1.0, abs(speed) / p.speed_ref)
        outward = abs(target) > abs(self.value) and target * self.value >= 0.0
        tau = p.smooth_s if outward else p.return_s
        if tau <= 0.0 or dt <= 0.0:
            self.value = target if tau <= 0.0 else self.value
        else:
            self.value += (target - self.value) * (1.0 - math.exp(-dt / tau))
        return self.value

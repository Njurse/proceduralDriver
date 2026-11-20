"""Chase-camera orbit for the right stick: pure angle math, no rendering.

The orbit angle says where the camera sits around the car, measured in the
car's yaw frame: 0 is behind (the chase view), +pi/2 on the car's right side
looking left, -pi/2 on its left side looking right, pi in front looking
back.  The camera always looks at the car.
"""

import math
from dataclasses import dataclass

STICK_DEADZONE = 0.3
ORBIT_TIME_CONSTANT = 0.08  # seconds; ~0.25 s to swing most of the way round


def wrap_angle(a: float) -> float:
    """Angle in [-pi, pi)."""
    return (a + math.pi) % math.tau - math.pi


def stick_heading(x: float, y: float) -> float | None:
    """Orbit angle the stick points at (x right, y down), None when centred.

    Stick left puts the camera on the car's left, right on its right, down in
    front of it, up behind it.
    """
    if math.hypot(x, y) < STICK_DEADZONE:
        return None
    return math.atan2(x, -y)


def approach_angle(current: float, target: float, dt: float,
                   tau: float = ORBIT_TIME_CONSTANT) -> float:
    """Exponential move toward ``target`` along the shorter way round."""
    k = 1.0 - math.exp(-dt / tau)
    return wrap_angle(current + wrap_angle(target - current) * k)


def orbit_offset(angle: float, dist: float, height: float) -> tuple[float, float, float]:
    """Camera position relative to the car in its yaw frame (x right, y up,
    z forward)."""
    return (math.sin(angle) * dist, height, -math.cos(angle) * dist)


@dataclass
class OrbitCamera:
    """Right stick sets the target; release swings back to the chase view.
    A click (R3) latches the current angle, a second click releases it."""

    angle: float = 0.0
    latched: float | None = None
    _click_was_down: bool = False

    def update(self, dt: float, stick_x: float, stick_y: float, click: bool) -> float:
        target = stick_heading(stick_x, stick_y)
        if click and not self._click_was_down:
            if self.latched is None:
                self.latched = self.angle if target is None else target
            else:
                self.latched = None
        self._click_was_down = click
        if target is None:
            target = 0.0 if self.latched is None else self.latched
        self.angle = approach_angle(self.angle, target, dt)
        return self.angle

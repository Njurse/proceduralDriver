"""Small scalar/vector helpers used across the simulation."""

import math


def clamp(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def lerp(a, b, t):
    return a + (b - a) * t


def inv_lerp(a, b, value):
    if b == a:
        return 0.0
    return (value - a) / (b - a)


def smoothstep(edge0, edge1, x):
    t = clamp(inv_lerp(edge0, edge1, x), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def damp(current, target, speed, dt):
    """Exponential smoothing that is stable for any dt."""
    return target + (current - target) * math.exp(-speed * dt)


def damp_vec(current, target, speed, dt):
    """Smooth two 3-component vectors (or Panda Vec3) in place and return it."""
    alpha = 1.0 - math.exp(-speed * dt)
    current[0] += (target[0] - current[0]) * alpha
    current[1] += (target[1] - current[1]) * alpha
    current[2] += (target[2] - current[2]) * alpha
    return current


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

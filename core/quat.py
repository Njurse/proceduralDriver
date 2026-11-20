"""Small numpy quaternion helpers, (x, y, z, w) order like the C LONGQUATERNION."""

import math

import numpy as np


def from_yaw(heading: float) -> np.ndarray:
    """Rotation about +Y by heading (handling.c InitCarPhysics, minus its tiny pitch)."""
    return np.array([0.0, math.sin(heading / 2.0), 0.0, math.cos(heading / 2.0)])


def to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (columns = body axes in world), handling.c LongQuaternion2Matrix."""
    x, y, z, w = q
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def integrate(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """Advance q by world-frame angular velocity omega (rad/s) over dt.

    GlobalTimeStep adds (omega, 0) * q / 2 per step and renormalises
    approximately (0x1800 - |q|^2 >> 13); here the exact rotation
    exp(omega dt / 2) is applied and q is renormalised exactly (DEVIATION).
    """
    angle = float(np.linalg.norm(omega)) * dt
    if angle < 1e-12:
        return q
    axis = omega / np.linalg.norm(omega)
    s = math.sin(angle / 2.0)
    dq = np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2.0)])
    out = multiply(dq, q)
    return out / np.linalg.norm(out)


def multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])

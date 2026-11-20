"""Fixed-point helpers for REDRIVER2 C interop (Game/dr2math.h).

The handling model itself runs in float64 (see core.physics_helpers); these
helpers exist to read/convert C-scaled data (ONE = 4096 = 1.0, 4096 angle
units = one full turn) and to mirror the C macros where code still needs them.
Python's ``>>`` on ints is an arithmetic shift, which is what the C macros
compile to for signed ints on every target REDRIVER2 builds for.
"""

import math

ONE = 4096
ONE_BITS = 12
FIXED_SHIFT = ONE_BITS  # backwards-compatible alias


def FixHalfRound(x: int, bits: int = ONE_BITS) -> int:
    """dr2math.h FixHalfRound: ((x) + (1 << (bits-1))) >> bits.

    Rounds half toward +infinity for every sign: FixHalfRound(-1) == 0 and
    FixHalfRound(-2048) == 0, FixHalfRound(-2049) == -1.
    """
    return (x + (1 << (bits - 1))) >> bits


def FIXEDH(a: int) -> int:
    """dr2math.h FIXEDH: FixHalfRound(a, ONE_BITS)."""
    return FixHalfRound(a, ONE_BITS)


def FIXED(a: int) -> int:
    """dr2math.h FIXED: a >> ONE_BITS (arithmetic shift, floors negatives)."""
    return a >> ONE_BITS


def ABS(a: int) -> int:
    return a if a >= 0 else -a


def clamp(a, lo, hi):
    """Clamp a into [lo, hi]; works for ints and floats."""
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a


# PsyCross/src/gte/rcossin_tbl.h holds round(sin(i*2pi/4096) * 4096) and the
# matching cos for i in 0..4095; regenerating it with round() reproduces all
# 8192 entries exactly (checked against the C header), so no data is ported.
_RSIN_TBL = tuple(round(math.sin(i * math.tau / ONE) * ONE) for i in range(ONE))
_RCOS_TBL = tuple(round(math.cos(i * math.tau / ONE) * ONE) for i in range(ONE))


def RSIN(a: int) -> int:
    """dr2math.h RSIN: rcossin_tbl[(a & 4095) * 2]."""
    return _RSIN_TBL[a & (ONE - 1)]


def RCOS(a: int) -> int:
    """dr2math.h RCOS: rcossin_tbl[(a & 4095) * 2 + 1]."""
    return _RCOS_TBL[a & (ONE - 1)]


def ratan2(y, x) -> int:
    """atan2 in 4096-per-turn units, C sign convention (LIBGTE.C ratan2).

    Like the C it returns a signed angle in [-2048, 2048] (not masked to
    0..4095) and 0 for (0, 0).  Intentional deviation: computed with float
    atan2 and rounded instead of the 1025-entry ratan_tbl lookup, so results
    can differ from the PSX table by +-1 unit (table quantisation artifact).
    """
    if x == 0 and y == 0:
        return 0
    return round(math.atan2(y, x) * ONE / math.tau)


def angle_to_rad(units: float) -> float:
    """C angle units (4096 per turn) to radians."""
    return units * math.tau / ONE


def rad_to_angle(rad: float) -> float:
    """Radians to C angle units (4096 per turn)."""
    return rad * ONE / math.tau


def vec3_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_length(a):
    return int(math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


def vec3_scale(a, s):
    return (a[0] * s // ONE, a[1] * s // ONE, a[2] * s // ONE)


class Vec3:
    """Minimal integer Vec3 with ONE scaling (4096 = 1.0)."""

    __slots__ = ("x", "y", "z")

    def __init__(self, x=0, y=0, z=0):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __add__(self, o):
        return Vec3(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        return Vec3(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, s):
        if isinstance(s, Vec3):
            return Vec3(self.x * s.x // ONE, self.y * s.y // ONE, self.z * s.z // ONE)
        return Vec3(self.x * s // ONE, self.y * s // ONE, self.z * s // ONE)

    def dot(self, o):
        return self.x * o.x + self.y * o.y + self.z * o.z

    def length(self):
        return int(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def to_float(self):
        return (self.x / ONE, self.y / ONE, self.z / ONE)

    def __repr__(self):
        return f"Vec3({self.x},{self.y},{self.z})"

"""Fixed-point helpers ported from REDRIVER2 (dr2math.h).

Core vehicle state/forces use integer math with ONE=4096 (4096 = 1.0).
Floats are only used at the render/HUD boundary.  Python ints are arbitrary
precision, so arithmetic is exact and traceable to the C source; note that C's
``>> 12`` truncates toward zero for negatives, which differs from Python's
floor division -- ``FIXEDH`` (FixHalfRound) mitigates this.
"""

ONE = 4096
FIXED_SHIFT = 12


def FixHalfRound(x: int, bits: int = FIXED_SHIFT) -> int:
    """Round x to nearest 2**bits, C-style (toward zero on ties-ish)."""
    """Round x to the nearest multiple of 2**bits, ties away-ish (C style)."""
    if x >= 0:
        return (x + (1 << (bits - 1))) >> bits
    return (x - (1 << (bits - 1))) >> bits


def FIXEDH(a: int) -> int:
    """FixHalfRound(a, 12) -- half-round to integer world units."""
    return FixHalfRound(a, FIXED_SHIFT)


def FIXED(a: int) -> int:
    """a >> 12 -- truncate fixed-point to integer (toward zero)."""
    """a >> 12 -- truncate fixed-point to integer."""
    if a >= 0:
        return a >> FIXED_SHIFT
    # C truncates toward zero.
    return -((-a) >> FIXED_SHIFT)


def ABS(a: int) -> int:
    return a if a >= 0 else -a


def clamp(a, lo, hi) -> int:  # works for int/float inputs
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a


# Trig tables: 4096-entry circle.  We approximate with Python math but keep the
# 4096-quantised angle interface so call sites match REDRIVER2.
import math as _math

def _build_trig():
    tbl = []
    for i in range(ONE):
        ang = (i / ONE) * 2.0 * _math.pi
        tbl.append((_math.sin(ang), _math.cos(ang)))  # (sin, cos) per C layout
    return tbl

_TRIG = _build_trig()


def RSIN(a: int) -> int:
    """Sine of angle a (4096 circle units). Returns fixed-point int."""
    idx = a & (ONE - 1)
    return int(_TRIG[idx][0] * ONE)


def RCOS(a: int) -> int:
    """Cosine of angle a (4096 circle units). Returns fixed-point int."""
    idx = a & (ONE - 1)
    return int(_TRIG[idx][1] * ONE)


def ratan2(y, x) -> int:
    """atan2 in 4096 circle units."""
    if x == 0 and y == 0:
        return 0
    ang = _math.atan2(y, x)  # radians
    val = int((ang / (2.0 * _math.pi)) * ONE)
    return val & (ONE - 1)


def vec3_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vec3_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vec3_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vec3_length(a):
    return int(_math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]))


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
        import math as m
        return int(m.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    def to_float(self):
        return (self.x / ONE, self.y / ONE, self.z / ONE)

    def __repr__(self):
        return f"Vec3({self.x},{self.y},{self.z})"


if __name__ == "__main__":
    # sanity
    print("RSIN(0)=", RSIN(0), "RCOS(0)=", RCOS(0))
    print("FIXEDH(4096*1.5)=", FIXEDH(int(4096 * 1.5)))
    print("FIXED(4096*1.5)=", FIXED(int(4096 * 1.5)))
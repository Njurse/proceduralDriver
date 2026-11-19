import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fixedpoint import ONE, FIXED, FIXEDH, RSIN, RCOS, clamp, ABS


def test_one():
    assert ONE == 4096


def test_fixed():
    assert FIXED(4096 * 3) == 3
    assert FIXED(-4096) == -1


def test_fixedh():
    assert FIXEDH(4096 * 2 + 2048) == 3  # 2.5 -> 3 (half round up)
    assert FIXEDH(0) == 0


def test_rsin_rcos():
    assert RSIN(0) == 0
    assert RCOS(0) == 4096
    # sin(90deg) ~ 1
    q = ONE // 4
    assert abs(RSIN(q) - 4096) < 5


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10


def test_abs():
    assert ABS(-5) == 5
    assert ABS(5) == 5
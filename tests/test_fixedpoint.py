import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fixedpoint import (  # noqa: E402
    ABS,
    FIXED,
    FIXEDH,
    ONE,
    RCOS,
    RSIN,
    FixHalfRound,
    clamp,
    ratan2,
)


def test_one():
    assert ONE == 4096


def test_fixed_is_arithmetic_shift():
    # dr2math.h: FIXED(a) = a >> 12 (floors negatives, unlike trunc-to-zero)
    assert FIXED(4096 * 3) == 3
    assert FIXED(-4096) == -1
    assert FIXED(-1) == -1
    assert FIXED(-4097) == -2
    assert FIXED(4095) == 0


def test_fixedh_matches_c_for_negatives():
    # dr2math.h: FixHalfRound(x, 12) = (x + 2048) >> 12
    assert FIXEDH(4096 * 2 + 2048) == 3
    assert FIXEDH(0) == 0
    assert FIXEDH(-1) == 0
    assert FIXEDH(-2048) == 0
    assert FIXEDH(-2049) == -1
    assert FIXEDH(-6144) == -1  # -1.5 rounds toward +inf like the C
    assert FixHalfRound(-1, 13) == 0
    assert FixHalfRound(-4097, 13) == -1


def test_rsin_rcos_match_c_table():
    # Values copied from PsyCross/src/gte/rcossin_tbl.h
    assert (RSIN(0), RCOS(0)) == (0, 0x1000)
    assert (RSIN(1), RCOS(1)) == (0x0006, 0x1000)
    assert (RSIN(11), RCOS(11)) == (0x0045, 0x0FFF)
    assert (RSIN(19), RCOS(19)) == (0x0077, 0x0FFE)
    assert RSIN(1024) == 4096
    assert RCOS(2048) == -4096
    assert RSIN(4095) == -6  # 0xFFFA
    assert RSIN(-1) == RSIN(4095)  # macro masks with & 4095
    assert RSIN(4096 + 11) == RSIN(11)


def test_rsin_rcos_whole_table_checksum():
    # sum((i+1) * entry) over all 4096 sin and cos entries of the C header
    assert sum((i + 1) * RSIN(i) for i in range(4096)) == -10937012224
    assert sum((i + 1) * RCOS(i) for i in range(4096)) == -8388608


def test_ratan2_signed_like_c():
    assert ratan2(0, 0) == 0
    assert ratan2(0, 100) == 0
    assert ratan2(100, 0) == 1024
    assert ratan2(0, -100) == 2048
    assert ratan2(-100, 0) == -1024  # C returns signed, not masked
    assert ratan2(100, 100) == 512
    assert ratan2(-100, -100) == -1536


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(11, 0, 10) == 10
    assert clamp(0.5, 0.0, 1.0) == 0.5


def test_abs():
    assert ABS(-5) == 5
    assert ABS(5) == 5

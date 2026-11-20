"""World scale: one float world unit is one metre.

Driver 2 stores world positions as plain integers ("DR2 units"); fixed-point
ONE = 4096 is a separate thing that scales fractions (matrices, mass,
traction, powerRatio) and the 4096ths-of-a-unit velocity accumulators.  This
module is the single place that turns DR2 data into SI units.

DR2_UNITS_PER_METRE = 150 comes from the Havana (a 1957 Chevrolet Bel Air
class sedan: length 5.08 m, width 1.87 m, wheelbase 2.92 m):
  * OBJ / cPoints length 750 units  -> 148 units/m
  * OBJ width 282 units              -> 151 units/m
  * wheelbase 242 + 199 = 441 units  -> 151 units/m
and is cross-checked by the physics: twistRateY = 224 gives a yaw radius of
gyration of 1.83 m at this scale, and the resting roof height comes out at
about 1.5 m.  DR2 gravity (-7456 per frame) is then 10.9 m/s^2.
"""

import math

DR2_UNITS_PER_METRE = 150.0
ONE = 4096.0  # fixed-point 1.0
C_FRAME_HZ = 30.0  # Driver 2 ran car physics once per 30 Hz frame
ANGLE_UNITS_PER_TURN = 4096.0


def dr2_length(units: float) -> float:
    """DR2 world units -> metres."""
    return units / DR2_UNITS_PER_METRE


def dr2_velocity(lv: float) -> float:
    """C linearVelocity / hd.wheel_speed (4096ths of a unit per frame) -> m/s."""
    return lv / ONE * C_FRAME_HZ / DR2_UNITS_PER_METRE


def to_dr2_velocity(mps: float) -> float:
    """m/s -> C linearVelocity units (inverse of dr2_velocity)."""
    return mps * DR2_UNITS_PER_METRE / C_FRAME_HZ * ONE


def dr2_accel(per_frame: float) -> float:
    """A C linearVelocity change applied once per frame -> m/s^2."""
    return dr2_velocity(per_frame) * C_FRAME_HZ


def dr2_angle(units: float) -> float:
    """C angle units (4096 per turn) -> radians."""
    return units * math.tau / ANGLE_UNITS_PER_TURN


def dr2_inverse_inertia(twist_rate: float) -> float:
    """car_cosmetics twistRate -> inverse inertia per unit mass (1/m^2).

    StepOneCar adds FIXEDH(lever x force) * twistRate to angularVelocity once
    per frame, and angularVelocity / 2^24 is rad per frame (GlobalTimeStep's
    quaternion update), so alpha = (r x a) * twistRate * U^2 / 2^24.
    """
    return twist_rate * DR2_UNITS_PER_METRE**2 / 2.0**24

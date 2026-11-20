"""Driver 2 car forces in float64 SI units (y up, metres, seconds).

Ports of Game/C/wheelforces.c StepOneCar (gravity, speed, cPoint lift),
AddWheelForcesDriver1 (suspension, tyre slide, thrust, drag, wheel_speed)
and ConvertTorqueToAngularAcceleration.  The C applied all of these once per
30 Hz frame as velocity changes; here every term is an acceleration (C value
per frame x 30 Hz) so any fixed tick gives the same motion.  Places that
deviate from the C on purpose are marked DEVIATION.

Coordinates follow the C physics frame: body +Z is forward, +Y up, and a
positive heading / wheel angle turns toward +X (the C's "right").
"""

import math
from dataclasses import dataclass, field

import numpy as np

import config
from core.engine import gear_thrust_scale
from core.physics_helpers import HANDLING_TYPES, get_friction_scales_driver1
from core.units import (
    C_FRAME_HZ,
    DR2_UNITS_PER_METRE,
    ONE,
    dr2_accel,
    dr2_inverse_inertia,
    dr2_length,
    dr2_velocity,
)

F = C_FRAME_HZ
UP = np.array([0.0, 1.0, 0.0])

GRAVITY = dr2_accel(-7456)  # wheelforces.c GRAVITY_FORCE, m/s^2
# StepOneCar spring constants: susForce = 230 * new - 100 * old compression,
# once per frame.  DEVIATION: written as spring 130 + damper 100 on the
# compression rate, so the damping does not depend on the tick rate.
SUS_STIFFNESS = (230 - 100) * F * F / ONE  # m/s^2 per metre of compression
SUS_DAMPING = 100 * F / ONE  # m/s^2 per m/s of compression rate
SUS_REST_OFFSET = dr2_length(14)
SUS_MAX = dr2_length(42)
SUS_GLITCH = dr2_length(800)  # C: a compression this big reads as 12
SUS_GLITCH_VALUE = dr2_length(12)
WHEEL_DRAG = 12 * F / ONE  # 1/s per wheel (C: vel * 12 >> 12 per frame)
# Terminal speed of the stock car: 2 driven wheels x 4915 thrust against
# 4 wheels x 12/4096 drag per frame (~41 m/s).
C_TOP_SPEED = dr2_velocity(2 * 4915 * ONE / (4 * 12))
# DEVIATION: drag along the heading is scaled so full throttle in top gear
# settles at config.CAR_TOP_SPEED_KMH; sideways drag keeps the C value, so
# cornering and slides are unchanged.
FORWARD_WHEEL_DRAG = (WHEEL_DRAG * C_TOP_SPEED * gear_thrust_scale(len(config.GEAR_RATIOS) - 1)
                      / (config.CAR_TOP_SPEED_KMH / 3.6))
FRICTION_DRY = (4096 * 32400 / 32768 + 500) / ONE  # (forcefac*(32400-wet)>>15)+500
FRICTION_GRASS = (2048 * 32400 / 32768 + 500) / ONE
SLIDE_CAP = dr2_velocity(12500)
SLIDE_SATURATE = dr2_velocity(50000)
LOCKED_FLAT_SPEED = dr2_velocity(8000)  # |vx| + |vz| below this: flat normal
WHEEL_SPEED_AIR_FWD = dr2_velocity(1703936 + 0x4000)
WHEEL_SPEED_AIR_REV = dr2_velocity(-1245184 + 0x4000)
ANGULAR_DAMPING = 128 * F / ONE  # 1/s, cl.angularDamping = 128
EXTRA_ANGULAR_DAMPING = F / 8.0  # 1/s, avel / 8 per frame after a lift
# cPoint lift (StepOneCar "do lifting")
LIFT_MAX_DEPTH = dr2_length(800)
LIFT_KILL_VY = dr2_length(120)
LIFT_FRICTION = 67 / ONE
LIFT_FRICTION_LIMIT = tuple(dr2_velocity(v) for v in (1005, 6030, 1005, 5025, 1884, 5025))
# DEVIATION: tyre forces turn the body through a shortened vertical lever.
# The stock player car uses the full contact-patch lever (wheelPos[1]); this
# borrows AddWheelForcesDriver1's own pursuer "more stability" rule, which
# scales that lever by 12/32 (gCopDifficultyLevel 2) or 19/32.  A float port
# and an integer emulation of the stock code (tools/dr2_fixed_emu.py) both
# lift the front wheels on launch and roll over at full lock, while the stock
# game's handling probe speed curve (67/98/130/147 u/frame at 1-4 s) matches
# the emulation only with the front wheels kept on the road.  The y entry
# only multiplies the horizontal force components, exactly like the C rule.
# 12/32 (the hard-cop value): with 19/32 the car still rolled over at full
# lock from 50 m/s, a speed the 200+ km/h gearing now reaches.
TYRE_LEVER_SCALE = np.array([1.0, 12.0 / 32.0, 1.0])
SURFACE_ROAD = 0
SURFACE_GRASS = 1


def lateral_stiffness(speed: float) -> float:
    """AddWheelForcesDriver1 oldSpeed curve, as a fraction (C value / ONE).

    DEVIATION: speed is the true horizontal speed, not the C's octagonal
    max + min / 2 estimate.
    """
    s = 1.5 * speed * DR2_UNITS_PER_METRE / F  # hd.speed * 3 >> 1
    s = 3696.0 - 72.0 * s if s < 32.0 else 1424.0 - s
    return min(s, 3300.0) / ONE


@dataclass
class CarGeometry:
    """car_cosmetics in metres."""

    wheel_disp: np.ndarray  # (4, 3); 0,2 front, 1,3 rear
    c_points: np.ndarray  # (12, 3); 0..3 are the bottom corners
    inv_inertia_y: float  # 1/m^2 about the body's non-forward axes
    inv_inertia_z: float  # about the forward axis
    wheel_radius: float
    power_ratio: int
    traction: int
    # where the body model's origin sits in the physics body frame: cars.c
    # DrawCar draws the model at where + R * cog, and the frames differ by a
    # negation (see main.py), so this is -cog after FixCarCos.
    model_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @classmethod
    def from_cosmetics(cls, cos) -> "CarGeometry":
        wheel_disp = np.array(cos.wheel_disp, dtype=float)
        c_points = np.array(cos.c_points, dtype=float)
        # cosmetic.c FixCarCos: centre the wheelbase on the body origin by
        # moving wheels and cPoints along z (C int division truncates to 0).
        dz = float(int(-(wheel_disp[0, 2] + wheel_disp[1, 2] - 14) / 2))
        wheel_disp[:, 2] += dz
        c_points[:, 2] += dz
        cog = np.array(cos.cog, dtype=float)
        cog[2] -= dz  # handling.c UpdateCarPoints moves the cog the other way
        return cls(
            wheel_disp=wheel_disp / DR2_UNITS_PER_METRE,
            c_points=c_points / DR2_UNITS_PER_METRE,
            inv_inertia_y=dr2_inverse_inertia(cos.twist_rate_y),
            inv_inertia_z=dr2_inverse_inertia(cos.twist_rate_z),
            wheel_radius=dr2_length(cos.wheel_size),
            power_ratio=cos.power_ratio,
            traction=cos.traction,
            model_offset=-cog / DR2_UNITS_PER_METRE,
        )


def flat_ground(x: float, z: float):
    """Ground query: (height, unit normal, surface)."""
    return 0.0, UP, SURFACE_ROAD


@dataclass
class WheelOut:
    compression: float = 0.0
    locked: bool = False
    on_grass: bool = False
    slide: float = 0.0  # |sideways speed| at the contact patch, m/s
    world_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Forces:
    acc: np.ndarray
    alpha: np.ndarray
    thrust: float
    wheelspin: bool
    reset_auto_brake: bool
    wheel_speed: float  # hd.wheel_speed, m/s along the heading
    wheels: list


def heading_of(rot: np.ndarray) -> float:
    """hd.direction = ratan2(m[0][2], m[2][2])."""
    return math.atan2(rot[0, 2], rot[2, 2])


def _cpoint_lift(pos, vel, avel, rot, geo, ground):
    """StepOneCar 'calculate lifting factor' + 'do lifting'.

    Returns (acc, torque, position correction, kill vertical velocity, lifted).
    """
    count = 4 if rot[1, 1] > 0.5 else 12
    lift = 0.0
    best = None
    for idx in range(count - 1, -1, -1):
        lever = rot @ geo.c_points[idx]
        point = pos + lever
        height, normal, _ = ground(point[0], point[2])
        depth = height - point[1]
        if 0.0 < depth < LIFT_MAX_DEPTH and depth * normal[1] > lift:
            lift = depth * normal[1]
            best = (lever, normal, 3 if idx > 3 else 0)
    if best is None:
        return None
    lever, normal, fri = best
    point_vel = vel + np.cross(avel, lever)
    r_n = float(lever @ normal)
    disp = 1.0 + (float(lever @ lever) - r_n * r_n) * geo.inv_inertia_y
    impulse = -0.5 * float(point_vel @ normal) / disp  # m/s per frame
    reaction = impulse * normal
    for c in range(3):
        limit = LIFT_FRICTION_LIMIT[fri + 2 - c]
        reaction[c] -= min(max(LIFT_FRICTION * point_vel[c], -limit), limit)
    acc = reaction * F
    return acc, np.cross(lever, acc), lift * normal, lift > LIFT_KILL_VY


def car_forces(body, pad, geo: CarGeometry, dt: float, ground=flat_ground,
               hnd_type: int = 0, thrust_scale: float = 1.0) -> Forces:
    """One StepOneCar evaluation.  ``body`` has pos, vel, avel, rot,
    compression (4,); ``pad`` is the core.physics_helpers.PadState.

    May move ``body.pos`` (cPoint lift) and zero ``body.vel[1]`` like the C.
    """
    rot = body.rot
    acc = np.array([0.0, GRAVITY, 0.0])
    torque = np.zeros(3)
    speed = math.hypot(body.vel[0], body.vel[2])
    extra_damping = False

    lifted = _cpoint_lift(body.pos, body.vel, body.avel, rot, geo, ground)
    if lifted is not None:
        l_acc, l_torque, correction, kill_vy = lifted
        acc += l_acc
        torque += l_torque
        body.pos = body.pos + correction
        if kill_vy:
            body.vel[1] = 0.0
        extra_damping = True

    heading = heading_of(rot)
    fwd_flat = np.array([math.sin(heading), 0.0, math.cos(heading)])
    stiff = lateral_stiffness(speed)

    wheels = [WheelOut() for _ in range(4)]
    v_fwd = float(fwd_flat @ body.vel)
    drag_x = body.vel[0] * WHEEL_DRAG + (FORWARD_WHEEL_DRAG - WHEEL_DRAG) * v_fwd * fwd_flat[0]
    drag_z = body.vel[2] * WHEEL_DRAG + (FORWARD_WHEEL_DRAG - WHEEL_DRAG) * v_fwd * fwd_flat[2]
    grass0 = ground(*(body.pos + rot @ geo.wheel_disp[0])[[0, 2]])[2] == SURFACE_GRASS
    fr = get_friction_scales_driver1(
        thrust=pad.thrust, wheelspin=pad.wheelspin, handbrake=pad.handbrake,
        auto_brake=pad.auto_brake, wheel_speed=body.wheel_speed,
        front_on_grass=grass0, hnd_type=hnd_type,
        power_ratio=geo.power_ratio, traction=geo.traction)
    thrust_acc = dr2_accel(fr.thrust * ONE) * thrust_scale  # DEVIATION: per-gear scale
    reset_auto_brake = False
    steer_dir = heading + pad.wheel_angle
    sx, sz = math.sin(steer_dir), math.cos(steer_dir)

    # friction loss when the car is on its side (where.m[1][1] < 2048)
    up_y = rot[1, 1]
    tilt = 1.0
    if up_y < 0.5:
        a = 1.0 - up_y
        tilt = 1.0 - a * a if a <= 1.0 else 0.0

    for i in range(4):
        w = wheels[i]
        w.locked = fr.locked[i]
        lever = rot @ geo.wheel_disp[i]
        wpos = body.pos + lever
        w.world_pos = wpos
        height, normal, surface = ground(wpos[0], wpos[2])
        w.on_grass = surface == SURFACE_GRASS
        friction = FRICTION_GRASS if w.on_grass else FRICTION_DRY

        comp = (height - wpos[1]) * normal[1] + SUS_REST_OFFSET
        comp = max(comp, 0.0)
        if comp > SUS_GLITCH:
            comp = SUS_GLITCH_VALUE
        comp = min(comp, SUS_MAX)
        old = body.compression[i]
        w.compression = comp
        if comp == 0.0 and old == 0.0:
            continue

        pv = body.vel + np.cross(body.avel, lever)
        sus = SUS_STIFFNESS * comp + SUS_DAMPING * (comp - old) / dt
        n = normal
        wdir = heading + (0.0 if i & 1 else pad.wheel_angle)
        w.slide = abs(pv[0] * math.cos(wdir) - pv[2] * math.sin(wdir))
        if w.locked:
            # a locked tyre fights the whole velocity, not just the lateral part
            mag = math.hypot(pv[0], pv[2])
            lat = (pv[0] / mag, pv[2] / mag) if mag > 0.0 else (0.0, 1.0)
            if abs(pv[0]) + abs(pv[2]) < LOCKED_FLAT_SPEED:
                n = UP
        else:
            lat = (math.cos(wdir), -math.sin(wdir))
            fwd_w = (math.sin(wdir), math.cos(wdir))

        slide = pv[0] * lat[0] + pv[2] * lat[1]
        if abs(slide) > SLIDE_SATURATE:
            slide = math.copysign(SLIDE_CAP, slide)
        else:
            slide = min(max(stiff * slide, -SLIDE_CAP), SLIDE_CAP)

        fx = fz = 0.0
        if i & 1:  # rear
            if not w.locked:
                side = fr.rear * slide
                if HANDLING_TYPES[hnd_type].auto_brake_on and side * pad.wheel_angle > 0:
                    reset_auto_brake = True
                fx += fwd_w[0] * thrust_acc
                fz += fwd_w[1] * thrust_acc
            else:
                side = fr.front * slide / 4.0
        else:  # front
            side = fr.front * slide
            if w.locked:
                side *= 0.75
                fac = -2.0 * side * (lat[0] * sz - lat[1] * sx)
                fx += fac * sz * F
                fz += -fac * sx * F

        fx += sus * n[0] - side * lat[0] * F - drag_x
        fz += sus * n[2] - side * lat[1] * F - drag_z

        coef = friction * tilt
        if n[1] < 0.8:
            coef *= n[1] * 1.25
        force = np.array([fx * coef, sus * n[1] - body.vel[1] * WHEEL_DRAG, fz * coef])
        acc += force
        torque += np.cross(lever * TYRE_LEVER_SCALE, force)

    for i in range(4):
        body.compression[i] = wheels[i].compression

    if body.compression[1] == 0.0 and body.compression[3] == 0.0:
        if fr.thrust > 0:
            wheel_speed = WHEEL_SPEED_AIR_FWD
        elif fr.thrust < 0:
            wheel_speed = WHEEL_SPEED_AIR_REV
        else:
            wheel_speed = 0.0
    else:
        wheel_speed = float(fwd_flat @ body.vel)

    alpha = _torque_to_angular_acceleration(torque, rot, body.avel, geo, extra_damping)
    return Forces(acc, alpha, fr.thrust, fr.wheelspin, reset_auto_brake, wheel_speed, wheels)


def _torque_to_angular_acceleration(torque, rot, avel, geo, extra_damping):
    """ConvertTorqueToAngularAcceleration: twistRateY off-axis, twistRateZ
    about the body's forward axis, plus angular damping."""
    fwd = rot[:, 2]
    along = fwd * float(fwd @ torque)
    alpha = geo.inv_inertia_y * (torque - along) + geo.inv_inertia_z * along
    alpha -= avel * ANGULAR_DAMPING
    if extra_damping:
        alpha -= avel * EXTRA_ANGULAR_DAMPING
    return alpha

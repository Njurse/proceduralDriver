"""Full 3D vehicle dynamics for the driving simulator.

Coordinate system: y-up (matching the REDRIVER2 import).  Panda3D rendering
maps physics (x, y_up, z_forward) -> Panda3D (x, y_forward=z_forward, z_up=y_up).

Model: per-wheel spring/damper suspension, longitudinal drive/brake forces
(RWD rear axle), lateral tire friction with a friction circle, diagonal inertia
tensor in body frame.  Integration is semi-implicit Euler; call sites should
use small substeps.  Behavior is *not* bit-identical to the 2D port — this is
the 3D successor.
"""

import math

try:
    import config
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config
from core.car_profile import load_cosmetics

G = 9.8

# Car dimensions in metres (derived from colBox [143, 89, 379] game units).
CAR_W = 143 * 0.01 * 2
CAR_H = 89 * 0.01 * 2
CAR_L = 379 * 0.01 * 2
MASS = 1500.0
# Diagonal inertia in the body frame (x=right axis, y=up, z=forward).
I_PITCH = MASS * (CAR_H * CAR_H + CAR_L * CAR_L) / 12.0   # about x (right)
I_YAW = MASS * (CAR_W * CAR_W + CAR_L * CAR_L) / 12.0     # about y (up)
I_ROLL = MASS * (CAR_W * CAR_W + CAR_H * CAR_H) / 12.0    # about z (forward)

SUSP_REST = 0.55        # free spring length (m)
SUSP_K = (MASS * G / 4.0) / 0.15   # stiffness so ~0.15 m sag under weight
SUSP_DAMP = 2.0 * math.sqrt(SUSP_K * (MASS / 4.0)) * 0.7
SUSP_MIN = 0.0

MAX_DRIVE = MASS * 8.0   # N (RWD total)
MAX_BRAKE = MASS * 12.0  # N (all wheels)
MU_CONCRETE = 1.05
MU_GRASS = 0.6
LAT_GAIN = 0.18          # tanh lateral slip gain


def _mat_mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )


def _rot_mat(yaw, pitch, roll):
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    Ry = ((cy, 0.0, sy), (0.0, 1.0, 0.0), (-sy, 0.0, cy))
    Rx = ((1.0, 0.0, 0.0), (0.0, cp, -sp), (0.0, sp, cp))
    Rz = ((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0))
    return _mat_mul(Ry, _mat_mul(Rx, Rz))


def _mat_vec(m, v):
    return (m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2])


def _mat_t_vec(m, v):
    return (m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
            m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
            m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _wheel_offsets(cos):
    offs = []
    for disp in cos.wheel_disp:
        x, y, z = disp[0] * 0.01, disp[1] * 0.01, disp[2] * 0.01
        # game units: y negative = below body origin; keep sign (y-up here).
        offs.append((x, y, z))
    return offs


class Input:
    def __init__(self):
        self.steer = 0.0
        self.throttle = 0.0
        self.brake = 0.0
        self.handbrake = False
        self.fast_steer = False


class WheelState:
    def __init__(self):
        self.world_pos = (0.0, 0.0, 0.0)
        self.compression = 0.0
        self.on_ground = False
        self.drive_force = 0.0
        self.lateral_force = 0.0
        self.slip_lateral = 0.0
        self.slip_longitudinal = 0.0
        self.wheel_speed = 0.0


class Vehicle3D:
    """3D vehicle: y-up.  Yaw is heading, pitch about local x, roll about z."""

    def __init__(self, cosmetics=None):
        self.cos = cosmetics or load_cosmetics()
        self.wheels = [WheelState() for _ in range(4)]
        self.wheel_offsets = _wheel_offsets(self.cos)
        self.reset(0.0, 0.0, 0.0)

    def reset(self, x=0.0, z=0.0, yaw=0.0):
        # start at suspension equilibrium (attach height ~0.4 m above ground,
        # wheels ~1.16 m below the body origin)
        self.pos = [x, 1.56, z]
        self.vel = [0.0, 0.0, 0.0]
        self.yaw = yaw
        self.pitch = 0.0
        self.roll = 0.0
        # angular velocity in body frame: (pitch_rate[x], yaw_rate[y], roll_rate[z])
        self.ang_vel = [0.0, 0.0, 0.0]
        self.wheel_angle = 0

    # convenient float views ---------------------------------------------------
    @property
    def speed(self):
        return math.sqrt(self.vel[0] ** 2 + self.vel[1] ** 2 + self.vel[2] ** 2)

    @property
    def forward_speed(self):
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        return self.vel[0] * sy + self.vel[2] * cy

    def process_input(self, inp) -> None:
        self.steer = float(getattr(inp, "steer", 0.0))
        fast = bool(getattr(inp, "fast_steer", False))
        self.throttle = float(getattr(inp, "throttle", 0.0))
        self.brake = float(getattr(inp, "brake", 0.0))
        self.handbrake = bool(getattr(inp, "handbrake", False))
        # steering: reuse digital increments from the 2D model
        inc = 64 if fast else 32
        target = int(self.steer * 511)
        if self.wheel_angle < target:
            self.wheel_angle = min(target, self.wheel_angle + inc)
        elif self.wheel_angle > target:
            self.wheel_angle = max(target, self.wheel_angle - inc)
        if abs(self.steer) < 0.05:
            if self.wheel_angle > 0:
                self.wheel_angle = max(0, self.wheel_angle - 16)
            elif self.wheel_angle < 0:
                self.wheel_angle = min(0, self.wheel_angle + 16)

    def surface_at(self, x, z):
        # Flat asphalt test plane by default (quad: height, normal, mu, grass).
        return (0.0, (0.0, 1.0, 0.0), MU_CONCRETE, False)

    def step(self, dt, substeps=4):
        sub_dt = dt / substeps
        for _ in range(substeps):
            self._integrate(sub_dt)

    def _integrate(self, dt):
        R = _rot_mat(self.yaw, self.pitch, self.roll)
        up = _mat_vec(R, (0.0, 1.0, 0.0))
        fwd = _mat_vec(R, (0.0, 0.0, 1.0))
        right = _mat_vec(R, (1.0, 0.0, 0.0))

        steer_norm = self.wheel_angle / 511.0 if abs(self.wheel_angle) > 0 else 0.0
        front_steer = steer_norm * 0.5  # rad
        drive_total = self.throttle * MAX_DRIVE
        brake_total = self.brake * MAX_BRAKE

        sum_f = [0.0, -MASS * G, 0.0]
        sum_t = [0.0, 0.0, 0.0]
        drive_f = [0.0, 0.0, 0.0]  # longitudinal force applied at the CG (no torque)
        any_contact = False
        fx_s, fz_s = math.cos(front_steer), math.sin(front_steer)
        # rotate the forward/right basis about the up axis by the steer angle
        fwd_r = (fwd[0] * fx_s + right[0] * fz_s,
                 fwd[1] * fx_s + right[1] * fz_s,
                 fwd[2] * fx_s + right[2] * fz_s)
        right_r = (right[0] * fx_s - fwd[0] * fz_s,
                   right[1] * fx_s - fwd[1] * fz_s,
                   right[2] * fx_s - fwd[2] * fz_s)

        for i, o in enumerate(self.wheel_offsets):
            front = (o[2] > 0.0)
            r = _mat_vec(R, o)
            attach = (self.pos[0] + r[0], self.pos[1] + r[1], self.pos[2] + r[2])
            g_h, g_n, mu, grass = self.surface_at(attach[0], attach[2])
            rw_fwd, rw_right = (fwd_r, right_r) if front else (fwd, right)
            # spring (car-up)
            compression = max(SUSP_MIN, SUSP_REST - (attach[1] - g_h))
            v_attach = (self.vel[0] + self._rot_vel(o, R)[0],
                        self.vel[1] + self._rot_vel(o, R)[1],
                        self.vel[2] + self._rot_vel(o, R)[2])
            v_susp = v_attach[0] * up[0] + v_attach[1] * up[1] + v_attach[2] * up[2]
            f_spring = max(0.0, SUSP_K * compression - SUSP_DAMP * v_susp)

            contact = f_spring > 1.0
            wheel = self.wheels[i]
            wheel.on_ground = contact
            wheel.compression = compression
            wheel.world_pos = attach
            if not contact:
                wheel.drive_force = 0.0
                wheel.lateral_force = 0.0
                wheel.slip_lateral = 0.0
                wheel.slip_longitudinal = 0.0
                wheel.wheel_speed = 0.0
                continue
            any_contact = True
            mu_eff = MU_GRASS if grass else mu

            # longitudinal (drive RWD / brake all)
            if front:
                f_long = -brake_total / 4.0
            else:
                f_long = drive_total / 2.0 - brake_total / 4.0
            f_long = max(-mu_eff * f_spring, min(mu_eff * f_spring, f_long))

            # lateral tire friction opposes lateral slip
            v_lat = (v_attach[0] * rw_right[0] + v_attach[1] * rw_right[1]
                     + v_attach[2] * rw_right[2])
            f_lat = -mu_eff * f_spring * math.tanh(v_lat * LAT_GAIN)
            if self.handbrake and not front:
                f_lat *= 0.45

            wheel.drive_force = f_long
            wheel.lateral_force = f_lat
            wheel.slip_lateral = abs(v_lat) * 0.05
            wheel.slip_longitudinal = abs(f_long) / max(1.0, f_spring) * 0.2
            wheel.wheel_speed = self.forward_speed / 0.35  # ~0.35 m rolling radius

            f_w = (up[0] * f_spring + rw_right[0] * f_lat,
                   up[1] * f_spring + rw_right[1] * f_lat,
                   up[2] * f_spring + rw_right[2] * f_lat)
            sum_f[0] += f_w[0]; sum_f[1] += f_w[1]; sum_f[2] += f_w[2]
            t = _cross(r, f_w)
            sum_t[0] += t[0]; sum_t[1] += t[1]; sum_t[2] += t[2]
            # longitudinal force acts at the CG (anti-squat/anti-dive simplification)
            drive_f[0] += rw_fwd[0] * f_long
            drive_f[1] += rw_fwd[1] * f_long
            drive_f[2] += rw_fwd[2] * f_long

        sum_f[0] += drive_f[0]; sum_f[1] += drive_f[1]; sum_f[2] += drive_f[2]
        # linear integration
        self.vel[0] += (sum_f[0] / MASS) * dt
        self.vel[1] += (sum_f[1] / MASS) * dt
        self.vel[2] += (sum_f[2] / MASS) * dt
        # air drag (quadratic, applied opposite to velocity)
        speed = self.speed
        if speed > 0.1:
            decel = min(0.0004 * speed * speed, speed / dt)  # m/s^2
            factor = max(0.0, decel * dt / speed)
            self.vel = [c * (1.0 - factor) for c in self.vel]

        self.pos[0] += self.vel[0] * dt
        self.pos[1] += self.vel[1] * dt
        self.pos[2] += self.vel[2] * dt
        if self.pos[1] < 0.25:
            self.pos[1] = 0.25
            if self.vel[1] < 0.0:
                self.vel[1] *= -0.2

        # angular motion — body frame axes: x = right (pitch axis),
        # y = up (yaw axis), z = forward (roll axis)
        tau_body = _mat_t_vec(R, sum_t)
        self.ang_vel[0] *= (1.0 - min(1.0, 1.2 * dt))
        self.ang_vel[1] *= (1.0 - min(1.0, 0.6 * dt))
        self.ang_vel[2] *= (1.0 - min(1.0, 1.2 * dt))
        self.ang_vel[0] += (tau_body[0] / I_PITCH) * dt
        self.ang_vel[1] += (tau_body[1] / I_YAW) * dt
        self.ang_vel[2] += (tau_body[2] / I_ROLL) * dt
        # yaw assist: map steering to a speed-scaled yaw rate so cornering is
        # responsive even though the tire model saturates its lateral force.
        # The lateral friction still provides slide/handbrake behaviour.
        steer_norm = (self.wheel_angle / 511.0) if abs(self.wheel_angle) > 0 else 0.0
        target_yaw = steer_norm * min(1.0, self.speed / 14.0) * 1.6
        self.ang_vel[1] += (target_yaw - self.ang_vel[1]) * min(1.0, 5.0 * dt)

        self.ang_vel = [max(-4.0, min(4.0, w)) for w in self.ang_vel]
        self.pitch += self.ang_vel[0] * dt
        self.yaw += self.ang_vel[1] * dt
        self.roll += self.ang_vel[2] * dt
        self.pitch = max(-0.6, min(0.6, self.pitch))
        self.roll = max(-0.6, min(0.6, self.roll))

    def _rot_vel(self, o_local, R):
        # angular velocity at the wheel attachment (body ang_vel cross local offset)
        wx, wy, wz = self.ang_vel
        v_local = (wy * o_local[2] - wz * o_local[1],
                   wz * o_local[0] - wx * o_local[2],
                   wx * o_local[1] - wy * o_local[0])
        # rotate body-frame offset velocity into world frame
        return _mat_vec(R, v_local)


if __name__ == "__main__":
    v = Vehicle3D()
    v.reset(0.0, 0.0, 0.0)
    inp = Input()
    inp.throttle = 1.0
    v.process_input(inp)
    for i in range(180):
        v.step(1.0 / 60.0, substeps=4)
    print("y", round(v.pos[1], 3), "speed", round(v.speed, 2),
          "yaw", round(v.yaw, 3), "pitch", round(v.pitch, 3))
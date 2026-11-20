"""Integer re-implementation of the stock Driver 2 player car on a flat road.

A reference to check core.redriver2_physics against: the C's own fixed-point
maths (FIXEDH, >> shifts, C division, RSIN/RCOS tables, ratan2), one 30 Hz
frame at a time, for hndType 0 on a flat dry road with no other cars.

Ported from Game/C: handling.c (ProcessCarPad digital steering and pedals,
InitCarPhysics, GlobalTimeStep, RebuildCarMatrix, LongQuaternion2Matrix),
wheelforces.c (StepOneCar, AddWheelForcesDriver1, GetFrictionScalesDriver1,
ConvertTorqueToAngularAcceleration), cosmetic.c (FixCarCos) and gamesnd.c
(ControlCarRevs, GetEngineRevs).  Left out: surfaces other than flat road,
collisions, damage/bend hooks, JERICHO hooks (stock = no handler), analogue
steering, wetness, 32-bit overflow wrap-around.

Usage:  python tools/dr2_fixed_emu.py   (prints a launch and a full-lock run)
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fixedpoint import FIXEDH, RCOS, RSIN, FixHalfRound, ratan2  # noqa: E402

GRAVITY_FORCE = -7456
SPRING_RATE = 230
SPRING_DAMPING = 100
ANGULAR_DAMPING = 128
FRICTION_LIMIT = (1005 * 4096, 6030 * 4096, 1005 * 4096, 5025 * 4096, 1884 * 4096,
                  5025 * 4096)
GEARD = ((0, 0, 163, 144, 135), (86, 133, 260, 90, 85), (186, 233, 360, 60, 57),
         (286, 326, 9999, 48, 45))
PAD_LEFT, PAD_RIGHT, PAD_ACCEL, PAD_BRAKE, PAD_HANDBRAKE = 1, 2, 4, 8, 16


def cdiv(a: int, b: int) -> int:
    """C integer division (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def rtv(m, v, t):
    """gte_rtv0tr: (R * v) >> 12 + t."""
    return [((m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2]) >> 12) + t[i]
            for i in range(3)]


class Car:
    def __init__(self, cos, direction: int = 0):
        wd = [list(w) for w in cos.wheel_disp]
        cp = [list(c) for c in cos.c_points]
        dz = cdiv(-(wd[0][2] + wd[1][2] - 14), 2)  # FixCarCos
        for p in wd + cp:
            p[2] += dz
        self.wd, self.cp = wd, cp
        self.twist_y, self.twist_z = cos.twist_rate_y, cos.twist_rate_z
        self.power = cos.power_ratio
        # InitCarPhysics
        d = wd[0][2] + wd[1][2]
        ty, odz = cdiv(d, 5), cdiv(d, 32)
        sn, cs = RSIN(direction // 2), RCOS(direction // 2)
        self.orient = [FIXEDH(-cs * ty), sn, FIXEDH(sn * ty), cs]
        self.fpos = [0, (-wd[0][1] + 14 - 14) << 4, 0]  # wheels touching flat ground
        self.vel = [0, 0, 0]
        self.avel = [0, 0, 0]
        self.comp = [14 - odz, odz + 14, 14 - odz, odz + 14]
        self.locked = [0, 0, 0, 0]
        self.m = [[4096, 0, 0], [0, 4096, 0], [0, 0, 4096]]
        self.t = [0, 0, 0]
        self.rebuild(self.fpos, self.orient)
        self.direction = direction
        self.thrust = self.wheel_angle = self.wheel_speed = 0
        self.auto_brake = self.handbrake = self.wheelspin = 0
        self.revs = self.gear = self.changing_gear = 0
        self.speed = 0

    # handling.c RebuildCarMatrix + LongQuaternion2Matrix
    def rebuild(self, fpos, q):
        self.t = [fpos[0] >> 4, fpos[1] >> 4, fpos[2] >> 4]
        qx, qy, qz, qw = q
        osm = qx * qx + qy * qy + qz * qz + qw * qw
        sm = 4096
        if osm < 1024:
            q[0] = q[1] = q[2] = 0
        else:
            sm = 0x1800 - (osm >> 13)
            q[0], q[1], q[2] = FIXEDH(sm * qx), FIXEDH(sm * qy), FIXEDH(sm * qz)
            sm = FIXEDH(sm * qw)
        q[3] = sm
        qx, qy, qz, qw = q
        h = FixHalfRound
        yy, zz, xx = h(qy * qy, 11), h(qz * qz, 11), h(qx * qx, 11)
        zw, xy, xz = h(qz * qw, 11), h(qx * qy, 11), h(qx * qz, 11)
        yw, xw, yz = h(qy * qw, 11), h(qx * qw, 11), h(qy * qz, 11)
        self.m = [[4096 - (yy + zz), xy - zw, xz + yw],
                  [xy + zw, 4096 - (xx + zz), yz - xw],
                  [xz - yw, yz + xw, 4096 - (xx + yy)]]

    # handling.c ProcessCarPad (digital, player, stock pedals)
    def process_pad(self, pad):
        if self.auto_brake > 90:
            self.auto_brake = 90
        if pad & PAD_HANDBRAKE:
            self.handbrake = 1
        else:
            self.handbrake = 0
            self.wheelspin = 0
        if pad & PAD_RIGHT:
            self.wheel_angle = min(self.wheel_angle + 32, 352)
        if pad & PAD_LEFT:
            self.wheel_angle = max(self.wheel_angle - 32, -352)
        if pad & (PAD_LEFT | PAD_RIGHT):
            self.auto_brake += 1
        else:
            self.auto_brake = 0
            if self.wheel_angle < -64:
                self.wheel_angle += 64
            elif self.wheel_angle < 65:
                self.wheel_angle = 0
            else:
                self.wheel_angle -= 64
        self.thrust = 0
        if pad & PAD_BRAKE:
            rws = FIXEDH(cdiv(self.wheel_speed * 1500, 1024))
            rws = -5000 if -rws < 23 else ((rws + 278) * -4778) >> 8
            self.thrust = FIXEDH(self.power * rws)
        elif pad & PAD_ACCEL:
            self.thrust = FIXEDH(self.power * 4915)
            if self.changing_gear:
                self.thrust = 1

    # wheelforces.c GetFrictionScalesDriver1 (hndType 0, road)
    def friction_scales(self):
        front = 1453 if self.thrust < 0 else (937 if self.thrust < 1 else 820)
        ab = self.auto_brake
        if self.wheelspin == 0 and ab > 0 and self.wheel_speed > 0:
            q = ab << 1
            if ab > 13:
                ab, q = 13, 26
            front += (q + ab) * 15
        lock = 1 if (self.thrust < 0 and self.wheel_speed > 41943) else 0
        self.locked = [lock] * 4
        if self.handbrake == 0:
            if self.wheelspin:
                front += 600
        else:
            if self.thrust > -1:
                self.thrust = 0
            front += -375 if self.wheel_speed < 1 else 656
            self.locked[1] = self.locked[3] = 1
            self.wheelspin = 0
        if self.wheel_speed < 0 and self.thrust > -1 and self.handbrake == 0:
            front -= 400
        rear = 0x780 - front
        if self.wheelspin:
            self.thrust = FIXEDH(self.power * 5000)
        if self.thrust < 0 and self.wheel_speed > 41943:
            front, rear = cdiv(front * 10, 8), cdiv(rear * 10, 8)
        else:
            front = cdiv(front * 36 - front, 32)
        return cdiv(front * 32, 32), cdiv(rear * 32, 32)

    def step_one_car(self):
        vel, avel = list(self.vel), list(self.avel)
        acc = [0, GRAVITY_FORCE, 0]
        self.aacc = [0, 0, 0]
        a, b = abs(FIXEDH(vel[0])), abs(FIXEDH(vel[2]))
        self.speed = b + a // 2 if a < b else a + b // 2
        m, t = self.m, self.t
        extra = 0
        count = 4 if m[1][1] > 2048 else 12
        lift = 0
        best = None
        for idx in range(count - 1, -1, -1):
            pp = rtv(m, self.cp[idx], t)
            depth = 0 - pp[1]
            if 1 <= depth <= 799 and lift < depth:
                lift = depth
                best = ([pp[i] - t[i] for i in range(3)], 3 if idx > 3 else 0)
        if lift:
            dl, fri = best
            pv = [FIXEDH(avel[1] * dl[2] - avel[2] * dl[1]) + vel[0],
                  FIXEDH(avel[2] * dl[0] - avel[0] * dl[2]) + vel[1],
                  FIXEDH(avel[0] * dl[1] - avel[1] * dl[0]) + vel[2]]
            ldn = FIXEDH(dl[1] * 4096)
            dsq = FIXEDH((dl[0] ** 2 + dl[1] ** 2 + dl[2] ** 2 - ldn * ldn) * self.twist_y) + 4096
            strike = (pv[1] >> 6) * (4096 >> 6)
            impulse = cdiv(strike, dsq) * -2048
            n = (0, 4096, 0)
            reaction = [0, 0, 0]
            for c in (2, 1, 0):
                limit = FRICTION_LIMIT[fri + 2 - c]
                loss = pv[c] * 67
                if loss <= limit:
                    limit = -limit if loss < -limit else loss
                reaction[c] = FIXEDH(impulse * n[c] - limit)
            for i in range(3):
                acc[i] += reaction[i]
            self.aacc[0] += FIXEDH(dl[1] * reaction[2] - dl[2] * reaction[1])
            self.aacc[1] += FIXEDH(dl[2] * reaction[0] - dl[0] * reaction[2])
            self.aacc[2] += FIXEDH(dl[0] * reaction[1] - dl[1] * reaction[0])
            self.t[1] += lift
            self.fpos[1] += lift * 16
            extra = 1
            if lift > 120:
                self.vel[1] = 0
        self.wheel_forces(vel, avel, acc)
        self.convert_torque(avel, extra)
        self.acc = acc

    def wheel_forces(self, vel, avel, acc):
        old_speed = self.speed * 3 >> 1
        old_speed = old_speed * -72 + 3696 if old_speed < 32 else 1424 - old_speed
        cdx, cdz = RSIN(self.direction), RCOS(self.direction)
        sdir = self.direction + self.wheel_angle
        sdx, sdz = RSIN(sdir), RCOS(sdir)
        front_fs, rear_fs = self.friction_scales()
        old_speed = min(old_speed, 3300)
        m, t = self.m, self.t
        for i in (3, 2, 1, 0):
            wp = rtv(m, self.wd[i], t)
            fc = (4096 * 32400 >> 15) + 500
            old = self.comp[i]
            new = FIXEDH((0 - wp[1]) * 4096) + 14
            new = max(new, 0)
            if new > 800:
                new = 12
            new = min(new, 42)
            if new == 0 and old == 0:
                self.comp[i] = 0
                continue
            lp = [wp[k] - t[k] for k in range(3)]
            pv0 = FIXEDH(avel[1] * lp[2] - avel[2] * lp[1]) + vel[0]
            pv2 = FIXEDH(avel[0] * lp[1] - avel[1] * lp[0]) + vel[2]
            sus = new * SPRING_RATE - old * SPRING_DAMPING
            n = (0, 4096, 0)
            if self.locked[i]:
                d = ratan2(pv0 >> 6, pv2 >> 6)
                lfx, lfz = RSIN(d), RCOS(d)
            else:
                wdir = self.direction + (0 if i & 1 else self.wheel_angle)
                lfz, lfx = -RSIN(wdir), RCOS(wdir)
            slide = cdiv(pv0, 64) * cdiv(lfx, 64) + cdiv(pv2, 64) * cdiv(lfz, 64)
            if slide > 50000:
                slide = 12500
            elif slide < -50000:
                slide = -12500
            else:
                slide = max(min(FIXEDH(old_speed * slide), 12500), -12500)
            f = [0, 0, 0]
            if i & 1:
                if not self.locked[i]:
                    side = FIXEDH(rear_fs * slide)
                    if side * self.wheel_angle > 0:
                        self.auto_brake = -1
                    f[0], f[2] = -lfz * self.thrust, lfx * self.thrust
                else:
                    side = FixHalfRound(front_fs * slide, 14)
            else:
                side = (front_fs * slide + 2048) >> 12
                if self.locked[i]:
                    side = (((front_fs * slide + 2048) >> 13) + side) >> 1
                    ff = FixHalfRound(FIXEDH(-side * lfx) * sdz - FIXEDH(-side * lfz) * sdx, 11)
                    f[0], f[2] = ff * sdz, -ff * sdx
            f[0] += (sus * n[0] - side * lfx) - vel[0] * 12
            f[2] += (sus * n[2] - side * lfz) - vel[2] * 12
            ang = m[1][1]
            if ang < 2048:
                ang = 4096 - ang
                ang = 4096 - FIXEDH(ang * ang) if ang <= 4096 else 0
                fc = FIXEDH(fc * ang)
            f[1] = FIXEDH(sus * n[1] - vel[1] * 12)
            f[0] = FIXEDH(f[0]) * fc >> 12
            f[2] = FIXEDH(f[2]) * fc >> 12
            for k in range(3):
                acc[k] += f[k]
            self.aacc[0] += FIXEDH(lp[1] * f[2] - lp[2] * f[1])
            self.aacc[1] += FIXEDH(lp[2] * f[0] - lp[0] * f[2])
            self.aacc[2] += FIXEDH(lp[0] * f[1] - lp[1] * f[0])
            self.comp[i] = new
        if self.comp[1] == 0 and self.comp[3] == 0:
            self.wheel_speed = (1703936 + 0x4000 if self.thrust >= 1 else
                                (-1245184 + 0x4000 if self.thrust <= -1 else 0))
        else:
            self.wheel_speed = cdiv(cdz, 64) * cdiv(vel[2], 64) + cdiv(cdx, 64) * cdiv(vel[0], 64)

    def convert_torque(self, avel, extra):
        m, a = self.m, self.aacc
        zd = (self.twist_z - self.twist_y) * FIXEDH(m[0][2] * a[0] + m[1][2] * a[1]
                                                   + m[2][2] * a[2])
        for i in range(3):
            a[i] = a[i] * self.twist_y + FIXEDH(m[i][2] * zd - avel[i] * ANGULAR_DAMPING)
            if extra:
                a[i] -= cdiv(avel[i], 8)

    # gamesnd.c ControlCarRevs / GetEngineRevs (player row)
    def control_revs(self):
        acc, old = self.thrust, self.revs
        self.changing_gear = 0
        if self.wheelspin == 0 and (self.comp[1] or self.comp[3] or acc == 0):
            target = self.engine_revs()
        else:
            target = 20160
            if self.comp[1] == 0 and self.comp[3] == 0:
                target = 30719
            old = max(old, 8000)
            self.gear = 0
        new = target
        if old - new > 1440:
            self.changing_gear = 1
            new = old - 1440
        if new - old > 1600:
            new = old + 1600
        self.revs = new

    def engine_revs(self):
        ws, acc = self.wheel_speed, self.thrust
        if ws > 0:
            ws >>= 11
            g = min(max(self.gear, 0), 3)
            for _ in range(8):
                lws = GEARD[g][0] if acc < 1 else GEARD[g][1]
                ng = g
                if ws < lws and g > 0:
                    ng = g - 1
                    if GEARD[ng][2] < ws:
                        ng = g
                elif GEARD[g][2] < ws:
                    ng = g + 1
                if ng == g:
                    break
                g = min(max(ng, 0), 3)
            self.gear = g
            return ws * (GEARD[g][3] if acc != 0 else GEARD[g][4])
        ws = cdiv(-ws, 2048)
        self.gear = 0
        return min(ws * (GEARD[0][3] if acc != 0 else GEARD[0][4]), 23000)

    def global_time_step(self):
        for i in range(3):
            self.vel[i] += self.acc[i]
            self.avel[i] += self.aacc[i]
        if self.vel[1] > 200000:
            self.vel[1] = cdiv(self.vel[1] * 3, 4)
        if self.speed == 0:
            if abs(self.vel[0]) + abs(self.vel[1]) + abs(self.vel[2]) < 1000:
                self.vel = [0, 0, 0]
                self.avel = [0, 0, 0]
            else:
                self.speed = 1
        self.avel = [max(min(a, 0x800000), -0x800000) for a in self.avel]
        st = self.fpos + self.orient + self.vel + self.avel
        for _ in range(4):
            d0 = self._delta(st)
            tp = [st[j] + (d0[j] >> 2) for j in range(13)]
            self._rebuild_state(tp)
            d1 = self._delta(tp)
            st = [st[j] + ((d0[j] + d1[j]) >> 3) for j in range(13)]
            self._rebuild_state(st)
        self.fpos, self.orient = st[0:3], st[3:7]
        self.direction = ratan2(self.m[0][2], self.m[2][2])

    def _rebuild_state(self, s):
        q = s[3:7]
        self.rebuild(s[0:3], q)
        s[3:7] = q

    @staticmethod
    def _delta(s):
        o = s[3:7]
        av = [FixHalfRound(s[10 + i], 13) for i in range(3)]
        return [s[7] >> 8, s[8] >> 8, s[9] >> 8,
                FIXEDH(-o[1] * av[2] + o[2] * av[1] + o[3] * av[0]),
                FIXEDH(o[0] * av[2] - o[2] * av[0] + o[3] * av[1]),
                FIXEDH(-o[0] * av[1] + o[1] * av[0] + o[3] * av[2]),
                FIXEDH(-o[0] * av[0] - o[1] * av[1] - o[2] * av[2]),
                0, 0, 0, 0, 0, 0]

    def frame(self, pad: int):
        self.process_pad(pad)
        self.step_one_car()
        self.control_revs()
        self.global_time_step()

    # readouts
    def speed_mps(self, units_per_metre=150.0):
        h = math.hypot(self.vel[0], self.vel[2])
        return h / 4096 * 30 / units_per_metre

    def pitch_deg(self):  # + = nose up
        return math.degrees(math.asin(max(-1.0, min(1.0, self.m[1][2] / 4096))))

    def roll_deg(self):
        return math.degrees(math.asin(max(-1.0, min(1.0, self.m[1][0] / 4096))))


def _demo():
    from core.car_profile import load_cosmetics
    cos = load_cosmetics()
    car = Car(cos)
    for _ in range(60):
        car.frame(0)
    t60 = None
    max_up = 0.0
    air = 0
    for f in range(1, 301):
        car.frame(PAD_ACCEL)
        max_up = max(max_up, car.pitch_deg())
        air += car.comp[0] == 0 and car.comp[2] == 0
        if t60 is None and car.speed_mps() >= 26.8224:
            t60 = f / 30
        if f % 30 == 0:
            print(f"t={f / 30:4.1f}s speed={car.speed:4d} u/frame ({car.speed_mps():5.2f} m/s)"
                  f" gear={car.gear} revs={car.revs} pitch={car.pitch_deg():5.2f}")
    print(f"0-60 mph {t60}s  max nose-up {max_up:.2f} deg  front-airborne frames {air}")
    car = Car(cos)
    for _ in range(60):
        car.frame(0)
    while car.speed_mps() < 30.0:
        car.frame(PAD_ACCEL)
    worst = 0.0
    for f in range(1, 91):
        car.frame(PAD_RIGHT)
        worst = max(worst, abs(car.roll_deg()))
        if f % 15 == 0:
            print(f"lock t={f / 30:4.2f}s speed={car.speed_mps():5.2f} m/s dir={car.direction}"
                  f" roll={car.roll_deg():6.2f} pitch={car.pitch_deg():5.2f}")
    print(f"full lock from 30 m/s: max |roll| {worst:.2f} deg")


if __name__ == "__main__":
    _demo()

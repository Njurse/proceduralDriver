"""The chase camera's follow rig: how the view trails the car, as pure maths.

game/chase_camera.py places the camera from what ``ChaseRig.update``
returns.  Everything is smoothed with frame-rate independent filters, so the
same drive looks the same at 30 or 144 fps:

* ``yaw``: the direction the camera looks along, trailing the car's heading
  (plus the orbit angle) through a critically damped spring, so it swings
  wide into turns and settles behind on the straight;
* ``dist``: the follow distance, pulled back with speed and by a lag on
  hard acceleration (and pushed in under braking), sprung the same way;
* ``fov``: widened with speed;
* ``roll``: a small lean into the turn from the lateral acceleration.
"""

import math
from dataclasses import dataclass


def smooth_damp(cur, target, vel, smooth_time, dt):
    """One step of a critically damped spring from ``cur`` toward ``target``
    (Game Programming Gems 4, 1.10): stable for any dt.  Returns (value, velocity)."""
    omega = 2.0 / max(smooth_time, 1e-4)
    x = omega * dt
    decay = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)
    change = cur - target
    temp = (vel + omega * change) * dt
    return target + (change + temp) * decay, (vel - omega * temp) * decay


def _follow(cur, target, dt, tau):
    """First-order low-pass, exact for any dt."""
    return cur + (target - cur) * (1.0 - math.exp(-dt / max(tau, 1e-4)))


def _wrap(a):
    return (a + math.pi) % math.tau - math.pi


@dataclass
class RigParams:
    dist: float = 9.4               # metres behind the car at rest
    fov: float = 66.0               # horizontal degrees at rest
    yaw_lag_s: float = 0.14         # spring time of the camera's swing after the heading
    dist_lag_s: float = 0.35        # spring time of the follow distance
    speed_ref: float = 45.0         # m/s at which the speed effects are full
    pull_back: float = 1.8          # extra metres behind at speed_ref
    fov_gain: float = 8.0           # extra degrees at speed_ref
    fov_tau: float = 0.6            # seconds
    accel_lag: float = 0.12         # metres back per m/s^2 of forward acceleration
    accel_lag_max: float = 1.2      # metres, either way
    accel_tau: float = 0.25         # smoothing of the acceleration signal
    roll_gain: float = 0.35         # degrees of lean per m/s^2 of lateral acceleration
    roll_max: float = 3.0           # degrees
    roll_tau: float = 0.3           # seconds
    max_dt: float = 0.1             # longer frames (hitches, pauses) count as this


class ChaseRig:
    """State of the follow rig; ``reset`` after a teleport."""

    def __init__(self, params=None):
        self.p = params or RigParams()
        self.reset()

    def reset(self):
        self.ready = False
        self.yaw = self.yaw_vel = 0.0
        self.dist = self.dist_vel = 0.0
        self.fov = self.p.fov
        self.roll = 0.0
        self.accel = 0.0
        self._fwd_speed = 0.0
        self._heading = 0.0

    def update(self, dt, heading, orbit_angle, speed, fwd_speed, snap=False):
        """Advance by ``dt``: returns (yaw, dist, fov, roll), yaw in the heading
        convention (the camera sits ``dist`` behind along it), roll in degrees
        (positive: lean right, into a right-hand turn).  ``snap`` puts the yaw
        straight on its target (a fixed camera angle)."""
        p = self.p
        target_yaw = heading - orbit_angle
        if not self.ready:
            self.ready = True
            self.yaw, self.dist = target_yaw, p.dist
            self._fwd_speed, self._heading = fwd_speed, heading
            return self.yaw, self.dist, self.fov, self.roll
        dt = min(max(dt, 0.0), p.max_dt)
        if dt <= 0.0:
            return self.yaw, self.dist, self.fov, self.roll
        # signals: forward acceleration and yaw rate, smoothed
        self.accel = _follow(self.accel, (fwd_speed - self._fwd_speed) / dt, dt, p.accel_tau)
        yaw_rate = _wrap(heading - self._heading) / dt
        self._fwd_speed, self._heading = fwd_speed, heading
        f = min(1.0, abs(speed) / p.speed_ref)
        # yaw: unwrap the target next to the current value, then spring to it
        target_yaw = self.yaw + _wrap(target_yaw - self.yaw)
        if snap:
            self.yaw, self.yaw_vel = target_yaw, 0.0
        else:
            self.yaw, self.yaw_vel = smooth_damp(self.yaw, target_yaw, self.yaw_vel,
                                                 p.yaw_lag_s, dt)
        lag = max(-p.accel_lag_max, min(p.accel_lag_max, self.accel * p.accel_lag))
        self.dist, self.dist_vel = smooth_damp(self.dist, p.dist + p.pull_back * f + lag,
                                               self.dist_vel, p.dist_lag_s, dt)
        self.fov = _follow(self.fov, p.fov + p.fov_gain * f, dt, p.fov_tau)
        lateral = abs(speed) * yaw_rate
        roll = max(-p.roll_max, min(p.roll_max, lateral * p.roll_gain))
        self.roll = _follow(self.roll, roll, dt, p.roll_tau)
        # keep yaw bounded without a jump in the spring's frame
        if abs(self.yaw) > 4.0 * math.pi:
            self.yaw = _wrap(self.yaw)
        return self.yaw, self.dist, self.fov, self.roll

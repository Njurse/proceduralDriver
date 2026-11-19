"""Smooth third-person chase camera for space and atmospheric flight."""

import logging
import math

from panda3d.core import Quat, Vec3

import config
from core.mathutil import clamp, damp

log = logging.getLogger("galaxy_explorer.camera")


def _quat_dot(a, b):
    return a.getR() * b.getR() + a.getI() * b.getI() + a.getJ() * b.getJ() + a.getK() * b.getK()


def _quat_nlerp(current, target, t):
    """Normalized lerp between quaternions (stable 'lerp' style blend)."""
    t = max(0.0, min(1.0, t))
    # Take shortest path.
    sign = 1.0
    if _quat_dot(current, target) < 0.0:
        sign = -1.0
    result = Quat(
        current.getR() + (sign * target.getR() - current.getR()) * t,
        current.getI() + (sign * target.getI() - current.getI()) * t,
        current.getJ() + (sign * target.getJ() - current.getJ()) * t,
        current.getK() + (sign * target.getK() - current.getK()) * t,
    )
    result.normalize()
    return result


class ChaseCamera:
    def __init__(self, base):
        self.base = base
        self.position = Vec3(0.0, 0.0, 0.0)
        self.quat = Quat()
        self.fov = 70.0
        self._initialised = False
        self.foot_pitch = 0.0
        try:
            base.camLens.setNearFar(0.5, 30000.0)
            base.camLens.setFov(self.fov)
        except Exception as exc:
            log.debug("[camera] lens setup failed: %s", exc)

    def _settings(self, mode):
        if mode == "atmosphere":
            return config.CAM_ATMO_DIST, config.CAM_ATMO_HEIGHT
        if mode == "onfoot":
            return config.FOOT_CAM_DIST, config.FOOT_CAM_HEIGHT
        return config.CAM_SPACE_DIST, config.CAM_SPACE_HEIGHT

    def _desired(self, ship_pos, quat, mode, height_sampler=None,
                 boost=False, speed_ratio=0.0):
        base_dist, base_height = self._settings(mode)
        # Pull the camera back and lower the angle at high speed/boost for a
        # more dramatic, wide-angle chase feel.
        speed_factor = 1.0 + 0.32 * clamp(speed_ratio, 0.0, 1.0)
        boost_factor = 1.32 if boost else 1.0
        distance = base_dist * speed_factor * boost_factor
        height = base_height * (1.0 + 0.15 * clamp(speed_ratio, 0.0, 1.0))
        if mode == "onfoot":
            # User-adjustable pitch (mouse up/down).
            height += getattr(self, "foot_pitch", 0.0) * 6.0
            distance += abs(getattr(self, "foot_pitch", 0.0)) * 1.5
        forward = quat.getForward()
        up = quat.getUp() * 0.62 + Vec3(0.0, 0.0, 1.0) * 0.38
        # Bank the camera slightly with the ship's roll in atmosphere mode for
        # a more dynamic, cinematic feel.
        if mode == "atmosphere":
            roll = quat.getHpr().z
            bank = math.radians(clamp(roll * 0.25, -12.0, 12.0))
            up = up * math.cos(bank) + quat.getRight() * math.sin(bank) * 0.3
        up.normalize()
        target = ship_pos - forward * distance + up * height
        if height_sampler is not None:
            # Collision: keep camera above terrain at target and at mid-point.
            ground = float(height_sampler(target.x, target.y))
            target.z = max(target.z, ground + 2.5)
            mid = (ship_pos + target) * 0.5
            try:
                gmid = float(height_sampler(mid.x, mid.y))
                if mid.z < gmid + 1.5:
                    target.z += (gmid + 1.5 - mid.z)
            except Exception:
                pass
        return target

    def snap(self, ship_pos, quat, mode, height_sampler=None):
        self.position = self._desired(ship_pos, quat, mode, height_sampler)
        self.quat = Quat(quat)
        self.quat.normalize()
        self._initialised = True
        self._apply()

    def update(self, dt, ship_pos, quat, mode, height_sampler=None,
              boost=False, speed=0.0):
        max_speed = (config.ATMO_MAX_SPEED if mode == "atmosphere"
                     else config.SPACE_MAX_SPEED)
        speed_ratio = clamp(speed / max(max_speed, 1.0), 0.0, 1.0)
        desired = self._desired(ship_pos, quat, mode, height_sampler, boost,
                                speed_ratio)
        target_quat = Quat(quat)
        target_quat.normalize()
        if not self._initialised:
            self.position = desired
            self.quat = target_quat
            self._initialised = True
        else:
            # More responsive orientation tracking for a tighter, more
            # exciting chase feel (especially during high-speed turns).
            smooth = config.CAM_SMOOTH * (1.35 if speed_ratio > 0.6 else 1.0)
            self.position = damp(self.position, desired, smooth, dt)
            rot_speed = config.CAM_SMOOTH * (2.1 if speed_ratio > 0.6 else 1.8)
            t = 1.0 - math.exp(-rot_speed * dt)
            self.quat = _quat_nlerp(self.quat, target_quat, t)
        self._apply()

        # Wider, more dramatic FOV during boost / high-speed flight.
        target_fov = 70.0
        if boost:
            target_fov = 82.0
        elif speed_ratio > 0.7:
            target_fov = 70.0 + 6.0 * (speed_ratio - 0.7) / 0.3
        self.fov = damp(self.fov, target_fov, 4.0, dt)
        try:
            self.base.camLens.setFov(self.fov)
        except Exception as exc:
            log.debug("[camera] FOV update failed: %s", exc)

    def _apply(self):
        camera = self.base.camera
        camera.setPos(self.position)
        camera.setQuat(self.quat)

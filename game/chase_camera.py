"""The chase camera: orbit from the right stick or I/J/K/L, trail the car on
the core.chase_rig springs (swing into turns, pull back and widen with speed,
lean into corners), bob and sway with the body, never under the hills (a
DriveSim mixin)."""

import math

import numpy as np

import config
from core.chase_rig import RigParams
from game.car_view import PHYS_TO_PANDA

CAM_TERRAIN_CLEARANCE = 0.6  # keep the chase camera this far above the hills
CAMERA_NEAR_M = 0.2          # close enough for the wheel close-up


class ChaseCamera:
    """Orbit input and the chase camera that hangs back from the car."""

    def _orbit_angle(self, dt):
        if self.cam_override is not None:
            self.orbit.angle = self.cam_override
            return self.cam_override
        sx = float(self.key_cam_right) - float(self.key_cam_left)
        sy = float(self.key_cam_down) - float(self.key_cam_up)
        click = self.key_cam_latch
        if self.pad.connected:
            sx = max(-1.0, min(1.0, sx + self.pad.right_x))
            sy = max(-1.0, min(1.0, sy + self.pad.right_y))
            click = click or self.pad.right_thumb
        if config.CHASE_CAM_FLIP_X:     # stick (and J/L) left: look round from the right
            sx = -sx
        return self.orbit.update(dt, sx, sy, click)

    def _update_chase_cam(self, dt):
        v = self.vehicle
        h = v.heading
        angle = self._orbit_angle(dt)
        speed = math.hypot(v.vel[0], v.vel[2])
        yaw, dist, fov, roll = self.cam_rig.update(dt, h, angle, speed, v.forward_speed,
                                                   snap=self.cam_override is not None)
        # car yaw frame -> physics world: right (cos h, 0, -sin h), fwd (sin h, 0, cos h)
        right = np.array([math.cos(h), 0.0, -math.sin(h)])
        fwd = np.array([math.sin(h), 0.0, math.cos(h)])
        # the rig's yaw trails heading - orbit angle: the camera sits dist behind
        # along it, as high as the rest framing scaled with the pull-back
        offset = -np.array([math.sin(yaw), 0.0, math.cos(yaw)]) * dist
        offset[1] = config.CHASE_CAM_HEIGHT * dist / config.CHASE_CAM_DIST
        # Rough ground and slides: the camera hangs back from the body's
        # vertical and sideways motion instead of being welded to it, so
        # bumps bob the view and drifts sway it.  First-order and clamped.
        k = 1.0 - math.exp(-dt / config.CHASE_CAM_BOB_TAU)
        kick = config.CHASE_CAM_KICK_MAX
        self.cam_vert += (v.vel[1] - self.cam_vert) * k
        self.cam_side += (float(v.vel @ right) - self.cam_side) * k
        offset[1] -= max(-kick, min(kick, self.cam_vert)) * config.CHASE_CAM_BOB
        offset -= right * (max(-kick, min(kick, self.cam_side)) * config.CHASE_CAM_SWAY)
        eye = v.pos + offset
        ground = self.terrain.height
        eye[1] = max(eye[1], ground(eye[0], eye[2]) + CAM_TERRAIN_CLEARANCE)
        # a crest between the car and the camera: lift the eye over it
        mid = 0.5 * (v.pos + eye)
        over = ground(mid[0], mid[2]) + CAM_TERRAIN_CLEARANCE - mid[1]
        if over > 0.0:
            eye[1] += 2.0 * over
        self.camera.setPos(*(PHYS_TO_PANDA @ eye))
        # Aim ahead of the car in the chase view, at the car from the
        # side/front; panning sideways lifts the aim point so the view opens
        # up past the car instead of staring at the roof.
        ahead = config.CHASE_CAM_LOOK_AHEAD * max(0.0, math.cos(angle))
        look_h = (config.CHASE_CAM_LOOK_HEIGHT
                  + config.CHASE_CAM_ORBIT_LOOK_RISE * abs(math.sin(angle)))
        aim = v.pos + fwd * ahead + np.array([0.0, look_h, 0.0])
        self.camera.lookAt(*(PHYS_TO_PANDA @ aim))
        self.camera.setR(roll)          # Panda: +R dips the right side
        self.camLens.setFov(fov)


def rig_params():
    """The chase rig's knobs from config."""
    return RigParams(
        dist=config.CHASE_CAM_DIST, fov=config.CHASE_CAM_FOV,
        yaw_lag_s=config.CHASE_CAM_YAW_LAG_S, dist_lag_s=config.CHASE_CAM_DIST_LAG_S,
        speed_ref=config.CHASE_CAM_SPEED_REF, pull_back=config.CHASE_CAM_PULL_BACK,
        fov_gain=config.CHASE_CAM_FOV_GAIN, accel_lag=config.CHASE_CAM_ACCEL_LAG,
        accel_lag_max=config.CHASE_CAM_ACCEL_LAG_MAX, roll_gain=config.CHASE_CAM_ROLL_GAIN,
        roll_max=config.CHASE_CAM_ROLL_MAX)

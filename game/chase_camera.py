"""The chase camera: orbit from the right stick or I/J/K/L, bob and sway
with the body, never under the hills (a DriveSim mixin)."""

import math

import numpy as np

import config
from core.orbit_camera import orbit_offset
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
        return self.orbit.update(dt, sx, sy, click)

    def _update_chase_cam(self, dt):
        v = self.vehicle
        h = v.heading
        angle = self._orbit_angle(dt)
        # car yaw frame -> physics world: right (cos h, 0, -sin h), fwd (sin h, 0, cos h)
        right = np.array([math.cos(h), 0.0, -math.sin(h)])
        fwd = np.array([math.sin(h), 0.0, math.cos(h)])
        ox, oy, oz = orbit_offset(angle, config.CHASE_CAM_DIST, config.CHASE_CAM_HEIGHT)
        offset = right * ox + fwd * oz
        offset[1] = oy
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
        eye[1] = max(eye[1], self.terrain.height(eye[0], eye[2]) + CAM_TERRAIN_CLEARANCE)
        self.camera.setPos(*(PHYS_TO_PANDA @ eye))
        # Aim ahead of the car in the chase view, at the car from the
        # side/front; panning sideways lifts the aim point so the view opens
        # up past the car instead of staring at the roof.
        ahead = config.CHASE_CAM_LOOK_AHEAD * max(0.0, math.cos(angle))
        look_h = (config.CHASE_CAM_LOOK_HEIGHT
                  + config.CHASE_CAM_ORBIT_LOOK_RISE * abs(math.sin(angle)))
        aim = v.pos + fwd * ahead + np.array([0.0, look_h, 0.0])
        self.camera.lookAt(*(PHYS_TO_PANDA @ aim))

"""On-foot controller and cube avatar.

The foot controller mirrors the flight controllers' interface (position, quat,
speed) so the game can treat the walker as just another controller, while
remaining grounded on a heightfield.
"""

import math

from panda3d.core import Quat, Vec3

import config
from core.geometry import make_box
from core.mathutil import clamp, damp


class FootController:
    """Grounded third-person walker."""

    def __init__(self):
        self.position = Vec3(0.0, 0.0, 0.0)
        self.velocity = Vec3(0.0, 0.0, 0.0)
        self.quat = Quat()
        self.yaw = 0.0
        self.on_ground = True
        self.speed = 0.0
        self._jump_pressed_prev = False
        self._coyote = 0.0
        self._jump_buffer = 0.0
        self._landed_this_frame = False
        self.turn_rate = 0.0
        # Jetpack.
        self.jetpack_fuel = config.JETPACK_FUEL_MAX
        self.jetpack_active = False

    def reset(self, position, yaw=0.0):
        self.position = Vec3(position)
        self.velocity = Vec3(0.0, 0.0, 0.0)
        self.yaw = float(yaw)
        self.quat = Quat()
        self.quat.setHpr((self.yaw, 0.0, 0.0))
        self.on_ground = True
        self.speed = 0.0
        self._coyote = config.FOOT_COYOTE_TIME
        self._jump_buffer = 0.0
        self._landed_this_frame = False
        self.turn_rate = 0.0

    def update(self, dt, inp, mouse_dx, mouse_dy, height_sampler, bounds=None):
        # Turn with mouse.
        turn_input = (-mouse_dx * config.FOOT_TURN_SENSITIVITY * 60.0
                     - float(getattr(inp, "yaw_axis", 0.0)) * 90.0)
        self.turn_rate = damp(self.turn_rate, turn_input, 12.0, dt)
        self.yaw += self.turn_rate * dt

        # Movement input in body frame.
        forward_input = float(getattr(inp, "throttle_axis", 0.0))
        strafe_input = float(getattr(inp, "lateral_axis", 0.0))

        cos_y = math.cos(math.radians(self.yaw))
        sin_y = math.sin(math.radians(self.yaw))
        fx, fy = sin_y, cos_y
        rx, ry = cos_y, -sin_y

        move_x = fx * forward_input + rx * strafe_input * config.FOOT_STRAFE_MULT
        move_y = fy * forward_input + ry * strafe_input * config.FOOT_STRAFE_MULT
        length = math.hypot(move_x, move_y)
        if length > 1e-6:
            move_x /= length
            move_y /= length

        sprint = bool(getattr(inp, "sprint", False))
        target_speed = config.FOOT_WALK_SPEED
        if sprint:
            target_speed *= config.FOOT_SPRINT_MULT

        # Horizontal velocity smoothing (snappier).
        desired_hx = move_x * target_speed
        desired_hy = move_y * target_speed
        alpha = 1.0 - math.exp(-14.0 * dt)
        self.velocity.x += (desired_hx - self.velocity.x) * alpha
        self.velocity.y += (desired_hy - self.velocity.y) * alpha

        # friction when no input
        if abs(forward_input) < 0.05 and abs(strafe_input) < 0.05:
            fric = math.exp(-8.0 * dt)
            self.velocity.x *= fric
            self.velocity.y *= fric

        # Gravity.
        self.velocity.z -= config.FOOT_GRAVITY * dt

        # Coyote / jump buffer timers.
        jump_now = bool(getattr(inp, "jump", False))
        if jump_now and not self._jump_pressed_prev:
            self._jump_buffer = config.FOOT_JUMP_BUFFER
        self._jump_pressed_prev = jump_now
        if self.on_ground:
            self._coyote = config.FOOT_COYOTE_TIME
        else:
            self._coyote = max(0.0, self._coyote - dt)
        self._jump_buffer = max(0.0, self._jump_buffer - dt)

        # Jump (buffered + coyote).
        if self._jump_buffer > 0.0 and self._coyote > 0.0:
            self.velocity.z = config.FOOT_JUMP_SPEED
            self.on_ground = False
            self._coyote = 0.0
            self._jump_buffer = 0.0
            self._landed_this_frame = False

        # --- Jetpack fuel ---
        # Recharge only while grounded and not currently jumping.
        if self.on_ground and not (self._jump_buffer > 0.0 and self._coyote > 0.0):
            self.jetpack_fuel = min(
                config.JETPACK_FUEL_MAX,
                self.jetpack_fuel + config.JETPACK_RECHARGE * dt,
            )
        # Drain when jetpack held and fuel > 0.
        if inp.jetpack and self.jetpack_fuel > 0.0:
            self.jetpack_active = True
            # Upward thrust.
            self.velocity.z += config.JETPACK_THRUST * dt
            # Forward assist along horizontal velocity direction.
            horz_len = math.hypot(self.velocity.x, self.velocity.y)
            if horz_len > 0.0:
                forward_x = self.velocity.x / horz_len
                forward_y = self.velocity.y / horz_len
                self.velocity.x += config.JETPACK_FORWARD_ASSIST * forward_x * dt
                self.velocity.y += config.JETPACK_FORWARD_ASSIST * forward_y * dt
            self.jetpack_fuel -= config.JETPACK_DRAIN * dt
            if self.jetpack_fuel < 0.0:
                self.jetpack_fuel = 0.0
        else:
            self.jetpack_active = False

        # Integrate.
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        self.position.z += self.velocity.z * dt

        # Ground clamp.
        self._landed_this_frame = False
        if height_sampler is not None:
            try:
                ground = float(height_sampler(self.position.x, self.position.y))
            except Exception:
                ground = 0.0
            floor = ground + config.FOOT_FOOT_HEIGHT
            if self.position.z <= floor:
                if not self.on_ground and self.velocity.z < 0.0:
                    self._landed_this_frame = True
                self.position.z = floor
                if self.velocity.z < 0.0:
                    self.velocity.z = 0.0
                self.on_ground = True
            else:
                self.on_ground = False

        # Bounds (hangar).
        if bounds is not None:
            try:
                if not bounds(self.position):
                    # push back: simple clamp by velocity zero
                    self.velocity.x *= 0.0
                    self.velocity.y *= 0.0
            except Exception:
                pass

        self.quat.setHpr((self.yaw, 0.0, 0.0))
        self.speed = math.hypot(self.velocity.x, self.velocity.y)


class FootAvatar:
    """A cube that rocks while walking."""

    def __init__(self, base):
        self.base = base
        self.root = base.render.attachNewNode("foot_root")
        self.body = make_box(config.FOOT_AVATAR_SIZE,
                             config.FOOT_AVATAR_SIZE,
                             config.FOOT_AVATAR_SIZE, "foot_cube")
        self.body.reparentTo(self.root)
        self._phase = 0.0
        self._roll = 0.0
        self._bob = 0.0
        self._squash = 0.0
        self._bank = 0.0
        self._spin = 0.0

    def attach_to(self, parent):
        self.root.reparentTo(parent)

    def set_transform(self, position, quat):
        self.root.setPos(position)
        self.root.setQuat(quat)

    def update(self, dt, moving, speed, turn_rate=0.0, landed=False):
        if moving and speed > 0.1:
            self._phase += dt * (6.0 + speed * 0.6)
            target_roll = math.sin(self._phase) * 8.0
            target_bob = abs(math.sin(self._phase)) * 0.12
        else:
            target_roll = 0.0
            target_bob = 0.0
        # bank into turns
        target_bank = clamp(turn_rate * 0.06, -12.0, 12.0)
        # squash on landing
        if landed:
            self._squash = -config.FOOT_LAND_SQUASH
        self._squash = damp(self._squash, 0.0, 6.0, dt)
        # spin impulse on jump handled externally; decay
        self._spin = damp(self._spin, 0.0, 4.0, dt)

        self._roll = damp(self._roll, target_roll, 8.0, dt)
        self._bob = damp(self._bob, target_bob, 8.0, dt)
        self._bank = damp(self._bank, target_bank, 6.0, dt)
        sx = 1.0 + self._squash * 0.5
        sy = 1.0 + self._squash * 0.5
        sz = 1.0 - self._squash
        self.body.setScale(sx, sy, sz)
        self.body.setR(self._roll)
        self.body.setP(self._bank)
        self.body.setZ(self._bob)

    def destroy(self):
        import contextlib
        with contextlib.suppress(Exception):
            self.root.removeNode()

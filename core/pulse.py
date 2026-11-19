"""Pulse engine: fast travel with countdown, lock, and instant pullout."""

import logging

from panda3d.core import Vec3

import config

log = logging.getLogger("galaxy_explorer.pulse")


def _surface_normal(sampler, x, y, eps=2.0):
    dx = (float(sampler(x + eps, y)) - float(sampler(x - eps, y))) / (2.0 * eps)
    dy = (float(sampler(x, y + eps)) - float(sampler(x, y - eps))) / (2.0 * eps)
    n = Vec3(-dx, -dy, 1.0)
    n.normalize()
    return n


class PulseDrive:
    IDLE = "idle"
    COUNTDOWN = "countdown"
    ACTIVE = "active"

    def __init__(self, fuel):
        self.fuel = fuel
        self.state = self.IDLE
        self.countdown = 0.0
        self.y_hold = 0.0
        self.pullout_flash = 0.0   # remaining seconds of flash
        self.fov_rebound = 0.0     # remaining seconds of FOV rebound
        self._pulse_speed = 0.0

    def can_pulse(self, controller, height_sampler):
        """Whether pulsing is allowed right now."""
        if height_sampler is None:
            return True   # space/system: no surface constraint
        try:
            ground = float(height_sampler(controller.position.x,
                                         controller.position.y))
        except Exception as exc:
            log.debug("[pulse] height sample failed: %s", exc)
            return True
        agl = controller.position.z - ground
        if agl < config.PULSE_MIN_ALTITUDE:
            return False
        try:
            normal = _surface_normal(height_sampler, controller.position.x,
                                    controller.position.y)
        except Exception as exc:
            log.debug("[pulse] surface normal failed: %s", exc)
            normal = Vec3(0, 0, 1)
        forward = controller.quat.getForward()
        if forward.dot(normal) < config.PULSE_SURFACE_COS:
            return False
        return True

    def update(self, dt, inp, controller, height_sampler, audio, fx,
               can_pulse_mode):
        pulsing = (self.state == self.ACTIVE)
        self.fuel.regen(dt, pulsing)

        # Pullout (instant) during active pulse.
        if self.state == self.ACTIVE:
            pullout = (getattr(inp, "brake", False)
                       or getattr(inp, "throttle_axis", 0.0) < -0.2)
            if pullout or self.fuel.pulse_fuel <= 0:
                self._pullout(controller, audio, fx)
                return
            # Drain fuel.
            self.fuel.drain_pulse(dt)
            if self.fuel.pulse_fuel <= 0:
                self._pullout(controller, audio, fx)
                return
            # Locked trajectory: no rotation, high speed.
            forward = controller.quat.getForward()
            speed = (config.SPACE_MAX_SPEED * config.SPACE_BOOST_MULT
                     * config.PULSE_SPEED_MULT)
            self._pulse_speed = speed
            controller.position += forward * speed * dt
            controller.speed = speed
            controller.throttle = 1.0
            controller.velocity = forward * speed
            fx.set_active(True)
            audio.set_pulse(True)
            fx.set_countdown(0.0)
            return

        # Cooldown timers for flash/rebound.
        if self.pullout_flash > 0:
            self.pullout_flash = max(0.0, self.pullout_flash - dt)
        if self.fov_rebound > 0:
            self.fov_rebound = max(0.0, self.fov_rebound - dt)

        fx.set_active(False)
        audio.set_pulse(False)

        # Y-hold tracking.
        if getattr(inp, "boost", False):
            self.y_hold += dt
        else:
            if self.state == self.COUNTDOWN:
                self.state = self.IDLE
                self.countdown = 0.0
                fx.set_countdown(0.0)
            self.y_hold = 0.0

        allowed = can_pulse_mode and self.fuel.can_pulse() and self.can_pulse(
            controller, height_sampler)

        if self.state == self.IDLE:
            if (self.y_hold >= 0.3 and allowed):
                self.state = self.COUNTDOWN
                self.countdown = config.PULSE_COUNTDOWN
        elif self.state == self.COUNTDOWN:
            self.countdown -= dt
            fx.set_countdown(self.countdown)
            if self.countdown <= 0:
                self.countdown = 0.0
                self.state = self.ACTIVE
                fx.set_countdown(0.0)
                audio.play_pulse_start()
            if not getattr(inp, "boost", False):
                self.state = self.IDLE
                self.countdown = 0.0
                fx.set_countdown(0.0)

        if self.state != self.ACTIVE:
            fx.set_countdown(self.countdown if self.state == self.COUNTDOWN else 0.0)

    def _pullout(self, controller, audio, fx):
        self.state = self.IDLE
        self.countdown = 0.0
        self.y_hold = 0.0
        controller.speed = config.PULSE_PULLOUT_SPEED
        controller.throttle = 0.3
        controller.velocity = controller.quat.getForward() * controller.speed
        self.pullout_flash = config.PULSE_PULLOUT_FLASH
        self.fov_rebound = config.PULSE_PULLOUT_FLASH + 0.35
        fx.trigger_pullout()
        audio.play_pulse_end()
        audio.set_pulse(False)

    def to_dict(self):
        return {"fuel": self.fuel.to_dict()}

    @classmethod
    def from_dict(cls, data):
        from core.fuel import FuelSystem
        fs = FuelSystem.from_dict(data.get("fuel") if isinstance(data, dict) else None)
        return cls(fs)

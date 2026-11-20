"""Engine sound model: rpm, load and mix levels from the car's state.

Pure math, no audio device: core/vehicle_audio.py turns each Mix into
samples.  Every level is smoothed here so nothing jumps between frames.

  rpm   wheel speed x gear ratio once the clutch is in; at low speed (and
        with the rear wheels spinning or in the air) it blends toward a
        free-revving value set by the throttle.  Never below idle or above
        redline.  An upshift drops it quickly to the new gear's rpm.
  load  the throttle, cut to zero during a gear change: on-load is louder
        and brighter than off-load (engine braking).
  mix   idle loop at low rpm, engine loop above, with an equal-power
        crossfade; skid loop from tyre slip with on/off hysteresis.
"""

import math
from dataclasses import dataclass

import config
from core.engine import wheel_rpm


@dataclass
class SoundInput:
    speed: float = 0.0  # m/s along the heading, signed
    gear: int = 0
    changing_gear: bool = False
    throttle: float = 0.0  # 0..1
    wheelspin: bool = False
    rear_airborne: bool = False
    slip: float = 0.0  # m/s, the largest tyre slip
    handbrake: bool = False


@dataclass(frozen=True)
class Mix:
    rpm: float = config.IDLE_RPM
    load: float = 0.0  # 0 off-load .. 1 full throttle
    idle_level: float = 0.0
    engine_level: float = 0.0
    skid_level: float = 0.0


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def approach(current: float, target: float, dt: float, tau: float) -> float:
    """First-order lag with time constant ``tau`` seconds."""
    if tau <= 0.0:
        return target
    return target + (current - target) * math.exp(-dt / tau)


def target_rpm(inp: SoundInput) -> float:
    free = config.IDLE_RPM + inp.throttle * (config.FREE_REV_RPM - config.IDLE_RPM)
    clutch = smoothstep(0.0, config.CLUTCH_FULL_SPEED, abs(inp.speed))
    if inp.wheelspin or inp.rear_airborne:
        clutch *= config.SPIN_CLUTCH
    rpm = free + (wheel_rpm(inp.speed, inp.gear) - free) * clutch
    return min(max(rpm, config.IDLE_RPM), config.REDLINE_RPM)


def crossfade(rpm: float) -> tuple[float, float]:
    """(idle, engine) equal-power weights."""
    x = smoothstep(config.IDLE_RPM * 1.1, config.IDLE_RPM * 2.0, rpm)
    return math.cos(x * math.pi / 2.0), math.sin(x * math.pi / 2.0)


class EngineSound:
    def __init__(self):
        self.rpm = config.IDLE_RPM
        self.load = 0.0
        self.idle = config.IDLE_GAIN
        self.engine = 0.0
        self.skidding = False
        self.skid = 0.0

    def update(self, dt: float, inp: SoundInput) -> Mix:
        target = target_rpm(inp)
        if target > self.rpm:
            tau = config.RPM_RISE_TIME
        elif inp.changing_gear:
            tau = config.RPM_SHIFT_DROP_TIME
        else:
            tau = config.RPM_FALL_TIME
        self.rpm = approach(self.rpm, target, dt, tau)
        load = 0.0 if inp.changing_gear else inp.throttle
        self.load = approach(self.load, load, dt, config.LOAD_TIME)

        idle_w, engine_w = crossfade(self.rpm)
        on, off = config.ENGINE_GAIN_ON_LOAD, config.ENGINE_GAIN_OFF_LOAD
        engine = (off + (on - off) * self.load) * engine_w
        self.engine = approach(self.engine, engine, dt, config.LEVEL_TIME)
        self.idle = approach(self.idle, config.IDLE_GAIN * idle_w, dt, config.LEVEL_TIME)

        slip = inp.slip
        if inp.handbrake and abs(inp.speed) > config.SKID_HB_SPEED_MIN:
            slip = max(slip, config.SKID_SLIP_ON)
        if slip > config.SKID_SLIP_ON:
            self.skidding = True
        elif slip < config.SKID_SLIP_OFF:
            self.skidding = False
        skid = 0.0
        if self.skidding:
            span = config.SKID_SLIP_FULL - config.SKID_SLIP_OFF
            skid = config.SKID_GAIN * min(max((slip - config.SKID_SLIP_OFF) / span, 0.3), 1.0)
        tau = config.SKID_ATTACK_TIME if skid > self.skid else config.SKID_RELEASE_TIME
        self.skid = approach(self.skid, skid, dt, tau)
        return Mix(self.rpm, self.load, self.idle, self.engine, self.skid)

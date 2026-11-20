"""Vehicle audio output: streams the engine sound model through pygame.

Every frame core/engine_sound.EngineSound gives a Mix (rpm, load, levels).
This module renders it into short sample blocks and queues them on one
pygame channel, so pitch follows rpm continuously:

- each loop (engine, idle, skid) is read with a fractional playback rate
  (linear interpolation), engine rate = rpm / ENGINE_SAMPLE_RPM;
- rates and levels ramp linearly across each block from the last block's
  values, so nothing clicks or zippers;
- off-load the engine is low-passed (muffled); on-load it is the raw loop;
- loop ends are crossfaded into their starts once at load time.
"""

import logging
import math
import os

import numpy as np

import config
from core.engine_sound import EngineSound, Mix, SoundInput

try:
    import pygame

    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

log = logging.getLogger("vehicle_audio")

LOOP_FADE_S = 0.03
MUFFLE_TAPS = 8  # moving-average low-pass for the off-load timbre


def sound_input(vehicle) -> SoundInput:
    """Snapshot of a core.redriver2_physics.Vehicle for the sound model."""
    return SoundInput(
        speed=vehicle.forward_speed,
        gear=vehicle.gear,
        changing_gear=vehicle.changing_gear,
        throttle=vehicle.throttle_input,
        wheelspin=bool(vehicle.wheelspin),
        rear_airborne=vehicle.compression[1] == 0.0 and vehicle.compression[3] == 0.0,
        slip=skid_slip(vehicle),
    )


def skid_slip(vehicle) -> float:
    """Tyre slip the skid loop follows, m/s; 0 when no tyre is skidding.

    handling.c CheckCarEffects screeches only with a rear wheel on the ground
    and the rear axle sliding sideways (|hd.rear_vel| > 15000) or spinning.
    Here: the sideways or spin slip of grounded rear tyres plus the slip of
    grounded locked tyres (handbrake, braking lockup).  Front sideways slide
    is left out (steered fronts slide a little in any turn), wheels in the air
    are left out, and below SKID_SPEED_MIN nothing skids unless the rears spin.
    """
    if not vehicle.wheelspin and math.hypot(vehicle.vel[0], vehicle.vel[2]) < config.SKID_SPEED_MIN:
        return 0.0
    slip = 0.0
    for i, w in enumerate(vehicle.wheels):
        if w.compression <= 0.0:
            continue
        if i & 1:
            slip = max(slip, w.slip_lateral, w.slip_longitudinal)
        elif w.locked:
            slip = max(slip, w.slip_longitudinal)
    return slip


def seamless(samples: np.ndarray, fade: int) -> np.ndarray:
    """Crossfade the last ``fade`` samples into the first so the loop wraps
    without a click.  The result is ``fade`` samples shorter."""
    fade = min(fade, len(samples) // 2)
    if fade < 2:
        return samples.astype(np.float32)
    out = samples[:-fade].astype(np.float32)
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] = samples[:fade] * ramp + samples[-fade:] * (1.0 - ramp)
    return out


class Loop:
    """A looping mono buffer read at a smoothly varying playback rate."""

    def __init__(self, samples: np.ndarray):
        self.data = samples
        self.phase = 0.0

    def render(self, n: int, rate0: float, rate1: float) -> np.ndarray:
        steps = np.linspace(rate0, rate1, n, endpoint=False)
        pos = self.phase + np.concatenate(([0.0], np.cumsum(steps[:-1])))
        size = len(self.data)
        pos %= size
        i0 = pos.astype(np.int64)
        frac = (pos - i0).astype(np.float32)
        i1 = (i0 + 1) % size
        self.phase = float((self.phase + steps.sum()) % size)
        return self.data[i0] * (1.0 - frac) + self.data[i1] * frac


class AudioManager:
    def __init__(self, cosmetics, muted=False):
        self.cos = cosmetics
        self.muted = muted
        self.model = EngineSound()
        self.mix = Mix()
        self._last = Mix()
        self.channel = None
        self.loops = {}
        self._muffle_tail = np.zeros(MUFFLE_TAPS - 1, np.float32)
        if _HAS_PYGAME and not muted:
            self._init_mixer()

    def _init_mixer(self):
        try:
            pygame.mixer.init(frequency=config.AUDIO_RATE, size=-16, channels=2,
                              buffer=512)
        except Exception as exc:
            log.warning("[audio] mixer unavailable; vehicle audio disabled: %s", exc)
            return
        self.rate = pygame.mixer.get_init()[0]
        for key in ("engine", "idle", "skid"):
            path = self.cos.sounds.get(key, "")
            if not path or not os.path.exists(path):
                log.warning("[audio] %s sound missing: %s", key, path)
                continue
            try:
                arr = pygame.sndarray.array(pygame.mixer.Sound(path)).astype(np.float32)
            except Exception as exc:
                log.warning("[audio] failed to load %s: %s", path, exc)
                continue
            mono = arr.mean(axis=1) if arr.ndim == 2 else arr
            self.loops[key] = Loop(seamless(mono / 32768.0, int(LOOP_FADE_S * self.rate)))
        self.channel = pygame.mixer.Channel(0)

    # ------------------------------------------------------------ frame
    def update(self, dt, inp: SoundInput):
        self.mix = self.model.update(dt, inp)
        if self.channel is None:
            return
        for _ in range(2):  # keep one block playing and one queued
            if not self.channel.get_busy():
                self.channel.play(self._block())
            elif self.channel.get_queue() is None:
                self.channel.queue(self._block())
            else:
                break

    def _voice(self, key, n, rate0, rate1, level0, level1):
        loop = self.loops.get(key)
        if loop is None:
            return np.zeros(n, np.float32)
        return loop.render(n, rate0, rate1) * np.linspace(level0, level1, n, endpoint=False)

    def _block(self):
        n = config.AUDIO_BLOCK
        a, b = self._last, self.mix
        er = config.ENGINE_SAMPLE_RPM
        ir = config.IDLE_SAMPLE_RPM
        engine = self._voice("engine", n, a.rpm / er, b.rpm / er, a.engine_level, b.engine_level)
        padded = np.concatenate((self._muffle_tail, engine))
        self._muffle_tail = padded[-(MUFFLE_TAPS - 1):]
        muffled = np.convolve(padded, np.full(MUFFLE_TAPS, 1.0 / MUFFLE_TAPS), "valid")
        bright = np.linspace(a.load, b.load, n, endpoint=False)
        engine = muffled + (engine - muffled) * bright
        idle = self._voice("idle", n, a.rpm / ir, b.rpm / ir, a.idle_level, b.idle_level)
        skid = self._voice("skid", n, 1.0, 1.0, a.skid_level, b.skid_level)
        out = np.tanh((engine + idle + skid) * config.AUDIO_MASTER_VOLUME)
        self._last = b
        pcm = (out * 32767.0).astype(np.int16)
        return pygame.sndarray.make_sound(np.ascontiguousarray(np.stack((pcm, pcm), axis=1)))

    def set_paused(self, paused):
        """Hold the engine loop where it is (pause menu) or let it go on."""
        if self.channel is not None:
            (self.channel.pause if paused else self.channel.unpause)()

    def shutdown(self):
        if self.channel is None:
            return
        try:
            self.channel.stop()
            pygame.mixer.quit()
        except Exception as exc:
            log.debug("[audio] mixer quit failed: %s", exc)
        self.channel = None

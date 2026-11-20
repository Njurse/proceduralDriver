"""Vehicle audio: engine and idle loops driven by the gearbox revs.

Ports the player-car half of gamesnd.c: ControlCarRevs ramps two volumes
(revsvol, idlevol, in millibels), and the per-frame sound update sets

    engine pitch = revs / 4 + revsvol / 64 + 1500   (4096 = 1.0)
    idle pitch   = revs / 4 + 4096

so the engine note follows the revs, including the dip at every upshift.
pygame cannot change a sound's pitch, so each loop is pre-resampled into a
small bank of pitches that all loop at once; the two bank entries around the
wanted pitch share its volume.  Skid stays a simple slip-triggered loop.
"""

import logging
import math
import os
from dataclasses import dataclass

try:
    import config
except ModuleNotFoundError:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

try:
    import numpy as np
    import pygame

    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False

from core.units import C_FRAME_HZ

log = logging.getLogger("vehicle_audio")

SILENT_MB = -10000.0
IDLE_REVS = 7001.0  # below this with no gas the idle loop takes over
ENGINE_BANK = (0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.5)  # playback-rate ratios
IDLE_BANK = (1.0, 1.25, 1.5)


@dataclass
class VehicleAudioState:
    speed: float = 0.0  # m/s
    revs: float = 0.0  # gamesnd.c units, about 0..30000
    accelerating: bool = False  # thrust != 0, the C's "acc"
    handbrake: int = 0
    wheelspin: int = 0
    slip_lateral: float = 0.0  # m/s
    slip_longitudinal: float = 0.0


@dataclass(frozen=True)
class EngineVolumes:
    revsvol: float = SILENT_MB
    idlevol: float = -8000.0


def step_volumes(vol: EngineVolumes, dt: float, *, revs: float, accelerating: bool,
                 wheelspin: bool) -> EngineVolumes:
    """ControlCarRevs' player volume ramps (millibels), per C frame x dt."""
    k = dt * C_FRAME_HZ
    if not accelerating and revs < IDLE_REVS:
        idle = min(vol.idlevol + 200.0 * k, -6000.0)
        revsvol = max(vol.revsvol - 200.0 * k, SILENT_MB)
    else:
        revs_max = -5500.0 if accelerating else -6750.0
        idle = max(vol.idlevol - (256.0 if wheelspin else 64.0) * k, SILENT_MB)
        revsvol = min(vol.revsvol + (700.0 if wheelspin else 175.0) * k, revs_max)
    return EngineVolumes(revsvol, idle)


def engine_pitch(revs: float, revsvol: float) -> float:
    return (revs / 4.0 + revsvol / 64.0 + 1500.0) / 4096.0


def idle_pitch(revs: float) -> float:
    return (revs / 4.0 + 4096.0) / 4096.0


def gain(millibels: float) -> float:
    """Millibels to a linear gain, normalised so -5500 (the loudest the C
    drives the engine) is full volume; -10000 is silence."""
    if millibels <= SILENT_MB:
        return 0.0
    return min(1.0, 10.0 ** (millibels / 2000.0) * 10.0 ** (5500.0 / 2000.0))


def bank_weights(pitch: float, bank) -> list[float]:
    """Volume share of each bank entry: the two around ``pitch`` crossfade
    in log-pitch; outside the bank the nearest end takes it all."""
    w = [0.0] * len(bank)
    if pitch <= bank[0]:
        w[0] = 1.0
        return w
    if pitch >= bank[-1]:
        w[-1] = 1.0
        return w
    for i in range(len(bank) - 1):
        lo, hi = bank[i], bank[i + 1]
        if lo <= pitch <= hi:
            t = math.log(pitch / lo) / math.log(hi / lo)
            w[i], w[i + 1] = 1.0 - t, t
            return w
    return w


def _resample(snd, ratio):
    arr = pygame.sndarray.array(snd)
    n = arr.shape[0]
    m = max(2, int(n / ratio))
    x = np.linspace(0.0, n - 1, m)
    src = np.arange(n)
    if arr.ndim == 1:
        out = np.interp(x, src, arr)
    else:
        out = np.stack([np.interp(x, src, arr[:, c]) for c in range(arr.shape[1])], axis=1)
    return pygame.sndarray.make_sound(np.ascontiguousarray(out.astype(arr.dtype)))


class _Voice:
    """One looped sound played at several pitches on its own channels."""

    def __init__(self, path, bank, first_channel):
        self.bank = bank
        self.channels = []
        if not path or not os.path.exists(path):
            log.warning("[audio] sound missing: %s", path)
            return
        base = pygame.mixer.Sound(path)
        for i, ratio in enumerate(bank):
            ch = pygame.mixer.Channel(first_channel + i)
            snd = base if ratio == 1.0 else _resample(base, ratio)
            ch.play(snd, loops=-1)
            ch.set_volume(0.0)
            self.channels.append(ch)

    def set(self, pitch, volume):
        for ch, w in zip(self.channels, bank_weights(pitch, self.bank), strict=False):
            ch.set_volume(volume * w)

    def stop(self):
        for ch in self.channels:
            ch.stop()


class AudioManager:
    def __init__(self, cosmetics, muted=False):
        self.cos = cosmetics
        self.muted = muted
        self.volumes = EngineVolumes()
        self.engine_vol = 0.0
        self.idle_vol = 0.0
        self.skid_vol = 0.0
        self.skid_active = False
        self.engine = self.idle = None
        self.skid_ch = self.skid_snd = None
        if _HAS_PYGAME:
            self._init_mixer()

    def _init_mixer(self):
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=2)
            pygame.mixer.set_num_channels(len(ENGINE_BANK) + len(IDLE_BANK) + 2)
        except Exception as exc:
            log.warning("[audio] mixer unavailable; vehicle audio disabled: %s", exc)
            return
        snds = self.cos.sounds
        try:
            self.engine = _Voice(snds.get("engine", ""), ENGINE_BANK, 0)
            self.idle = _Voice(snds.get("idle", ""), IDLE_BANK, len(ENGINE_BANK))
            path = snds.get("skid", "")
            if path and os.path.exists(path):
                self.skid_snd = pygame.mixer.Sound(path)
                self.skid_ch = pygame.mixer.Channel(len(ENGINE_BANK) + len(IDLE_BANK))
                self.skid_ch.play(self.skid_snd, loops=-1)
                self.skid_ch.set_volume(0.0)
        except Exception as exc:
            log.warning("[audio] failed to load sounds: %s", exc)

    def update(self, dt, state: VehicleAudioState):
        self.volumes = step_volumes(self.volumes, dt, revs=state.revs,
                                    accelerating=state.accelerating,
                                    wheelspin=bool(state.wheelspin))
        master = 0.0 if self.muted else 1.0
        self.engine_vol = gain(self.volumes.revsvol)
        self.idle_vol = gain(self.volumes.idlevol)
        if self.engine is not None:
            self.engine.set(engine_pitch(state.revs, self.volumes.revsvol),
                            self.engine_vol * master)
        if self.idle is not None:
            self.idle.set(idle_pitch(state.revs), self.idle_vol * master)
        self._update_skid(dt, state)
        if self.skid_ch is not None:
            self.skid_ch.set_volume(self.skid_vol * master)

    def _update_skid(self, dt, state):
        slip = max(abs(state.slip_lateral), abs(state.slip_longitudinal))
        threshold = config.SKID_SLIP_THRESHOLD
        trigger = (slip > threshold
                   or (state.handbrake and abs(state.speed) > config.SKID_HB_SPEED_MIN))
        if trigger:
            self.skid_active = True
        elif slip < config.SKID_SLIP_RELEASE:
            self.skid_active = False
        if self.skid_active:
            lo, hi = config.SKID_VOLUME_MIN, config.SKID_VOLUME_MAX
            self.skid_vol = lo + (hi - lo) * min(slip / threshold, 1.0)
        else:
            fade = config.SKID_FADE_OUT_MS / 1000.0
            self.skid_vol = max(0.0, self.skid_vol - dt / fade)

    def shutdown(self):
        if not _HAS_PYGAME:
            return
        for voice in (self.engine, self.idle):
            if voice is not None:
                voice.stop()
        try:
            pygame.mixer.quit()
        except Exception as exc:
            log.debug("[audio] mixer quit failed: %s", exc)

"""Engine / wind / transition / UI audio, with procedurally cached loops."""

import logging
import math
import wave

import numpy as np
from panda3d.core import Filename

import config
from core.mathutil import clamp, damp

log = logging.getLogger("galaxy_explorer.audio")

_RATE = 22050


def _write_wav(path, samples, rate=_RATE, peak=0.8):
    data = np.asarray(samples, dtype=np.float64)
    maximum = float(np.abs(data).max()) if data.size else 0.0
    if maximum > 1e-9:
        data = data / maximum * peak
    pcm = (np.clip(data, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(pcm.tobytes())


def _warp(path):
    """Layered cinematic warp: sub-bass dive, noise riser, sweep, impact.

    Four layers mixed and soft-clipped so the transition feels deep and
    dramatic instead of a thin single-band whoosh:

    1. a sub-bass swell whose pitch *dives* (52 -> 28 Hz) for weight,
    2. dual-band filtered noise (body + hiss) with an accelerating riser,
    3. a rising sine sweep (90 -> 900 Hz) with tremolo shimmer,
    4. an end impact: pitch-dropping thump plus a bright noise burst.
    """
    if path.exists():
        return
    rng = np.random.default_rng(2027)
    seconds = 2.4
    count = int(_RATE * seconds)
    time = np.arange(count, dtype=np.float64) / _RATE

    # Layer 1: sub-bass dive for the low "the floor drops out" weight.
    sub_freq = 52.0 - 24.0 * (time / seconds)
    sub_phase = np.cumsum(sub_freq) / _RATE
    sub = np.sin(math.tau * sub_phase)
    swell = np.minimum(time / 0.45, 1.0)
    release = np.exp(-np.maximum(0.0, time - 1.55) * 3.5)
    sub = sub * swell * release

    # Layer 2: two smoothed noise bands, enveloped as an accelerating riser.
    body = np.convolve(rng.standard_normal(count),
                       np.ones(320) / 320, mode="same")
    hiss = np.convolve(rng.standard_normal(count),
                       np.ones(56) / 56, mode="same")
    riser = (time / seconds) ** 1.9
    whoosh = (body * 1.7 + hiss * 0.8) * riser

    # Layer 3: rising sweep with tremolo, kept under the noise at the end.
    sweep_freq = 90.0 + 810.0 * (time / seconds) ** 1.4
    sweep_phase = np.cumsum(sweep_freq) / _RATE
    tremolo = 0.55 + 0.45 * np.sin(math.tau * 6.0 * time) ** 2
    sweep = np.sin(math.tau * sweep_phase) * tremolo * riser ** 1.1

    # Layer 4: terminal impact - thump with falling pitch plus a bright burst.
    hit_start = seconds - 0.45
    impact = np.where(time >= hit_start,
                      np.exp(-(time - hit_start) * 11.0), 0.0)
    thump_freq = 34.0 + 120.0 * np.exp(-np.maximum(0.0, time - hit_start) * 7.0)
    thump_phase = np.cumsum(thump_freq) / _RATE
    thump = np.sin(math.tau * thump_phase) * impact * 1.25
    burst = (np.convolve(rng.standard_normal(count),
                         np.ones(22) / 22, mode="same") * impact * 0.95)

    mix = np.tanh((sub + whoosh + sweep * 0.8 + thump + burst) * 1.15)
    _write_wav(path, mix, peak=0.85)


def _hangar_hum(path):
    """Low freighter-hangar hum loop: beating low tones plus airy shimmer."""
    if path.exists():
        return
    rng = np.random.default_rng(5150)
    seconds = 6.0
    count = int(_RATE * seconds)
    time = np.arange(count, dtype=np.float64) / _RATE
    # Detuned pair around 48 Hz beats slowly; the octave adds body.
    hum = (np.sin(math.tau * 48.0 * time)
           + np.sin(math.tau * 48.7 * time) * 0.8
           + np.sin(math.tau * 96.3 * time) * 0.35)
    swell = 0.78 + 0.22 * np.sin(math.tau * 0.21 * time)
    air = np.convolve(rng.standard_normal(count),
                      np.ones(180) / 180, mode="same") * 0.10
    output = (hum * 0.30 + air) * swell
    fade = int(_RATE * 0.5)
    head = output[:fade].copy()
    ramp = np.linspace(0.0, 1.0, fade)
    output[-fade:] = output[-fade:] * (1.0 - ramp) + head * ramp
    _write_wav(path, output, peak=0.5)


def _pulse(path):
    if path.exists():
        return
    rng = np.random.default_rng(7711)
    seconds = 2.0
    count = int(_RATE * seconds)
    time = np.arange(count, dtype=np.float64) / _RATE
    rumble = np.sin(math.tau * 40.0 * time) * 0.6
    sweep = np.sin(math.tau * (60.0 + 200.0 * time) * time) * 0.35
    noise = rng.standard_normal(count) * 0.08
    smooth = np.convolve(noise, np.ones(80) / 80, mode="same")
    env = 0.5 + 0.5 * np.sin(math.tau * 0.5 * time)
    output = (rumble + sweep + smooth) * env
    fade = int(_RATE * 0.3)
    head = output[:fade].copy()
    ramp = np.linspace(0.0, 1.0, fade)
    output[-fade:] = output[-fade:] * (1.0 - ramp) + head * ramp
    _write_wav(path, output, peak=0.6)


def _wind(path):
    if path.exists():
        return
    rng = np.random.default_rng(4041)
    seconds = 4.0
    count = int(_RATE * seconds)
    time = np.arange(count, dtype=np.float64) / _RATE
    noise = rng.standard_normal(count)
    smooth = np.convolve(noise, np.ones(240) / 240, mode="same")
    envelope = (0.68 + 0.20 * np.sin(math.tau * 0.31 * time + 1.1)
                + 0.10 * np.sin(math.tau * 0.83 * time + 4.0))
    output = smooth * envelope
    fade = int(_RATE * 0.6)
    head = output[:fade].copy()
    ramp = np.linspace(0.0, 1.0, fade)
    output[-fade:] = output[-fade:] * (1.0 - ramp) + head * ramp
    _write_wav(path, output, peak=0.55)


def _click(path):
    """Short UI tick, also used as the volume-preview sound."""
    if path.exists():
        return
    count = int(_RATE * 0.09)
    time = np.arange(count, dtype=np.float64) / _RATE
    envelope = np.exp(-time * 48.0)
    tone = (np.sin(math.tau * 880.0 * time) * 0.7
            + np.sin(math.tau * 1320.0 * time) * 0.3)
    _write_wav(path, tone * envelope, peak=0.7)


class AudioManager:
    def __init__(self, base):
        self.base = base
        self.engine = None
        self.wind = None
        self.whoosh = None
        self.pulse = None
        self.ui_click = None
        self.hangar_hum = None
        self.engine_path = None
        self._engine_volume = 0.0
        self._wind_volume = 0.0
        self._pulse_volume = 0.0
        self.master_volume = float(getattr(config, "MASTER_VOLUME", 1.0))
        self.effects_volume = float(getattr(config, "EFFECTS_VOLUME", 1.0))
        self.engine_rate_mult = 1.0
        self._last_engine_rate = 1.0
        self._last_wind_rate = 1.0
        self._retiring = []  # (sound, base volume, seconds left)
        config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            _wind(config.CACHE_DIR / "wind_loop.wav")
            _warp(config.CACHE_DIR / "warp.wav")
            _hangar_hum(config.CACHE_DIR / "hangar_hum.wav")
            _pulse(config.CACHE_DIR / "pulse_loop.wav")
            _click(config.CACHE_DIR / "ui_click.wav")
        except OSError as exc:
            log.warning("[audio] could not generate cached sounds: %s", exc)
        self._load()

    def set_ship_profile(self, profile):
        if not isinstance(profile, dict):
            return
        self.engine_rate_mult = float(profile.get("engine_rate", 1.0))
        path = profile.get("engine_wav")
        if not path or path == self.engine_path:
            return
        if self.engine is not None:
            # Crossfade the old loop instead of stopping it abruptly.
            self._retiring.append((self.engine, self._engine_volume,
                                   config.ENGINE_FADE_SECONDS))
            self.engine = None
        self.engine = self._load_sound(path)
        self.engine_path = path
        self._engine_volume = 0.0  # crossfade in via update()
        if self.engine is not None:
            try:
                self.engine.setLoop(True)
                self.engine.setVolume(0.0)
                self.engine.play()
            except Exception as exc:
                log.warning("[audio] could not play engine %s: %s", path, exc)
                self.engine = None

    def set_volume_settings(self, master=None, effects=None):
        if master is not None:
            self.master_volume = clamp(float(master), 0.0, 1.0)
        if effects is not None:
            self.effects_volume = clamp(float(effects), 0.0, 1.0)
        scale = self._effect_scale
        if self.engine is not None:
            try:
                self.engine.setVolume(self._engine_volume * scale)
            except Exception as exc:
                log.debug("[audio] engine volume update failed: %s", exc)
        if self.wind is not None:
            try:
                self.wind.setVolume(self._wind_volume * scale)
            except Exception as exc:
                log.debug("[audio] wind volume update failed: %s", exc)

    @property
    def _effect_scale(self):
        return self.master_volume * self.effects_volume

    def _load_sound(self, path):
        try:
            handle = Filename.fromOsSpecific(str(path))
            sound = self.base.loader.loadSfx(handle)
            if sound is not None and sound.length() > 0.01:
                return sound
        except Exception as exc:
            log.debug("[audio] could not load %s: %s", path, exc)
        return None

    def _load(self):
        self.engine = None
        self.engine_path = None
        self.wind = self._load_sound(config.CACHE_DIR / "wind_loop.wav")
        if self.wind is not None:
            try:
                self.wind.setLoop(True)
                self.wind.setVolume(0.0)
                self.wind.setPlayRate(1.0)
                self.wind.play()
            except Exception as exc:
                log.warning("[audio] wind loop failed: %s", exc)
                self.wind = None
        self.warp = self._load_sound(config.CACHE_DIR / "warp.wav")
        self.hangar_hum = self._load_sound(config.CACHE_DIR / "hangar_hum.wav")
        if self.hangar_hum is not None:
            try:
                self.hangar_hum.setLoop(True)
                self.hangar_hum.setVolume(0.0)
                self.hangar_hum.setPlayRate(1.0)
                self.hangar_hum.play()
            except Exception as exc:
                log.warning("[audio] hangar hum failed: %s", exc)
                self.hangar_hum = None
        self.pulse = self._load_sound(config.CACHE_DIR / "pulse_loop.wav")
        self.ui_click = self._load_sound(config.CACHE_DIR / "ui_click.wav")
        if self.pulse is not None:
            try:
                self.pulse.setLoop(True)
                self.pulse.setVolume(0.0)
            except Exception as exc:
                log.warning("[audio] pulse loop failed: %s", exc)
                self.pulse = None

    def play_ui_click(self):
        """Menu navigation tick; doubles as the volume-preview sound."""
        if self.ui_click is None:
            return
        try:
            self.ui_click.stop()
            self.ui_click.setVolume(config.UI_CLICK_VOLUME
                                    * (0.5 + 0.5 * self.master_volume))
            self.ui_click.play()
        except Exception as exc:
            log.debug("[audio] UI click failed: %s", exc)

    def update(self, dt, mode, throttle, speed_ratio, boost):
        if self.engine is None and self.wind is None and not self._retiring:
            return
        speed_ratio = clamp(speed_ratio, 0.0, 1.4)
        if mode == "atmosphere":
            engine_target = 0.07 + 0.30 * abs(throttle)
            wind_target = 0.10 + 0.52 * speed_ratio
        else:
            engine_target = 0.14 + 0.52 * abs(throttle)
            wind_target = 0.0
            if boost and throttle > 0.05:
                engine_target += 0.12

        self._engine_volume = damp(self._engine_volume, engine_target, 5.0, dt)
        self._wind_volume = damp(self._wind_volume, wind_target, 3.0, dt)
        scale = self._effect_scale
        if self.engine is not None:
            try:
                self.engine.setVolume(self._engine_volume * scale)
                rate = 0.75 + 0.50 * abs(throttle)
                if boost and throttle > 0.05:
                    rate += 0.30
                if mode == "atmosphere":
                    rate = 0.85 + 0.30 * abs(throttle)
                self.engine.setPlayRate(rate * self.engine_rate_mult)
                self._last_engine_rate = rate
            except Exception as exc:
                log.debug("[audio] engine update failed: %s", exc)
                self.engine = None
        if self.wind is not None:
            try:
                self.wind.setVolume(self._wind_volume * scale)
                rate = 0.85 + 0.35 * clamp(speed_ratio, 0.0, 1.0)
                self.wind.setPlayRate(rate)
                self._last_wind_rate = rate
            except Exception as exc:
                log.debug("[audio] wind update failed: %s", exc)
                self.wind = None
        self._update_retiring(dt)

    def _update_retiring(self, dt):
        if not self._retiring:
            return
        remain = []
        scale = self._effect_scale
        for sound, base, seconds in self._retiring:
            seconds -= dt
            if seconds <= 0.0:
                try:
                    sound.stop()
                except Exception as exc:
                    log.debug("[audio] retiring engine stop failed: %s", exc)
                continue
            try:
                sound.setVolume(base * seconds / config.ENGINE_FADE_SECONDS * scale)
            except Exception as exc:
                log.debug("[audio] retiring engine fade failed: %s", exc)
                continue
            remain.append((sound, base, seconds))
        self._retiring = remain

    def set_hangar_hum(self, level):
        """Freighter-hangar proximity hum, 0..1 (0 mutes the loop)."""
        if self.hangar_hum is None:
            return
        try:
            self.hangar_hum.setVolume(
                clamp(float(level), 0.0, 1.0) * 0.55 * self._effect_scale)
        except Exception as exc:
            log.debug("[audio] hangar hum update failed: %s", exc)

    def play_transition(self):
        if self.warp is None:
            return
        try:
            self.warp.stop()
            self.warp.setVolume(0.85 * self._effect_scale)
            self.warp.play()
        except Exception as exc:
            log.debug("[audio] transition sound failed: %s", exc)

    def set_pulse(self, active):
        if self.pulse is None:
            return
        try:
            if active:
                if self.pulse.status() != 2:  # 2 = playing
                    self.pulse.play()
                self.pulse.setVolume(0.55 * self._effect_scale)
                self.pulse.setPlayRate(1.0)
            else:
                self.pulse.stop()
        except Exception as exc:
            log.debug("[audio] pulse sound failed: %s", exc)

    def play_pulse_start(self):
        self.play_transition()
        self.set_pulse(True)

    def play_pulse_end(self):
        if self.warp is not None:
            try:
                self.warp.stop()
                self.warp.setVolume(0.6 * self._effect_scale)
                self.warp.play()
            except Exception as exc:
                log.debug("[audio] pullout sound failed: %s", exc)
        self.set_pulse(False)

    def stop_all(self):
        for sound in (self.engine, self.wind, self.warp, self.pulse,
                      self.ui_click, self.hangar_hum):
            if sound is not None:
                try:
                    sound.stop()
                except Exception as exc:
                    log.debug("[audio] stop failed: %s", exc)
        for sound, _, _ in self._retiring:
            try:
                sound.stop()
            except Exception as exc:
                log.debug("[audio] retiring stop failed: %s", exc)
        self._retiring.clear()

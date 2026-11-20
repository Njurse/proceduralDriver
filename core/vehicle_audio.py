"""Vehicle audio controller (decoupled from physics).

Follows PORTING_NOTES.md section 5.  Uses pygame.mixer with reserved channels
for engine / idle / skid.  Receives a per-frame VehicleAudioState snapshot.
"""

import os
import logging
try:
    import config
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config

try:
    import pygame
    _HAS_PYGAME = True
except Exception:
    _HAS_PYGAME = False


class VehicleAudioState:
    def __init__(self):
        self.speed = 0.0
        self.throttle = 0.0
        self.handbrake = 0
        self.wheelspin = 0
        self.wheel_speeds = [0, 0, 0, 0]
        self.slip_lateral = 0.0
        self.slip_longitudinal = 0.0
        self.locked = [0, 0, 0, 0]


class AudioManager:
    def __init__(self, cosmetics, muted=False):
        self.cos = cosmetics
        self.muted = muted
        self.engine_ch = None
        self.idle_ch = None
        self.skid_ch = None
        self.engine_snd = None
        self.idle_snd = None
        self.skid_snd = None
        self.engine_vol = 0.0
        self.idle_vol = 0.0
        self.skid_vol = 0.0
        self.skid_active = False
        self.skid_cooldown = 0.0
        self.engine_load_factor = 0.0
        self._init_mixer()

    def _init_mixer(self):
        log = logging.getLogger("vehicle_audio")
        if not _HAS_PYGAME:
            log.warning("[audio] pygame is unavailable; vehicle audio disabled")
            return
        try:
            pygame.mixer.init(frequency=22050, size=-16, channels=3)
        except Exception as exc:
            log.warning("[audio] mixer init failed (%s); trying defaults", exc)
            try:
                pygame.mixer.init()
            except Exception as exc2:
                log.warning("[audio] mixer unavailable; vehicle audio disabled: %s",
                            exc2)
                return
        try:
            pygame.mixer.set_num_channels(8)
        except Exception as exc:
            log.debug("[audio] set_num_channels failed: %s", exc)
        self.engine_ch = pygame.mixer.Channel(0)
        self.idle_ch = pygame.mixer.Channel(1)
        self.skid_ch = pygame.mixer.Channel(2)

        snds = self.cos.sounds
        log.debug("[audio] sound paths: %s", snds)
        for key, attr in (("engine", "engine_snd"), ("idle", "idle_snd"),
                          ("skid", "skid_snd")):
            path = snds.get(key, "")
            if not path or not os.path.exists(path):
                log.warning("[audio] %s sound missing: %s", key, path)
                continue
            try:
                snd = pygame.mixer.Sound(path)
                snd.set_volume(0.0)
                setattr(self, attr, snd)
            except Exception as exc:
                log.warning("[audio] failed to load %s sound %s: %s",
                            key, path, exc)

        # start loops at zero volume
        if self.engine_snd is not None:
            self.engine_ch.play(self.engine_snd, loops=-1)
            self.engine_ch.set_volume(0.0)
        if self.idle_snd is not None:
            self.idle_ch.play(self.idle_snd, loops=-1)
            self.idle_ch.set_volume(0.0)
        if self.skid_snd is not None:
            self.skid_ch.play(self.skid_snd, loops=-1)
            self.skid_ch.set_volume(0.0)

        log = logging.getLogger("vehicle_audio")
        log.info("[audio] engine=%s idle=%s skid=%s",
                 self.engine_snd is not None, self.idle_snd is not None,
                 self.skid_snd is not None)

    def _clamp(self, v, lo, hi):
        return max(lo, min(hi, v))

    def update(self, dt, state: VehicleAudioState):
        if not _HAS_PYGAME:
            return
        # RPM proxy
        ws = [abs(x) for x in state.wheel_speeds]
        redline = getattr(config, "REDLINE_WHEEL_SPEED", 3300)
        rpm_proxy = self._clamp((max(ws) if ws else 0) / max(1.0, redline), 0.0, 1.0)

        throttle = self._clamp(state.throttle, 0.0, 1.0)
        engine_load = throttle
        engine_speed_factor = rpm_proxy * engine_load

        speed_nominal = getattr(config, "SPEED_NOMINAL", 120.0)
        speed_factor_norm = self._clamp(abs(state.speed) / max(1.0, speed_nominal), 0.0, 1.0)
        engine_load_factor = self._clamp(throttle * speed_factor_norm, 0.0, 1.0)

        # engine pitch/volume
        pmin = getattr(config, "ENGINE_PITCH_MIN", 0.8)
        pmax = getattr(config, "ENGINE_PITCH_MAX", 1.5)
        vmin = getattr(config, "ENGINE_VOLUME_MIN", 0.3)
        vmax = getattr(config, "ENGINE_VOLUME_MAX", 1.0)
        engine_pitch = pmin + (pmax - pmin) * engine_speed_factor
        engine_vol_target = vmin + (vmax - vmin) * engine_load_factor

        # idle conditions
        idle_enter_speed = getattr(config, "IDLE_ENTER_SPEED", 8.0)
        idle_exit_speed = getattr(config, "IDLE_EXIT_SPEED", 12.0)
        idle_thr_eps = getattr(config, "IDLE_THROTTLE_EPS", 0.05)

        # crossfade engine <-> idle
        if (abs(state.speed) < idle_enter_speed and throttle < idle_thr_eps
                and state.handbrake == 0 and state.wheelspin == 0):
            # fade to idle
            self.idle_vol = min(1.0, self.idle_vol + dt * 5.0)
            self.engine_vol = max(0.0, self.engine_vol - dt * 5.0)
        elif (abs(state.speed) > idle_exit_speed or throttle > 0.1):
            self.idle_vol = max(0.0, self.idle_vol - dt * 5.0)
            self.engine_vol = min(1.0, self.engine_vol + dt * 5.0)
        else:
            # hold
            pass

        # blend targets
        self.engine_vol = self.engine_vol * 0.9 + engine_vol_target * 0.1 * (1.0 - self.idle_vol)
        self.engine_vol = self._clamp(self.engine_vol, 0.0, 1.0)

        # apply engine pitch via... pygame Sound doesn't support pitch natively.
        # We approximate by volume only; pitch would need resampling (out of scope).
        if self.engine_ch is not None and self.engine_snd is not None:
            self.engine_ch.set_volume(self.engine_vol * (0.0 if self.muted else 1.0))
        if self.idle_ch is not None and self.idle_snd is not None:
            self.idle_ch.set_volume(self.idle_vol * (0.0 if self.muted else 1.0))

        # skid
        skid_thr = getattr(config, "SKID_SLIP_THRESHOLD", 0.4)
        skid_release = getattr(config, "SKID_SLIP_RELEASE", 0.2)
        skid_hb_min = getattr(config, "SKID_HB_SPEED_MIN", 4.0)
        skid_cooldown = getattr(config, "SKID_COOLDOWN_S", 0.05)

        slip_mag = max(abs(state.slip_lateral), abs(state.slip_longitudinal))
        trigger = False
        if slip_mag > skid_thr:
            trigger = True
        if state.wheelspin == 1 and throttle > 0.3:
            trigger = True
        if state.handbrake == 1 and abs(state.speed) > skid_hb_min:
            trigger = True
        if abs(state.slip_longitudinal) > getattr(config, "SKID_LON_THRESHOLD", 0.5) if hasattr(config, "SKID_LON_THRESHOLD") else 0.5:
            trigger = True

        self.skid_cooldown = max(0.0, self.skid_cooldown - dt)

        if trigger and not self.skid_active and self.skid_cooldown <= 0:
            self.skid_active = True
            self.skid_cooldown = skid_cooldown
        if not trigger and slip_mag < skid_release:
            self.skid_active = False

        if self.skid_active:
            vol_min = getattr(config, "SKID_VOLUME_MIN", 0.2)
            vol_max = getattr(config, "SKID_VOLUME_MAX", 1.0)
            slip_norm = self._clamp(slip_mag / max(0.01, skid_thr), 0.0, 1.0)
            self.skid_vol = vol_min + (vol_max - vol_min) * slip_norm
        else:
            fade_ms = getattr(config, "SKID_FADE_OUT_MS", 200) / 1000.0
            self.skid_vol = max(0.0, self.skid_vol - dt / max(0.01, fade_ms))

        if self.skid_ch is not None and self.skid_snd is not None:
            self.skid_ch.set_volume(self.skid_vol * (0.0 if self.muted else 1.0))

        self.engine_load_factor = engine_load_factor
        self.rpm_proxy = rpm_proxy

    def shutdown(self):
        if not _HAS_PYGAME:
            return
        for ch in (self.engine_ch, self.idle_ch, self.skid_ch):
            try:
                if ch is not None:
                    ch.stop()
            except Exception:
                pass
        try:
            pygame.mixer.quit()
        except Exception:
            pass


if __name__ == "__main__":
    from core.car_profile import load_cosmetics
    cos = load_cosmetics()
    am = AudioManager(cos)
    st = VehicleAudioState()
    st.speed = 50.0
    st.throttle = 0.7
    st.wheel_speeds = [3000, 2900, 3100, 2800]
    st.slip_lateral = 0.5
    am.update(1 / 60.0, st)
    print("engine_vol", round(am.engine_vol, 3), "idle_vol", round(am.idle_vol, 3),
          "skid_vol", round(am.skid_vol, 3), "skid_active", am.skid_active)
    am.shutdown()
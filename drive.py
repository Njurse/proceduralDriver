"""Standalone 3D driving simulator (REDRIVER2 handling, Panda3D renderer).

Run:  python drive.py            (windowed)
      python drive.py --offscreen --frames N
"""

import argparse
import logging
import math
import sys

import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    DirectionalLight,
    NodePath,
    Texture,
    TextureStage,
    Vec3,
    Vec4,
)

import config
from core.car_profile import load_cosmetics
from core.vehicle3d import Vehicle3D, Input
from core.textures import seamless_fbm, _to_texture
from core.vehicle_audio import AudioManager, VehicleAudioState

log = logging.getLogger("drive")

WORLD_SIZE = 400.0
CAM_DIST = 10.0
CAM_HEIGHT = 3.6


class DriveSim(ShowBase):
    def __init__(self, options=None):
        options = dict(options or {})
        ShowBase.__init__(self,
                          windowType=options.get("window_type", "onscreen"))
        self.disableMouse()
        self.clock = ClockObject.getGlobalClock()

        self.cos = load_cosmetics()
        self.vehicle = Vehicle3D(self.cos)
        self.input = Input()

        self._setup_window()
        self._build_world()
        self._build_car()
        self._build_hud()
        self._bind_keys()

        self.audio = AudioManager(self.cos)
        self.audio_state = VehicleAudioState()
        self.spin = 0.0

        self.taskMgr.add(self._update, "drive-update")

    # ------------------------------------------------------------ setup
    def _setup_window(self):
        self.setBackgroundColor(0.35, 0.46, 0.62, 1.0)
        try:
            props = self.win and self.win.getProperties()
        except Exception:
            props = None
        try:
            from panda3d.core import WindowProperties
            p = WindowProperties()
            p.setTitle("Driver2-ish 3D — Havana")
            p.setSize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            self.win.requestProperties(p)
        except Exception as exc:
            log.debug("[window] %s", exc)

    def _build_world(self):
        # asphalt texture: mid grey with fbm speckle
        size = 256
        base = np.full((size, size), 96.0)
        speckle = seamless_fbm(size, size, 24.0, 1337, 4) * 70.0 - 35.0
        fine = seamless_fbm(size, size, 80.0, 4242, 3) * 26.0 - 13.0
        grey = np.clip(base + speckle + fine, 40, 190).astype(np.uint8)
        rgb = np.stack([grey, grey, grey + 6], axis=-1)
        rgba = np.concatenate([rgb, np.full((size, size, 1), 255, np.uint8)],
                              axis=-1)
        self.ground_tex = _to_texture(rgba, "asphalt", repeat=True)

        cm = CardMaker("ground")
        cm.setFrame(-WORLD_SIZE / 2, WORLD_SIZE / 2,
                    -WORLD_SIZE / 2, WORLD_SIZE / 2)
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setP(-90)
        self.ground.setPos(0.0, 0.0, 0.0)
        self.ground.setTexture(self.ground_tex, 1)
        ts = TextureStage.getDefault()
        self.ground.setTexScale(ts, WORLD_SIZE / 6.0, WORLD_SIZE / 6.0)

        amb = AmbientLight("amb")
        amb.setColor(Vec4(0.55, 0.55, 0.62, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))
        dl = DirectionalLight("sun")
        dl.setColor(Vec4(0.95, 0.92, 0.85, 1.0))
        dl_node = self.render.attachNewNode(dl)
        dl_node.setHpr(45, -35, 0)
        self.render.setLight(dl_node)

    def _build_car(self):
        import os
        obj_path = getattr(config, "HAVANA_MODEL_OBJ", None)
        # Make the OBJ's own directory visible to Panda3D's model path so its
        # relative texture references (textures/PAGE_36.png) resolve.
        try:
            self.getModelPath().appendDir(os.path.dirname(obj_path))
        except Exception as exc:
            log.debug("[car] model path append failed: %s", exc)
        rel = None
        try:
            rel = os.path.relpath(obj_path, config.ROOT)
        except Exception:
            rel = obj_path
        try:
            self.car = self.loader.loadModel(rel)
        except Exception as exc:
            log.warning("[car] load failed: %s", exc)
            self.car = NodePath("fallback_car")
        if self.car.isEmpty():
            self.car = NodePath("fallback_car")

        scale = getattr(config, "HAVANA_SHIP_SCALE", 0.01)
        self.car.setScale(scale)
        # car authored with inverted vertical coords -> flip Y (local space)
        if getattr(config, "CAR_MODEL_INVERT_VERTICAL", False):
            self.car.setScale(scale, -scale, scale)

        # Rotate so the model's long axis (Z) aligns with Panda3D +Y (forward).
        self.car.setP(getattr(config, "DRIVE_CAR_PITCH_OFFSET", -90.0))
        self.car.setH(getattr(config, "DRIVE_CAR_YAW_OFFSET", 180.0))

        # Center the model on its tight bounds so it doesn't orbit its pivot.
        try:
            low, high = self.car.getTightBounds()
            self.car.setPos(-(low + high) * 0.5)
        except Exception as exc:
            log.debug("[car] centering failed: %s", exc)

        self.car.reparentTo(self.render)

        # Wheel disc nodes (spinning markers) at the wheel offsets.
        self.wheel_nodes = []
        self.wheel_spin = 0.0
        wheel_tex_path = None
        try:
            wheel_tex_path = self.cos.wheel_textures.get("clean")
        except Exception:
            wheel_tex_path = None
        wheel_tex = None
        if wheel_tex_path:
            try:
                import os
                load_path = wheel_tex_path
                try:
                    load_path = os.path.relpath(wheel_tex_path, config.ROOT)
                except ValueError:
                    pass
                wheel_tex = self.loader.loadTexture(load_path)
            except Exception as exc:
                log.debug("[car] wheel texture load failed: %s", exc)
                wheel_tex = None
        wheel_scale = getattr(config, "HAVANA_WHEEL_SCALE", 0.5)
        for off in self.vehicle.wheel_offsets:
            # physics (x, y_up, z) -> panda (x, z, y_up)
            wn = self.render.attachNewNode("wheel")
            # build a flat disc via CardMaker
            try:
                from panda3d.core import CardMaker
                cm = CardMaker("wheeldisc")
                cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                disc = wn.attachNewNode(cm.generate())
                disc.setScale(wheel_scale)
                if wheel_tex is not None:
                    disc.setTexture(wheel_tex, 1)
                # orient disc so its normal points along the axle (local X):
                # CardMaker normal is +Z; roll -90 makes it +X.
                disc.setR(-90.0)
            except Exception as exc:
                log.debug("[car] wheel disc build failed: %s", exc)
            wn.setPos(off[0], off[2], off[1])
            wn.reparentTo(self.car)
            self.wheel_nodes.append(wn)

    # ------------------------------------------------------------- input
    def _bind_keys(self):
        self.key_throttle = False
        self.key_brake = False
        self.key_left = False
        self.key_right = False
        self.key_hb = False
        self.key_fast = False
        for name, flag in [("w", "key_throttle"), ("s", "key_brake"),
                           ("a", "key_left"), ("d", "key_right"),
                           ("space", "key_hb"), ("lshift", "key_fast")]:
            self.accept(name, self._set_key, [flag, True])
            self.accept(name + "-up", self._set_key, [flag, False])
        self.accept("escape", self._quit)

    def _set_key(self, flag, value):
        setattr(self, flag, value)

    def _quit(self):
        sys.exit(0)

    # ------------------------------------------------------------- hud
    def _build_hud(self):
        self.speed_label = OnscreenText(
            text="0 km/h", pos=(-0.72, -0.82), scale=0.12,
            fg=(1, 1, 1, 1), align=2, shadow=(0, 0, 0, 0.6))
        self.state_label = OnscreenText(
            text="", pos=(-0.72, -0.93), scale=0.07,
            fg=(1, 1, 1, 1), align=2, shadow=(0, 0, 0, 0.6))

    # ------------------------------------------------------------- frame
    def _update(self, task):
        dt = min(self.clock.getDt(), 0.05)
        if dt <= 0.0:
            dt = 1.0 / 60.0

        # input -> vehicle
        inp = self.input
        inp.throttle = 1.0 if self.key_throttle else 0.0
        inp.brake = 1.0 if self.key_brake else 0.0
        inp.steer = (-1.0 if self.key_left else 0.0) + \
                    (1.0 if self.key_right else 0.0)
        inp.handbrake = self.key_hb
        inp.fast_steer = self.key_fast
        self.vehicle.process_input(inp)
        self.vehicle.step(dt, substeps=4)

        self._sync_car()
        self._update_chase_cam()
        self._update_hud()
        self._update_audio(dt)
        return task.cont

    def _sync_car(self):
        v = self.vehicle
        # physics (x, y_up, z) -> panda (x, y_fwd=z, z_up=y_up)
        self.car.setPos(v.pos[0], v.pos[2], v.pos[1])
        self.car.setHpr(
            math.degrees(v.yaw) + getattr(config, "DRIVE_CAR_YAW_OFFSET", 180.0),
            math.degrees(v.pitch) + getattr(config, "DRIVE_CAR_PITCH_OFFSET", -90.0),
            math.degrees(-v.roll))

        # spin the wheel discs about their axle (local X)
        if self.wheel_nodes:
            avg = sum(abs(w.wheel_speed) for w in v.wheels) / max(1, len(v.wheels))
            self.wheel_spin += math.degrees(avg / 0.35) * (1.0 / 60.0)
            for wn in self.wheel_nodes:
                try:
                    disc = wn.getChild(0)
                    disc.setR(-90.0 + self.wheel_spin)
                except Exception:
                    pass

    def _update_chase_cam(self):
        v = self.vehicle
        cy, sy = math.cos(v.yaw), math.sin(v.yaw)
        # panda: forward = (sin(yaw), cos(yaw), 0)
        cam_x = v.pos[0] - math.sin(v.yaw) * CAM_DIST
        cam_y = v.pos[2] - math.cos(v.yaw) * CAM_DIST
        cam_z = v.pos[1] + CAM_HEIGHT
        self.camera.setPos(cam_x, cam_y, cam_z)
        self.camera.lookAt(v.pos[0], v.pos[2], v.pos[1] + 0.8)

    def _update_hud(self):
        v = self.vehicle
        kmh = abs(v.forward_speed) * 3.6
        self.speed_label.setText(f"{kmh:.0f} km/h")
        hb = "HANDBRAKE " if v.handbrake else ""
        self.state_label.setText(
            f"{hb}" + ("DRIVE" if v.forward_speed >= 0 else "REVERSE"))

    def _update_audio(self, dt):
        v = self.vehicle
        st = self.audio_state
        st.speed = v.speed
        st.throttle = v.throttle
        st.handbrake = 1 if self.key_hb else 0
        st.wheelspin = 0
        st.wheel_speeds = [abs(w.wheel_speed) * 100.0 for w in v.wheels]
        st.slip_lateral = max((w.slip_lateral for w in v.wheels), default=0.0)
        st.slip_longitudinal = max((w.slip_longitudinal for w in v.wheels),
                                  default=0.0)
        st.locked = [w.locked if hasattr(w, "locked") else 0 for w in v.wheels]
        try:
            self.audio.update(dt, st)
        except Exception as exc:
            log.debug("[audio] %s", exc)

    def destroy(self):
        try:
            self.audio.shutdown()
        except Exception:
            pass
        super().destroy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    options = {"window_type": "offscreen" if args.offscreen else "onscreen"}
    game = DriveSim(options)
    if args.offscreen:
        for _ in range(max(1, args.frames)):
            game.taskMgr.step()
            game.graphicsEngine.renderFrame()
        log.info("Rendered %d offscreen frames.", max(1, args.frames))
        game.destroy()
        return 0
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
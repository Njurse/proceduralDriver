"""Procedural driving game: REDRIVER2 handling, Panda3D renderer.

Run:  python main.py            (windowed)
      python main.py --offscreen --frames N
      python main.py --shots DIR    (save camera test screenshots, offscreen)

Camera: right stick (or I/J/K/L) swings the camera round the car, release to
return to the chase view; R3 (or O) latches the current angle.
"""

import argparse
import contextlib
import logging
import math
import os
import sys

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    CullFaceAttrib,
    DirectionalLight,
    Filename,
    Fog,
    Mat4,
    NodePath,
    TextureStage,
    TransparencyAttrib,
    Vec4,
    getModelPath,
    loadPrcFileData,
)

import config
from core.car_profile import load_cosmetics
from core.gamepad import Gamepad
from core.orbit_camera import OrbitCamera, orbit_offset
from core.redriver2_physics import Input, PhysicsWorld, Vehicle
from core.textures import _to_texture, seamless_fbm
from core.vehicle_audio import AudioManager, sound_input

log = logging.getLogger("main")

WORLD_SIZE = 400.0
# --shots: (file name, orbit angle in degrees) after a short run-up
SHOTS = (("chase", 0.0), ("left_side", -90.0), ("right_side", 90.0),
         ("front_looking_back", 180.0))

# Physics frame (x, y up, z forward) -> Panda3D (x, y forward, z up).  This
# swap is a reflection; the C physics frame is itself mirrored relative to
# the screen, so steering right turns right on screen.
PHYS_TO_PANDA = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])


def _mat4(a, t=(0.0, 0.0, 0.0)):
    """Column-vector 3x3 transform + translation -> Panda's row-vector Mat4."""
    return Mat4(a[0, 0], a[1, 0], a[2, 0], 0.0,
                a[0, 1], a[1, 1], a[2, 1], 0.0,
                a[0, 2], a[1, 2], a[2, 2], 0.0,
                t[0], t[1], t[2], 1.0)


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class DriveSim(ShowBase):
    def __init__(self, options=None):
        options = dict(options or {})
        ShowBase.__init__(self,
                          windowType=options.get("window_type", "onscreen"))
        self.disableMouse()
        self.camLens.setFov(config.CHASE_CAM_FOV)
        self.clock = ClockObject.getGlobalClock()

        self.cos = load_cosmetics()
        self.vehicle = Vehicle(self.cos)
        self.physics = PhysicsWorld()
        self.physics.add(self.vehicle)
        self.input = Input()
        self.pad = Gamepad()
        self.reverse = False
        self._exited = False
        self.orbit = OrbitCamera()
        self.cam_override = options.get("cam_heading")  # radians, fixed orbit
        if options.get("window_type") == "offscreen":
            # fixed 60 Hz steps so offscreen runs and screenshots repeat
            self.clock.setMode(ClockObject.MNonRealTime)
            self.clock.setFrameRate(60.0)

        self._setup_window()
        self._build_world()
        self._build_car()
        self._build_hud()
        self._bind_keys()
        self._bind_window_events()

        self.audio = AudioManager(self.cos)
        self.spin = 0.0

        self.taskMgr.add(self._update, "drive-update")

    # ------------------------------------------------------------ setup
    def _setup_window(self):
        self.setBackgroundColor(0.35, 0.46, 0.62, 1.0)
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

        # Distance fog hides the edge of the ground quad.
        fog = Fog("drive-fog")
        fog.setColor(0.35, 0.46, 0.62)
        fog.setExpDensity(getattr(config, "DRIVE_FOG_DENSITY", 0.005))
        self.render.setFog(fog)

    def _build_car(self):
        import os
        obj_path = getattr(config, "HAVANA_MODEL_OBJ", None)
        # Make the OBJ's own directory visible to Panda3D's model path so its
        # relative texture references (textures/PAGE_36.png) resolve.
        try:
            getModelPath().appendDirectory(Filename.fromOsSpecific(os.path.dirname(obj_path)))
        except Exception as exc:
            log.debug("[car] model path append failed: %s", exc)
        rel = None
        try:
            rel = os.path.relpath(obj_path, config.ROOT)
        except Exception:
            rel = obj_path
        try:
            # noCache: the OBJs were rescaled in place, so a model-cache entry
            # from the old Driver 2 unit file must not be reused.
            self.car = self.loader.loadModel(Filename.fromOsSpecific(obj_path),
                                             noCache=True)
        except Exception as exc:
            log.warning("[car] load failed: %s", exc)
            self.car = NodePath("fallback_car")
        if self.car.isEmpty():
            self.car = NodePath("fallback_car")

        # The car body node carries the physics pose (see _sync_car); its local
        # frame is the C physics body frame in metres.  The OBJ is in metres
        # too, in Driver 2 model space (y down, nose at -z), which is the
        # physics body frame negated (cars.c UpdateCarDrawMatrix), so a -1
        # scale maps it in.  Both origins are the car's centre of gravity.
        self.car_body = self.render.attachNewNode("car_body")
        self.car.setScale(-1.0)
        # The OBJ faces are wound for Driver 2's y-down screen space, so cull
        # the other side.
        self.car.setAttrib(CullFaceAttrib.makeReverse())
        self.car.reparentTo(self.car_body)

        # Verify the OBJ's texture references resolved.  Panda leaves the mesh
        # untextured when the relative paths are missing, so fall back to a
        # procedural grey instead of rendering a blank/pink car.
        self._verify_model_textures(obj_path)
        try:
            tex_count = len(self.car.findAllTextures())
        except Exception:
            tex_count = 0
        log.info("[car] model=%s textures=%d", rel, tex_count)
        if tex_count == 0:
            log.warning("[car] no textures resolved for %s; using procedural grey",
                        rel)
            self.car.setTexture(self._grey_texture(), 1)

        # Wheel discs, placed every frame where cars.c DrawCarWheels draws them.
        self.wheel_nodes = []
        self.wheel_spin = 0.0
        wheel_tex = None
        try:
            wheel_tex_path = self.cos.wheel_textures.get("clean")
        except Exception:
            wheel_tex_path = None
        if wheel_tex_path:
            try:
                load_path = wheel_tex_path
                with contextlib.suppress(ValueError):
                    load_path = os.path.relpath(wheel_tex_path, config.ROOT)
                wheel_tex = self.loader.loadTexture(load_path)
            except Exception as exc:
                log.debug("[car] wheel texture load failed: %s", exc)
                wheel_tex = None
        diameter = 2.0 * self.vehicle.geo.wheel_radius
        # card local x/z span the disc, its normal (y) lies along the axle (x)
        disc_mat = np.array([[0.0, 1.0, 0.0],
                             [diameter, 0.0, 0.0],
                             [0.0, 0.0, diameter]])
        self.wheel_mounts = []
        for _ in range(4):
            mount = self.car_body.attachNewNode("wheel")
            spinner = mount.attachNewNode("spinner")
            try:
                cm = CardMaker("wheeldisc")
                cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                disc = spinner.attachNewNode(cm.generate())
                disc.setMat(_mat4(disc_mat))
                if wheel_tex is not None:
                    disc.setTexture(wheel_tex, 1)
                disc.setTwoSided(True)
                disc.setTransparency(TransparencyAttrib.MBinary)
            except Exception as exc:
                log.debug("[car] wheel disc build failed: %s", exc)
            self.wheel_mounts.append(mount)
            self.wheel_nodes.append(spinner)
        self.wheel_spins = [0.0] * 4

    def _verify_model_textures(self, obj_path):
        """Log a warning for every texture referenced by the MTL that is missing."""
        mtl_path = os.path.splitext(obj_path)[0] + ".mtl"
        try:
            with open(mtl_path) as handle:
                lines = handle.readlines()
        except Exception as exc:
            log.debug("[car] mtl check failed for %s: %s", mtl_path, exc)
            return
        base = os.path.dirname(obj_path)
        missing = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0] == "map_Kd":
                tex = os.path.normpath(os.path.join(base, parts[1]))
                if not os.path.exists(tex):
                    missing.append(tex)
        if missing:
            log.warning("[car] missing model textures: %s", missing)

    def _grey_texture(self, size=64):
        """Neutral procedural texture used when the model's textures fail."""
        base = np.full((size, size), 128.0)
        speckle = seamless_fbm(size, size, 12.0, 999, 3) * 24.0 - 12.0
        grey = np.clip(base + speckle, 70, 190).astype(np.uint8)
        rgb = np.stack([grey, grey, grey], axis=-1)
        rgba = np.concatenate(
            [rgb, np.full((size, size, 1), 255, np.uint8)], axis=-1)
        return _to_texture(rgba, "car_grey", repeat=True)

    # ------------------------------------------------------------- input
    def _bind_keys(self):
        self.key_throttle = False
        self.key_brake = False
        self.key_left = False
        self.key_right = False
        self.key_hb = False
        self.key_fast = False
        self.key_cam_up = self.key_cam_down = False
        self.key_cam_left = self.key_cam_right = self.key_cam_latch = False
        for name, flag in [("w", "key_throttle"), ("s", "key_brake"),
                           ("a", "key_left"), ("d", "key_right"),
                           ("space", "key_hb"), ("lshift", "key_fast"),
                           ("i", "key_cam_up"), ("k", "key_cam_down"),
                           ("j", "key_cam_left"), ("l", "key_cam_right"),
                           ("o", "key_cam_latch")]:
            self.accept(name, self._set_key, [flag, True])
            self.accept(name + "-up", self._set_key, [flag, False])
        self.accept("escape", self._quit)

    def _bind_window_events(self):
        # Shut the audio down cleanly when the window's X is used.
        self.exitFunc = self._on_exit
        try:
            if self.win is not None:
                self.win.setCloseRequestEvent("window-close")
                self.accept("window-close", self._quit)
        except Exception as exc:
            log.debug("[window] close handler unavailable: %s", exc)

    def _set_key(self, flag, value):
        setattr(self, flag, value)

    def _quit(self):
        self._on_exit()
        self.userExit()

    def _on_exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            self.audio.shutdown()
        except Exception as exc:
            log.debug("[audio] shutdown failed: %s", exc)

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

        # input -> vehicle (keyboard merged with gamepad)
        inp = self.input
        throttle = 1.0 if self.key_throttle else 0.0
        brake = 1.0 if self.key_brake else 0.0
        steer = (-1.0 if self.key_left else 0.0) + \
                (1.0 if self.key_right else 0.0)
        handbrake = self.key_hb
        fast = self.key_fast
        self.pad.poll()
        if self.pad.connected:
            throttle = max(throttle, self.pad.right_trigger)
            brake = max(brake, self.pad.left_trigger)
            steer = max(-1.0, min(1.0, steer + self.pad.left_x))
            handbrake = bool(handbrake or self.pad.a_button)
            fast = bool(fast or self.pad.right_shoulder)

        v = self.vehicle
        # No separate reverse gear: the C brake thrust reverses the car once
        # it is nearly stopped (ProcessCarPad).
        inp.throttle = throttle
        inp.brake = brake
        inp.steer = steer
        inp.handbrake = handbrake
        inp.fast_steer = fast
        inp.analog = self.pad.connected and abs(self.pad.left_x) > 0.0
        v.process_input(inp)
        self.physics.step(dt)
        self._clamp_to_world()

        self._sync_car(dt)
        self._update_chase_cam(dt)
        self._update_hud()
        self._update_audio(dt)
        return task.cont

    def _clamp_to_world(self):
        """Keep the car on the ground quad, bouncing it back inward."""
        bound = WORLD_SIZE / 2.0 - getattr(config, "DRIVE_BOUNDARY_MARGIN", 12.0)
        v = self.vehicle
        for pos_axis, vel_axis in ((0, 0), (2, 2)):
            if v.pos[pos_axis] > bound:
                v.pos[pos_axis] = bound
                if v.vel[vel_axis] > 0.0:
                    v.vel[vel_axis] *= -0.25
            elif v.pos[pos_axis] < -bound:
                v.pos[pos_axis] = -bound
                if v.vel[vel_axis] < 0.0:
                    v.vel[vel_axis] *= -0.25

    def _sync_car(self, dt):
        v = self.vehicle
        self.car_body.setMat(_mat4(PHYS_TO_PANDA @ v.rot, PHYS_TO_PANDA @ v.pos))

        for i, (mount, spinner) in enumerate(zip(self.wheel_mounts, self.wheel_nodes,
                                                 strict=True)):
            steer = v.wheel_angle if i in (0, 2) else 0.0
            mount.setMat(_mat4(_rot_y(steer), v.wheel_visual_offset(i)))
            self.wheel_spins[i] = (self.wheel_spins[i] + v.wheels[i].wheel_speed * dt) % math.tau
            spinner.setMat(_mat4(_rot_x(self.wheel_spins[i])))

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
        ox, oy, oz = orbit_offset(angle, config.CHASE_CAM_DIST, config.CHASE_CAM_HEIGHT)
        # car yaw frame -> physics world: right (cos h, 0, -sin h), fwd (sin h, 0, cos h)
        offset = np.array([ox * math.cos(h) + oz * math.sin(h), oy,
                           -ox * math.sin(h) + oz * math.cos(h)])
        self.camera.setPos(*(PHYS_TO_PANDA @ (v.pos + offset)))
        # aim ahead of the car in the chase view, at the car from the side/front
        ahead = config.CHASE_CAM_LOOK_AHEAD * max(0.0, math.cos(angle))
        aim = v.pos + np.array([math.sin(h) * ahead, config.CHASE_CAM_LOOK_HEIGHT,
                                math.cos(h) * ahead])
        self.camera.lookAt(*(PHYS_TO_PANDA @ aim))

    def _update_hud(self):
        v = self.vehicle
        kmh = abs(v.forward_speed) * 3.6
        self.speed_label.setText(f"{kmh:.0f} km/h")
        hb = "HANDBRAKE " if v.handbrake else ""
        self.state_label.setText(
            f"{hb}" + ("DRIVE" if v.forward_speed >= 0 else "REVERSE"))

    def _update_audio(self, dt):
        try:
            self.audio.update(dt, sound_input(self.vehicle))
        except Exception as exc:
            log.debug("[audio] %s", exc)

    def destroy(self):
        with contextlib.suppress(Exception):
            self.audio.shutdown()
        super().destroy()


def _save_shot(game, path):
    game.graphicsEngine.renderFrame()
    if not game.win.saveScreenshot(Filename.fromOsSpecific(path)):
        raise RuntimeError(f"screenshot failed: {path}")
    log.info("Saved %s", path)


def take_shots(game, out_dir):
    """Run up to speed, then save one shot per SHOTS angle and a chase shot
    mid-corner."""
    os.makedirs(out_dir, exist_ok=True)

    def run(frames):
        for _ in range(frames):
            game.taskMgr.step()

    game.key_throttle = True
    run(150)  # 2.5 s of gas, ~80 km/h
    for name, deg in SHOTS:
        game.cam_override = math.radians(deg)
        run(3)
        _save_shot(game, os.path.join(out_dir, f"{name}.png"))
    game.cam_override = None
    game.orbit = OrbitCamera()
    game.key_right = True
    run(40)
    _save_shot(game, os.path.join(out_dir, "corner_chase.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--cam-heading", type=float, default=None, metavar="DEG",
                    help="fix the camera orbit angle (0 chase, -90 left, 90 right, 180 front)")
    ap.add_argument("--shots", metavar="DIR",
                    help="save camera test screenshots to DIR (implies --offscreen)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

    if args.shots:
        args.offscreen = True
        loadPrcFileData("", "win-size 960 540")
    options = {"window_type": "offscreen" if args.offscreen else "onscreen"}
    if args.cam_heading is not None:
        options["cam_heading"] = math.radians(args.cam_heading)
    game = DriveSim(options)
    if args.shots:
        take_shots(game, args.shots)
        game.destroy()
        return 0
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

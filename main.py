"""Procedural driving game: REDRIVER2 handling on endless Perlin hills, Panda3D renderer.

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
import time

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    Filename,
    Fog,
    Point3,
    TextNode,
    Vec3,
    Vec4,
    loadPrcFileData,
)

import config
from core.gamepad import Gamepad
from core.orbit_camera import OrbitCamera
from core.redriver2_physics import Input
from core.textures import NEAREST_PRC, use_nearest
from core.vehicle_audio import AudioManager, sound_input
from game.car_view import PHYS_TO_PANDA, CarView
from game.chase_camera import CAMERA_NEAR_M, ChaseCamera
from game.controls import Controls, _NoPad
from ui.minimap import Minimap
from ui.pause import APPLY_SEED, CONTINUE, EXIT, EXPORT, RESTART, PauseMenu
from world.chunks import ChunkStreamer, detail_texture, road_detail_texture
from world.session import Session
from world.sky import build_sky
from world.worldmap import export_world

log = logging.getLogger("main")

# --shots: (file name, orbit angle in degrees) after a short run-up
SHOTS = (("world_chase", 0.0), ("world_left_side", -90.0), ("world_right_side", 90.0),
         ("world_front_looking_back", 180.0))


class DriveSim(Controls, ChaseCamera, CarView, ShowBase):
    def __init__(self, options=None):
        options = dict(options or {})
        loadPrcFileData("nearest", NEAREST_PRC)  # textures loaded from disk
        ShowBase.__init__(self,
                          windowType=options.get("window_type", "onscreen"))
        self.disableMouse()
        self.camLens.setFov(config.CHASE_CAM_FOV)
        self.camLens.setNearFar(CAMERA_NEAR_M, config.DRAW_DISTANCE_M)
        self.clock = ClockObject.getGlobalClock()

        # world, car, discovery and the save file (world/session.py)
        self.session = Session(save_path=options.get("save_path"), seed=options.get("seed"),
                               new=options.get("new", False))
        self.cos = self.session.cos
        self.input = Input()
        self._pad_prev = {}
        self._stick_prev = 0
        offscreen = options.get("window_type") == "offscreen"
        # XInput reads the controller whether or not the window has focus, so
        # offscreen runs ignore it: a held trigger made --shots runs differ.
        self.pad = _NoPad() if offscreen else Gamepad()
        self.reverse = False
        self._exited = False
        self.orbit = OrbitCamera()
        self.cam_override = options.get("cam_heading")  # radians, fixed orbit
        self.autopilot = None  # f(vehicle) -> steer, replaces the keys (--shots)
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
        # smoothed body motion the chase camera hangs back from (see config)
        self.cam_vert = 0.0
        self.cam_side = 0.0

        self.taskMgr.add(self._update, "drive-update")

    # The session swaps the world on a seed change, so these always follow it.
    @property
    def vehicle(self):
        return self.session.vehicle

    @property
    def physics(self):
        return self.session.physics

    @property
    def terrain(self):
        return self.session.world

    @property
    def paused(self):
        return self.session.paused

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
        # Endless terrain: chunk meshes streamed round the car (world/chunks.py),
        # vertex-coloured by height and slope over a tiled detail texture.
        self.chunks = ChunkStreamer(self.terrain, self.render,
                                    detail_texture(self.terrain.detail_texture),
                                    road_texture=road_detail_texture())
        v = self.vehicle
        self.chunks.update(v.pos[0], v.pos[2], budget=(2 * self.chunks.radius + 1) ** 2)

        amb = AmbientLight("amb")
        # ambient + sun at most 1.0, so lit colours never pass their texture
        amb.setColor(Vec4(0.42, 0.42, 0.46, 1.0))
        self.render.setLight(self.render.attachNewNode(amb))
        dl = DirectionalLight("sun")
        dl.setColor(Vec4(0.58, 0.56, 0.52, 1.0))
        dl_node = self.render.attachNewNode(dl)
        dl_node.setHpr(45, -35, 0)
        self.render.setLight(dl_node)

        # Sky dome round the camera; the fog and the clear colour take the
        # panorama's horizon haze so distant ground fades into it.
        self.sky, horizon = build_sky(config.SKYBOX_PNG, self.camera, self.render,
                                      config.SKY_RADIUS_M)
        self.setBackgroundColor(*horizon, 1.0)

        # Exp-squared fog: clear near the car, thick before the streaming edge.
        self.fog = Fog("drive-fog")
        self.fog.setMode(Fog.MExponentialSquared)
        self.fog.setColor(*horizon)
        self.fog.setExpDensity(config.DRIVE_FOG_DENSITY)
        self.render.setFog(self.fog)

    # ------------------------------------------------------------- pause
    def set_paused(self, paused):
        """Freeze physics and the engine sound and show the menu (saving)."""
        if paused == self.paused:
            return
        self.session.set_paused(paused)
        self.pause_menu.set_visible(paused)
        with contextlib.suppress(Exception):
            self.audio.set_paused(paused)

    def menu_action(self, action):
        """Carry out what the pause menu chose."""
        log.info("[menu] %s", action)
        if action == CONTINUE:
            self.set_paused(False)
        elif action == RESTART:
            self.session.restart()
            self._after_respawn()
            self.set_paused(False)
        elif action == EXPORT:
            try:
                path, (w, h) = export_world(self.terrain, self.session.explorer.discovered)
                self.pause_menu.set_status(f"Saved {path} ({w} x {h})")
                log.info("[export] %s (%d x %d)", path, w, h)
            except Exception as exc:
                self.pause_menu.set_status(f"Export failed: {exc}")
                log.warning("[export] %s", exc)
        elif action == EXIT:
            self._quit()
        elif isinstance(action, tuple) and action[0] == APPLY_SEED:
            self.session.change_seed(action[1])
            self.chunks.reset(self.terrain)
            self.minimap.set_world(self.terrain)
            self._after_respawn()
            self.pause_menu.set_status(f"New world: seed {action[1]}")

    def _after_respawn(self):
        v = self.vehicle
        self.chunks.update(v.pos[0], v.pos[2], budget=(2 * self.chunks.radius + 1) ** 2)
        self.orbit = OrbitCamera()
        self.cam_vert = self.cam_side = 0.0
        self._sync_car(0.0)
        self._update_chase_cam(1.0 / 60.0)

    def _quit(self):
        self._on_exit()
        self.userExit()

    def _on_exit(self):
        if self._exited:
            return
        self._exited = True
        try:
            self.session.save()
        except Exception as exc:
            log.warning("[save] %s", exc)
        try:
            self.audio.shutdown()
        except Exception as exc:
            log.debug("[audio] shutdown failed: %s", exc)

    # ------------------------------------------------------------- hud
    def _build_hud(self):
        use_nearest(TextNode.getDefaultFont())
        self.speed_label = OnscreenText(
            text="0 km/h", pos=(-0.72, -0.82), scale=0.12,
            fg=(1, 1, 1, 1), align=2, shadow=(0, 0, 0, 0.6))
        self.state_label = OnscreenText(
            text="", pos=(-0.72, -0.93), scale=0.07,
            fg=(1, 1, 1, 1), align=2, shadow=(0, 0, 0, 0.6))
        self.minimap = Minimap(self, self.terrain)
        self.pause_menu = PauseMenu(self, lambda: self.session.seed,
                                    lambda: len(self.session.explorer.discovered))

    # ------------------------------------------------------------- frame
    def _watch(self, v, spent):
        """Log biome changes, and frames whose physics + chunks ran long
        (at most one warning per 5 s)."""
        biome = self.terrain.regions.biome_at(v.pos[0], v.pos[2]).name
        if biome != getattr(self, "_biome", None):
            log.info("[world] entering %s at (%.0f, %.0f)", biome, v.pos[0], v.pos[2])
            self._biome = biome
        now = time.perf_counter()
        if spent > 1.0 / 30.0 and now - getattr(self, "_slow_at", -10.0) > 5.0:
            log.warning("slow frame: physics + chunks took %.0f ms", spent * 1000.0)
            self._slow_at = now

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
        self._pad_menu()
        if self.paused:
            return task.cont
        if self.pad.connected:
            throttle = max(throttle, self.pad.right_trigger)
            brake = max(brake, self.pad.left_trigger)
            steer = max(-1.0, min(1.0, steer + self.pad.left_x))
            handbrake = bool(handbrake or self.pad.a_button)
            fast = bool(fast or self.pad.right_shoulder)

        v = self.vehicle
        if self.autopilot is not None:
            steer = self.autopilot(v)
        # No separate reverse gear: the C brake thrust reverses the car once
        # it is nearly stopped (ProcessCarPad).
        inp.throttle = throttle
        inp.brake = brake
        inp.steer = steer
        inp.handbrake = handbrake
        inp.fast_steer = fast
        inp.analog = self.pad.connected and abs(self.pad.left_x) > 0.0
        t0 = time.perf_counter()
        self.session.step(inp, dt)
        self.chunks.update(v.pos[0], v.pos[2])
        self._watch(v, time.perf_counter() - t0)

        self._sync_car(dt)
        self._update_chase_cam(dt)
        self._update_hud()
        self._update_audio(dt)
        self.minimap.update((v.pos[0], v.pos[2]), v.heading,
                            self.session.explorer.discovered, self.chunks.loaded)
        return task.cont

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
        self._on_exit()   # saves (when saving is on) and stops the audio
        super().destroy()


def _save_shot(game, path):
    game.graphicsEngine.renderFrame()
    if not game.win.saveScreenshot(Filename.fromOsSpecific(path)):
        raise RuntimeError(f"screenshot failed: {path}")
    log.info("Saved %s", path)


def take_shots(game, out_dir):
    """Run up to speed, then save one shot per SHOTS angle, a chase shot
    mid-corner and a wide shot over the hills and loaded chunks."""
    os.makedirs(out_dir, exist_ok=True)

    def run(frames):
        for _ in range(frames):
            game.taskMgr.step()

    game.key_throttle = True
    t = game.terrain
    game.autopilot = lambda v: t.road_steer(v.pos[0], v.pos[2], v.heading)
    run(150)  # 2.5 s of gas down the road, ~80 km/h
    _save_shot(game, os.path.join(out_dir, "road_chase.png"))
    _save_shot(game, os.path.join(out_dir, "minimap_chase.png"))
    sky_chase(game)
    _save_shot(game, os.path.join(out_dir, "sky_chase.png"))
    for name, deg in SHOTS:
        game.cam_override = math.radians(deg)
        run(3)
        _save_shot(game, os.path.join(out_dir, f"{name}.png"))
    game.cam_override = None
    game.orbit = OrbitCamera()
    game.autopilot = None
    game.key_right = True
    run(40)
    _save_shot(game, os.path.join(out_dir, "world_corner_chase.png"))

    # wide: high behind the car, thin fog so the chunk edge shows
    v = game.vehicle
    h = v.heading
    eye = v.pos + np.array([-math.sin(h) * 110.0, 90.0, -math.cos(h) * 110.0])
    game.camera.setPos(*(PHYS_TO_PANDA @ eye))
    game.camera.lookAt(*(PHYS_TO_PANDA @ (v.pos + np.array([math.sin(h), 0, math.cos(h)]) * 60.0)))
    game.fog.setExpDensity(config.DRIVE_FOG_DENSITY * 0.15)
    _save_shot(game, os.path.join(out_dir, "world_hills.png"))
    game.fog.setExpDensity(config.DRIVE_FOG_DENSITY)

    # close-up of the front wheel, from outside and a little ahead of it
    wheel = game.wheel_mounts[0].getPos(game.render)
    side = math.copysign(1.0, game.vehicle.wheel_visual_offset(0)[0])
    right = game.render.getRelativeVector(game.car_body, Vec3(side, 0, 0))
    ahead = game.render.getRelativeVector(game.car_body, Vec3(0, 0, 1))
    up = Vec3(0, 0, 1)
    game.camera.setPos(wheel + right * 1.3 + ahead * 0.9 + up * 0.25)
    game.camera.lookAt(wheel)
    _save_shot(game, os.path.join(out_dir, "world_wheel_closeup.png"))

    biome_border(game)
    _save_shot(game, os.path.join(out_dir, "biome_border.png"))

    # the pause menu over the chase view, then the world export
    game.camLens.setNearFar(CAMERA_NEAR_M, config.DRAW_DISTANCE_M)
    game._update_chase_cam(1.0 / 60.0)
    game.set_paused(True)
    run(3)
    _save_shot(game, os.path.join(out_dir, "pause_menu.png"))
    game.minimap.toggle_zoom()
    game.minimap.update((game.vehicle.pos[0], game.vehicle.pos[2]), game.vehicle.heading,
                        game.session.explorer.discovered, game.chunks.loaded, force=True)
    game.pause_menu.set_visible(False)
    _save_shot(game, os.path.join(out_dir, "minimap_far.png"))
    path, size = export_world(game.terrain, game.session.explorer.discovered,
                              os.path.join(out_dir, "world_export.png"))
    log.info("[shots] exported %s %s", path, size)
    game.set_paused(False)
    game.minimap.toggle_zoom()
    overhead(game)
    _save_shot(game, os.path.join(out_dir, "roads_overview.png"))
    biome_shots(game, out_dir)


def biome_shots(game, out_dir, frames=90):
    """A chase shot down the road in each biome past plains and desert
    (biome_<name>.png), then a high overhead and a big export over them."""
    world = game.terrain
    game.chunks.radius = config.TERRAIN_VIEW_CHUNKS
    game.camLens.setNearFar(CAMERA_NEAR_M, config.DRAW_DISTANCE_M)
    game.fog.setExpDensity(config.DRIVE_FOG_DENSITY)
    game.key_right = False
    game.key_throttle = True
    game.autopilot = lambda v: world.road_steer(v.pos[0], v.pos[2], v.heading)
    for name in config.WORLD_BIOMES[2:]:
        game.vehicle.reset(*world.spawn(*world.regions.site_of(name)), ground=world)
        game.session.physics._accumulator = 0.0
        game._after_respawn()
        for _ in range(frames):
            game.taskMgr.step()
        _save_shot(game, os.path.join(out_dir, f"biome_{name}.png"))
    game.autopilot = None
    game.key_throttle = False

    overhead(game, height=1500.0, radius=14)
    _save_shot(game, os.path.join(out_dir, "roads_overview_biomes.png"))
    # 64 x 64 chunks (4 km) round the car at 4 m a pixel, whatever was driven
    v = game.vehicle
    ci, ck = int(v.pos[0] // config.TERRAIN_CHUNK_SIZE), int(v.pos[2] // config.TERRAIN_CHUNK_SIZE)
    square = {(ci + i, ck + k) for i in range(-32, 32) for k in range(-32, 32)}
    path, size = export_world(world, square, os.path.join(out_dir, "world_export_biomes.png"),
                              m_per_px=4.0)
    log.info("[shots] exported %s %s", path, size)


def sky_chase(game):
    """Camera low behind the car, tilted up at the sky."""
    v = game.vehicle
    fwd = np.array([math.sin(v.heading), 0.0, math.cos(v.heading)])
    up = np.array([0.0, 1.0, 0.0])
    game.camera.setPos(*(PHYS_TO_PANDA @ (v.pos - fwd * 7.0 + up * 1.4)))
    game.camera.lookAt(*(PHYS_TO_PANDA @ (v.pos + fwd * 30.0 + up * 9.0)))


def biome_border(game, max_radius=600.0):
    """Camera up high, looking along the biome border nearest the car."""
    world, v = game.terrain, game.vehicle
    best = None
    for r in np.arange(40.0, max_radius, 20.0):
        a = np.linspace(0.0, 2.0 * math.pi, int(r / 4), endpoint=False)
        x, z = v.pos[0] + r * np.sin(a), v.pos[2] + r * np.cos(a)
        k = int(np.argmin(world.border_weight(x, z)))
        if world.border_weight(x[k], z[k]) < 0.55:
            best = np.array([x[k], world.height(x[k], z[k]), z[k]])
            break
    if best is None:
        log.warning("[shots] no biome border within %.0f m", max_radius)
        best = v.pos.copy()
    w, gx, gz = world.regions.weights(best[0], best[2])
    b = int(np.argmax(w[0]))
    wx, wz = gx[0, b], gz[0, b]
    across = np.array([wx, 0.0, wz]) / max(math.hypot(wx, wz), 1e-9)
    along = np.array([-across[2], 0.0, across[0]])
    eye = best - along * 150.0 + np.array([0.0, 45.0, 0.0])
    game.chunks.update(best[0], best[2], budget=(2 * game.chunks.radius + 1) ** 2)
    game.camera.setPos(*(PHYS_TO_PANDA @ eye))
    game.camera.lookAt(*(PHYS_TO_PANDA @ (best + along * 40.0)))
    game.fog.setExpDensity(config.DRIVE_FOG_DENSITY * 0.3)


def overhead(game, height=750.0, radius=9):
    """Look straight down on the roads round the car: far plane out, fog
    off and a wider ring of chunks for this one frame."""
    v = game.vehicle
    game.camLens.setNearFar(CAMERA_NEAR_M, height + 200.0)
    game.fog.setExpDensity(0.0)
    game.chunks.radius = radius
    game.chunks.update(v.pos[0], v.pos[2], budget=(2 * radius + 1) ** 2)
    game.camera.setPos(*(PHYS_TO_PANDA @ (v.pos + np.array([0.0, height, 0.0]))))
    game.camera.lookAt(Point3(*(PHYS_TO_PANDA @ v.pos)), Vec3(0, 1, 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offscreen", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--cam-heading", type=float, default=None, metavar="DEG",
                    help="fix the camera orbit angle (0 chase, -90 left, 90 right, 180 front)")
    ap.add_argument("--shots", metavar="DIR",
                    help="save camera test screenshots to DIR (implies --offscreen)")
    ap.add_argument("--seed", type=int, default=None,
                    help="world seed (a save for another seed is left alone)")
    ap.add_argument("--new", action="store_true",
                    help="ignore the save and start a fresh world (random seed unless --seed)")
    ap.add_argument("--save", metavar="PATH", default=None,
                    help=f"save file (default {config.SAVE_PATH}; offscreen runs save "
                         "only when this is given)")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("-v", "--verbose", action="store_const", const="DEBUG", dest="log_level",
                    help="per-chunk detail (DEBUG)")
    ap.add_argument("-q", "--quiet", action="store_const", const="WARNING", dest="log_level")
    ap.add_argument("--log-file", nargs="?", const="logs/drive.log", metavar="PATH",
                    help="also log to PATH (default logs/drive.log)")
    args = ap.parse_args()

    handlers = [logging.StreamHandler()]
    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(level=args.log_level, handlers=handlers, datefmt="%H:%M:%S",
                        format="%(asctime)s.%(msecs)03d %(levelname)-5s %(name)s: %(message)s")
    log.info("[config] chunk %.0f m, view %d chunks, %d builds/frame, biomes: %s",
             config.TERRAIN_CHUNK_SIZE, config.TERRAIN_VIEW_CHUNKS,
             config.TERRAIN_BUILDS_PER_FRAME, ", ".join(config.WORLD_BIOMES))

    if args.shots:
        args.offscreen = True
        loadPrcFileData("", "win-size 960 540")
    options = {"window_type": "offscreen" if args.offscreen else "onscreen",
               "seed": args.seed, "new": args.new,
               "save_path": args.save or (None if args.offscreen else config.SAVE_PATH)}
    if args.cam_heading is not None:
        options["cam_heading"] = math.radians(args.cam_heading)
    game = DriveSim(options)
    if args.shots:
        take_shots(game, args.shots)
        game.destroy()
        return 0
    if args.offscreen:
        t0 = time.perf_counter()
        for _ in range(max(1, args.frames)):
            game.taskMgr.step()
            game.graphicsEngine.renderFrame()
        spent = time.perf_counter() - t0
        log.info("Rendered %d offscreen frames in %.1f s (%.1f ms a frame, %.0f fps).",
                 max(1, args.frames), spent, spent * 1000.0 / max(1, args.frames),
                 max(1, args.frames) / spent)
        game.destroy()
        return 0
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

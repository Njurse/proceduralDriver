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

import numpy as np
from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    CardMaker,
    ClockObject,
    ColorAttrib,
    CullFaceAttrib,
    DirectionalLight,
    Filename,
    Fog,
    Geom,
    GeomNode,
    GeomTriangles,
    GeomVertexData,
    GeomVertexFormat,
    GeomVertexWriter,
    Mat4,
    MaterialAttrib,
    NodePath,
    Point3,
    TextNode,
    TextureAttrib,
    TransparencyAttrib,
    Vec3,
    Vec4,
    getModelPath,
    loadPrcFileData,
)

import config
from core.gamepad import Gamepad
from core.orbit_camera import OrbitCamera, orbit_offset
from core.redriver2_physics import Input
from core.textures import NEAREST_PRC, _to_texture, load_color_keyed, seamless_fbm, use_nearest
from core.units import DR2_UNITS_PER_METRE
from core.vehicle_audio import AudioManager, sound_input
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
CAM_TERRAIN_CLEARANCE = 0.6  # keep the chase camera this far above the hills
CAMERA_NEAR_M = 0.2          # close enough for the wheel close-up
MAX_CAR_EXTENT_M = 20.0      # a longer model is in Driver 2 units, not metres
TYRE_WIDTH_M = 0.14          # gap between a wheel's outer and inner planes
TYRE_BAND_RADIUS = 0.94      # tread diamond corners, fraction of the wheel radius

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


def _flat_colours_from_materials(model):
    """Give untextured faces (wheel wells, the "flat" material) their material
    diffuse as a flat colour; unlit, they would otherwise draw white."""
    for path in model.findAllMatches("**/+GeomNode"):
        node = path.node()
        for i in range(node.getNumGeoms()):
            state = node.getGeomState(i)
            if state.hasAttrib(TextureAttrib) or not state.hasAttrib(MaterialAttrib):
                continue
            material = state.getAttrib(MaterialAttrib).getMaterial()
            if material is None or not material.hasDiffuse():
                continue
            colour = material.getDiffuse()
            node.setGeomState(i, state.setAttrib(ColorAttrib.makeFlat(colour)))


class _NoPad:
    """Stand-in Gamepad that is never connected."""

    connected = False

    def poll(self):
        pass


def _model_unit_scale(model, name):
    """Metres per model unit: 1, or 1/150 for an OBJ still in Driver 2 units.

    A car exported straight from Driver 2 units (or re-exported from Blender
    without the /150) is ~750 m long: the chase camera then sits inside it
    and every face is culled, so the car looks invisible.
    """
    bounds = model.getTightBounds()
    if not bounds:
        return 1.0
    extent = max(bounds[1] - bounds[0])
    if extent <= MAX_CAR_EXTENT_M:
        return 1.0
    log.warning("[car] %s is %.0f units long: treating it as Driver 2 units (/%d)",
                name, extent, DR2_UNITS_PER_METRE)
    return 1.0 / DR2_UNITS_PER_METRE


def _tyre_band(radius, width):
    """Black tread: a diamond (square on its corner) of four quads round the
    axle (x), from the outer plane at x = 0 to the inner one at x = width."""
    vdata = GeomVertexData("tyre", GeomVertexFormat.getV3n3(), Geom.UH_static)
    vertex = GeomVertexWriter(vdata, "vertex")
    normal = GeomVertexWriter(vdata, "normal")
    corners = [(radius * math.cos(a), radius * math.sin(a))
               for a in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)]
    prim = GeomTriangles(Geom.UH_static)
    for k in range(4):
        (y0, z0), (y1, z1) = corners[k], corners[(k + 1) % 4]
        ny, nz = (y0 + y1) / 2.0, (z0 + z1) / 2.0
        length = math.hypot(ny, nz)
        for x, y, z in ((0.0, y0, z0), (0.0, y1, z1), (width, y1, z1), (width, y0, z0)):
            vertex.addData3(x, y, z)
            normal.addData3(0.0, ny / length, nz / length)
        prim.addVertices(4 * k, 4 * k + 1, 4 * k + 2)
        prim.addVertices(4 * k, 4 * k + 2, 4 * k + 3)
    geom = Geom(vdata)
    geom.addPrimitive(prim)
    node = GeomNode("tyre_band")
    node.addGeom(geom)
    return node


def _rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class DriveSim(ShowBase):
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
        # Unlit like the PlayStation: the car shows its texture colours.  Lit,
        # an OBJ material with Ka 1 (Blender's default) summed ambient + sun
        # past 1.0 and washed the paint out to white.
        self.car_body.setLightOff(1)
        _flat_colours_from_materials(self.car)
        self.car.setScale(-_model_unit_scale(self.car, rel))
        self.car.setPos(*self.vehicle.geo.model_offset)
        # The OBJ faces are wound for Driver 2's y-down screen space, so cull
        # the other side.
        self.car.setAttrib(CullFaceAttrib.makeReverse())
        self.car.reparentTo(self.car_body)

        # Verify the OBJ's texture references resolved.  Panda leaves the mesh
        # untextured when the relative paths are missing, so fall back to a
        # procedural grey instead of rendering a blank/pink car.
        self._verify_model_textures(obj_path)
        car_textures = self.car.findAllTextures()
        for tex in car_textures:
            use_nearest(tex)
        tex_count = len(car_textures)
        log.info("[car] model=%s textures=%d", rel, tex_count)
        if tex_count == 0:
            log.warning("[car] no textures resolved for %s; using procedural grey",
                        rel)
            self.car.setTexture(self._grey_texture(), 1)

        # Wheels, placed every frame where cars.c DrawCarWheels draws them: an
        # outer and an inner textured plane facing the same way (the texture's
        # white background keyed out), joined by a black diamond tread band.
        self.wheel_nodes = []
        wheel_tex = None
        wheel_tex_path = self.cos.wheel_textures.get("clean")
        if wheel_tex_path and os.path.exists(wheel_tex_path):
            wheel_tex = load_color_keyed(wheel_tex_path, "wheel_clean")
        else:
            log.warning("[car] wheel texture missing: %s", wheel_tex_path)
        radius = self.vehicle.geo.wheel_radius
        # card local x/z span the plane, its normal (y) lies along the axle (x)
        disc_mat = np.array([[0.0, 1.0, 0.0],
                             [2.0 * radius, 0.0, 0.0],
                             [0.0, 0.0, 2.0 * radius]])
        self.wheel_mounts = []
        for i in range(4):
            mount = self.car_body.attachNewNode("wheel")
            spinner = mount.attachNewNode("spinner")
            inboard = -math.copysign(TYRE_WIDTH_M, self.vehicle.wheel_visual_offset(i)[0])
            for name, x in (("outer", 0.0), ("inner", inboard)):
                cm = CardMaker(f"wheel_{name}")
                cm.setFrame(-0.5, 0.5, -0.5, 0.5)
                plane = spinner.attachNewNode(cm.generate())
                plane.setMat(_mat4(disc_mat, (x, 0.0, 0.0)))
                if wheel_tex is not None:
                    plane.setTexture(wheel_tex, 1)
                plane.setTwoSided(True)
                plane.setTransparency(TransparencyAttrib.MBinary)
            band = spinner.attachNewNode(_tyre_band(radius * TYRE_BAND_RADIUS, inboard))
            band.setTwoSided(True)
            band.setTextureOff(1)
            band.setColor(0.04, 0.04, 0.04, 1.0, 1)
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
        self.accept("escape", self._escape)
        for key, fn in [("arrow_up", lambda: self._menu_move(-1)),
                        ("arrow_down", lambda: self._menu_move(1)),
                        ("arrow_left", lambda: self._menu_adjust(-1)),
                        ("arrow_right", lambda: self._menu_adjust(1)),
                        ("enter", self._menu_activate), ("backspace", self._menu_backspace),
                        ("m", self.minimap.toggle_zoom)]:
            self.accept(key, fn)
        for d in "0123456789":
            self.accept(d, self._menu_type, [d])

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
        if value and self.paused:   # W/S/A/D/space work the menu while paused
            {"key_throttle": lambda: self._menu_move(-1),
             "key_brake": lambda: self._menu_move(1),
             "key_left": lambda: self._menu_adjust(-1),
             "key_right": lambda: self._menu_adjust(1),
             "key_hb": self._menu_activate}.get(flag, lambda: None)()

    # ------------------------------------------------------------- pause
    def set_paused(self, paused):
        """Freeze physics and the engine sound and show the menu (saving)."""
        if paused == self.paused:
            return
        self.session.set_paused(paused)
        self.pause_menu.set_visible(paused)
        with contextlib.suppress(Exception):
            self.audio.set_paused(paused)

    def _escape(self):
        if not self.paused:
            self.set_paused(True)
        elif self.pause_menu.back():
            self.set_paused(False)

    def _menu_move(self, d):
        if self.paused:
            self.pause_menu.move(d)

    def _menu_adjust(self, d):
        if self.paused:
            self.pause_menu.adjust(d)

    def _menu_type(self, ch):
        if self.paused:
            self.pause_menu.type_char(ch)

    def _menu_backspace(self):
        if self.paused:
            self.pause_menu.backspace()

    def _menu_activate(self):
        if self.paused:
            self.menu_action(self.pause_menu.activate())

    def menu_action(self, action):
        """Carry out what the pause menu chose."""
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

    def _pad_menu(self):
        """Gamepad edges: Start pauses, Back zooms the map; in the menu the
        d-pad / left stick move, A selects, B goes back."""
        pad = self.pad
        if not pad.connected:
            return
        now = {"start": pad.start_button, "back": pad.map_button, "a": pad.a_button,
               "b": pad.brake, "up": pad.dpad_up, "down": pad.dpad_down,
               "left": pad.dpad_left, "right": pad.dpad_right}
        hit = {k for k, v in now.items() if v and not self._pad_prev.get(k)}
        self._pad_prev = now
        # XInput reports the stick up-positive, SDL down-positive
        y = pad.left_y * (-1.0 if pad.backend == "xinput" else 1.0)
        stick = (y > 0.6) - (y < -0.6)
        if stick and stick != self._stick_prev:
            hit.add("down" if stick > 0 else "up")
        self._stick_prev = stick
        if "back" in hit:
            self.minimap.toggle_zoom()
        if "start" in hit:
            self.set_paused(not self.paused)
            return
        if not self.paused:
            return
        if "up" in hit:
            self.pause_menu.move(-1)
        if "down" in hit:
            self.pause_menu.move(1)
        if "left" in hit:
            self.pause_menu.adjust(-1)
        if "right" in hit:
            self.pause_menu.adjust(1)
        if "a" in hit:
            self.menu_action(self.pause_menu.activate())
        elif "b" in hit and self.pause_menu.back():
            self.set_paused(False)

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
        self.session.step(inp, dt)
        self.chunks.update(v.pos[0], v.pos[2])

        self._sync_car(dt)
        self._update_chase_cam(dt)
        self._update_hud()
        self._update_audio(dt)
        self.minimap.update((v.pos[0], v.pos[2]), v.heading,
                            self.session.explorer.discovered, self.chunks.loaded)
        return task.cont

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
        # car yaw frame -> physics world: right (cos h, 0, -sin h), fwd (sin h, 0, cos h)
        right = np.array([math.cos(h), 0.0, -math.sin(h)])
        fwd = np.array([math.sin(h), 0.0, math.cos(h)])
        ox, oy, oz = orbit_offset(angle, config.CHASE_CAM_DIST, config.CHASE_CAM_HEIGHT)
        offset = right * ox + fwd * oz
        offset[1] = oy
        # Rough ground and slides: the camera hangs back from the body's
        # vertical and sideways motion instead of being welded to it, so
        # bumps bob the view and drifts sway it.  First-order and clamped.
        k = 1.0 - math.exp(-dt / config.CHASE_CAM_BOB_TAU)
        kick = config.CHASE_CAM_KICK_MAX
        self.cam_vert += (v.vel[1] - self.cam_vert) * k
        self.cam_side += (float(v.vel @ right) - self.cam_side) * k
        offset[1] -= max(-kick, min(kick, self.cam_vert)) * config.CHASE_CAM_BOB
        offset -= right * (max(-kick, min(kick, self.cam_side)) * config.CHASE_CAM_SWAY)
        eye = v.pos + offset
        eye[1] = max(eye[1], self.terrain.height(eye[0], eye[2]) + CAM_TERRAIN_CLEARANCE)
        self.camera.setPos(*(PHYS_TO_PANDA @ eye))
        # Aim ahead of the car in the chase view, at the car from the
        # side/front; panning sideways lifts the aim point so the view opens
        # up past the car instead of staring at the roof.
        ahead = config.CHASE_CAM_LOOK_AHEAD * max(0.0, math.cos(angle))
        look_h = (config.CHASE_CAM_LOOK_HEIGHT
                  + config.CHASE_CAM_ORBIT_LOOK_RISE * abs(math.sin(angle)))
        aim = v.pos + fwd * ahead + np.array([0.0, look_h, 0.0])
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
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")

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

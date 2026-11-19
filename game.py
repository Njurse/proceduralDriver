"""Application shell: mode state machine, transitions, input and persistence.

The heavy lifting lives in the small modules under ``core/`` and ``world/``.
This file only wires them together and decides when to switch scenes.
"""

import logging
import random
from pathlib import Path

from direct.showbase.ShowBase import ShowBase
from panda3d.core import (
    AmbientLight,
    ClockObject,
    DirectionalLight,
    Filename,
    PNMImage,
    Vec3,
    Vec4,
    WindowProperties,
)

import config
from core.audio import AudioManager
from core.camera import ChaseCamera
from core.flight import AtmosphereFlight, SpaceFlight
from core.foot import FootAvatar, FootController
from core.mathutil import clamp
from core.fuel import FuelSystem
from core.hud import HUD
from core.input import InputManager
from core.inventory import Inventory
from core.inventory_ui import InventoryOverlay
from core.map import MapOverlay
from core.mining import MiningLock
from core.pause import PAUSE_CONTINUE, PAUSE_EXIT, PAUSE_INVENTORY, PAUSE_SHIP, PauseOverlay
from core.pulse import PulseDrive
from core.pulse_fx import PulseFX
from core.save import SaveStore, default_state
from core.settings import load_settings, save_settings
from core.ship import Spaceship
from core.ship_profiles import get_profile
from core.trails import Trail
from core.transitions import Transition
from world.atmosphere import AtmosphereScene
from world.galaxy import Galaxy, GalaxyScene, make_starfield
from world.system import StarSystemScene
from world.traffic import TrafficManager

log = logging.getLogger("galaxy_explorer.game")

GALAXY = "galaxy"
SYSTEM = "system"
ATMOSPHERE = "atmosphere"
_MODES = (GALAXY, SYSTEM, ATMOSPHERE)


def _vec(value, fallback):
    try:
        return Vec3(float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return Vec3(fallback)


def _hpr(value, fallback=(0.0, 0.0, 0.0)):
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError, IndexError):
        return tuple(fallback)


def _plain_vec(value):
    return [float(value[0]), float(value[1]), float(value[2])]


class GalaxyExplorer(ShowBase):
    """Top level game object.  One instance per run."""

    def __init__(self, options=None):
        options = dict(options or {})
        self._options = options
        window_type = options.get("window_type", "onscreen")
        ShowBase.__init__(self, windowType=window_type)
        self.disableMouse()
        self.clock = ClockObject.getGlobalClock()

        self._load_state(options)
        self._setup_window()
        self._build_world()
        self._build_ui()
        self._bind_events()
        self._restore_or_start()

        self.taskMgr.add(self._update, "explorer-update")

    # ------------------------------------------------------------------ setup
    def _setup_window(self):
        self.setBackgroundColor(0.004, 0.006, 0.016, 1.0)
        try:
            props = WindowProperties()
            props.setTitle(f"{config.WINDOW_TITLE}  -  seed {self.seed}")
            props.setSize(config.WINDOW_WIDTH, config.WINDOW_HEIGHT)
            self.win.requestProperties(props)
        except Exception as exc:
            log.debug("[window] could not apply properties: %s", exc)
        try:
            self.camLens.setNearFar(0.5, 32000.0)
        except Exception as exc:
            log.debug("[window] could not set clip planes: %s", exc)

    def _load_state(self, options):
        persist = bool(options.get("persist", True))
        save_path = options.get("save_path", config.SAVE_PATH)
        self.save_store = SaveStore(save_path) if persist else None
        saved = self.save_store.load() if self.save_store else default_state()

        new_game = bool(options.get("new_game"))
        seed = options.get("seed")
        if seed is None and not new_game and saved.get("seed") is not None:
            seed = saved["seed"]
        if seed is None:
            seed = random.randrange(1, 1_000_000)
        self.seed = int(seed)

        resume = (not new_game and self.save_store is not None
                  and saved.get("seed") == self.seed)
        self._resume_state = saved if resume else None
        self.play_time = float(saved.get("play_time", 0.0)) if resume else 0.0

        self.visited = set()
        self.discovered = set()
        if resume:
            for index in saved.get("visited_systems", ()):
                try:
                    self.visited.add(int(index))
                except (TypeError, ValueError):
                    continue
            for key in saved.get("discovered_planets", ()):
                self.discovered.add(str(key))

    def _build_world(self):
        self.galaxy = Galaxy(self.seed)
        self.galaxy_scene = GalaxyScene(self, self.galaxy)
        self.starfield = make_starfield(
            config.STARFIELD_COUNT, config.STARFIELD_RADIUS, self.seed + 9)
        self.starfield.reparentTo(self.render)

        self.galaxy_root = self.render.attachNewNode("galaxy_root")
        self._build_galaxy_lights()
        self.traffic = None
        self._swap_hold = 0.0
        self._swap_target = None
        self.selected_ship = None

        selected_ship = None
        source = self._resume_state if self._resume_state else None
        if isinstance(source, dict):
            selected_ship = source.get("selected_ship")
        profile = get_profile(selected_ship) if selected_ship else None
        self.ship = Spaceship(self, profile=profile)
        self.ship.attach_to(self.galaxy_root)
        # Trails use world positions, so they must live outside the ship node.
        self.trail = Trail(self.render, self.ship.trail_color)
        self.selected_ship = profile["id"] if profile else "sparrow"

        self.space_flight = SpaceFlight()
        self.atmo_flight = AtmosphereFlight()
        self.chase_cam = ChaseCamera(self)

        self.mode = GALAXY
        self.system_index = None
        self.planet_index = None
        self.system_scene = None
        self.atmosphere = None
        self.galaxy_return = None
        self._boost = False
        self._save_timer = 0.0
        self.paused = False
        # Freighter hangar docking sequence.  States: None, "landing",
        # "docked" (on the pad) and "takeoff" (autopilot flying out).
        self._dock_state = None
        self._dock_hangar = None
        self._dock_ship = None
        # On-foot state.
        self.on_foot = False
        self.foot = None
        self.foot_avatar = None
        self.foot_mining = None
        self._foot_hangar = None
        self._foot_ship_pos = None
        self._foot_cam_pitch = 0.0

    def _build_galaxy_lights(self):
        ambient = AmbientLight("galaxy_ambient")
        ambient.setColor(Vec4(0.40, 0.46, 0.58, 1.0))
        self.galaxy_root.setLight(self.galaxy_root.attachNewNode(ambient))

        key = DirectionalLight("galaxy_key")
        key.setColor(Vec4(0.96, 0.94, 0.88, 1.0))
        key.setDirection(Vec3(0.35, 0.55, -0.75))
        self.galaxy_root.setLight(self.galaxy_root.attachNewNode(key))

        rim = DirectionalLight("galaxy_rim")
        rim.setColor(Vec4(0.32, 0.44, 0.74, 1.0))
        rim.setDirection(Vec3(0.15, 0.90, -0.35))
        self.galaxy_root.setLight(self.galaxy_root.attachNewNode(rim))

    def _build_ui(self):
        self._layout_aspect = None
        self.hud = HUD(self)
        self.map = MapOverlay(self, self.galaxy.systems)
        self.transition = Transition(self)
        self.pause = PauseOverlay(self)
        self.audio = AudioManager(self)
        self.input = InputManager(self)
        if hasattr(self, "ship") and self.ship is not None:
            self.audio.set_ship_profile(self.ship.profile)
        self.inventory = Inventory()
        self.mining = MiningLock()
        self.foot_mining = MiningLock(
            mining_range=config.FOOT_MINING_RANGE,
            cone_deg=config.FOOT_MINING_CONE_DEG)
        self.inventory_ui = InventoryOverlay(self)
        self.fuel = FuelSystem()
        self.pulse = PulseDrive(self.fuel)
        self.pulse_fx = PulseFX(self)

        # Preferences live in settings.json; an old save is used as fallback
        # when that file has been deleted.
        source = self._resume_state if self._resume_state else None
        self.pause.apply_settings(load_settings(
            fallback=source.get("settings") if source else None))
        if source and isinstance(source.get("inventory"), dict):
            self.inventory = Inventory.from_dict(source["inventory"])
        if source and isinstance(source.get("fuel"), dict):
            self.fuel = FuelSystem.from_dict(source["fuel"])
            self.pulse = PulseDrive(self.fuel)
        self._apply_settings()
        self.input.set_mouse_look(True)
        self._layout_overlays(force=True)

    def _bind_events(self):
        self.accept("mouse1", self._capture_mouse)
        self.accept("control-q", self._quit)
        self.accept("window-event", self._on_window_event)
        self.exitFunc = self._on_exit
        try:
            if self.win is not None:
                self.win.setCloseRequestEvent("window-close")
                self.accept("window-close", self._quit)
        except Exception as exc:
            log.debug("[window] close handler unavailable: %s", exc)

    def _on_window_event(self, window):
        # Re-flow overlays whenever the window changes size.  The true
        # aspect2d bounds are used because ShowBase may update them after this
        # callback; _update() also re-checks every frame.
        try:
            width = window.getXSize()
            height = max(1, window.getYSize())
        except Exception as exc:
            log.debug("[window] size query failed: %s", exc)
            return
        if width <= 0 or height <= 0:
            return
        self._layout_overlays(force=True)

    def _overlay_aspect(self):
        """Half-width of aspect2d in world units (the real visible bounds).

        ShowBase scales ``aspect2d`` by ``1 / aspect``; reading that scale is
        more reliable than ``getAspectRatio()``, which can report the new window
        size before the aspect node has been rescaled.
        """
        try:
            scale_x = abs(float(self.aspect2d.getScale().x))
            if scale_x > 1e-6:
                return 1.0 / scale_x
        except Exception as exc:
            log.debug("[layout] aspect2d scale unavailable: %s", exc)
        try:
            return max(float(self.getAspectRatio()), 0.5)
        except Exception as exc:
            log.debug("[layout] aspect ratio unavailable: %s", exc)
            return 1.0

    def _layout_overlays(self, force=False):
        """Re-flow every 2D overlay if the visible aspect changed."""
        aspect = self._overlay_aspect()
        if (not force and self._layout_aspect is not None
                and abs(aspect - self._layout_aspect) < 0.002):
            return
        self._layout_aspect = aspect
        self.hud.layout(aspect)
        self.map.layout(aspect)
        self.pause.layout(aspect)
        self.inventory_ui.layout(aspect)
        self.pulse_fx.layout(aspect)

    # ------------------------------------------------------------- restore/new
    def _restore_or_start(self):
        state = self._resume_state
        if state is None:
            self.space_flight.reset(self.galaxy.start_position,
                                    (0.0, 0.0, 0.0), 0.0)
            self._apply_mode()
            self._save_now()
            self.hud.show_message(
                f"Seed {self.seed}  -  fly to a star system and press E")
            return
        try:
            self._restore(state)
            self.hud.show_message("Flight log restored")
        except Exception as exc:  # corrupt save should never block play
            log.warning("[save] restore failed: %s", exc)
            self.space_flight.reset(self.galaxy.start_position,
                                    (0.0, 0.0, 0.0), 0.0)
            self._apply_mode()

    def _restore(self, state):
        mode = state.get("mode", GALAXY)
        if mode not in _MODES:
            mode = GALAXY
        index = state.get("system_index")
        planet_index = state.get("planet_index")
        ship = state.get("ship") or {}
        position = _vec(ship.get("pos"), self.galaxy.start_position)
        hpr = _hpr(ship.get("hpr"))
        throttle = float(ship.get("throttle", 0.0) or 0.0)
        self.galaxy_return = self._return_from_data(
            state.get("galaxy_return"))

        if mode in (SYSTEM, ATMOSPHERE) and self.galaxy.system_at(index) is not None:
            self._build_system(index)
            planet = self._planet_at(index, planet_index)
            if mode == ATMOSPHERE and planet is not None:
                self._build_atmosphere(planet)
                self.atmo_flight.reset(position, hpr, throttle)
                self.mode = ATMOSPHERE
                self.planet_index = planet.index
            else:
                self.space_flight.reset(position, hpr, throttle)
                self.mode = SYSTEM
        else:
            self.space_flight.reset(position, hpr, throttle)
            self.mode = GALAXY
        self._apply_mode()
        # Restore on-foot state if saved (planet surfaces only).
        if state.get("on_foot") and self.mode == ATMOSPHERE:
            try:
                self.atmo_flight.landed = True
                self.on_foot = False
                self._exit_to_foot()
                foot = state.get("foot") or {}
                if self.foot is not None and isinstance(foot, dict):
                    pos = _vec(foot.get("pos"), self.foot.position)
                    self.foot.position = Vec3(pos)
                    self.foot.yaw = float(foot.get("yaw", self.foot.yaw))
                    self.foot.quat.setHpr((self.foot.yaw, 0.0, 0.0))
                    if self.foot_avatar is not None:
                        self.foot_avatar.set_transform(self.foot.position,
                                                       self.foot.quat)
            except Exception as exc:
                log.warning("[save] on-foot restore failed: %s", exc)
                self._foot_teardown()

    @staticmethod
    def _return_from_data(data):
        if not isinstance(data, dict):
            return None
        return {
            "pos": _plain_vec(_vec(data.get("pos"), (0.0, 0.0, 0.0))),
            "hpr": list(_hpr(data.get("hpr"))),
            "throttle": float(data.get("throttle", 0.0) or 0.0),
        }

    # ----------------------------------------------------------------- scenes
    def _detach_ship(self):
        parent = self.ship.root.getParent()
        if not parent.isEmpty():
            self.ship.root.reparentTo(self.render)

    def _build_system(self, index):
        info = self.galaxy.system_at(index)
        if info is None:
            return False
        self._detach_ship()
        if self.system_scene is not None:
            self.system_scene.destroy()
            self.system_scene = None
        self.system_scene = StarSystemScene(self, info)
        self.system_scene.update(0.0)
        self.system_index = index
        self.visited.add(index)
        return True

    def _build_atmosphere(self, planet):
        if self.system_scene is None:
            return False
        self._detach_ship()
        if self.atmosphere is not None:
            self.atmosphere.destroy()
            self.atmosphere = None
        self.atmosphere = AtmosphereScene(self, self.system_scene.info, planet)
        return True

    def _planet_at(self, system_index, planet_index):
        info = self.galaxy.system_at(system_index)
        if info is None or planet_index is None:
            return None
        try:
            wanted = int(planet_index)
        except (TypeError, ValueError):
            return None
        for planet in info.planets:
            if planet.index == wanted:
                return planet
        return None

    def _current_planet(self):
        if self.mode != ATMOSPHERE:
            return None
        return self._planet_at(self.system_index, self.planet_index)

    def _controller(self):
        return self.atmo_flight if self.mode == ATMOSPHERE else self.space_flight

    def _mode_root(self):
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            return self.atmosphere.root
        if self.mode == SYSTEM and self.system_scene is not None:
            return self.system_scene.root
        return self.galaxy_root

    def _apply_mode(self):
        self.galaxy_scene.set_visible(self.mode == GALAXY)
        if self.mode == ATMOSPHERE:
            self.starfield.hide()
        else:
            self.starfield.show()
        if self.system_scene is not None:
            self.system_scene.set_visible(self.mode == SYSTEM)
        if self.atmosphere is not None:
            self.atmosphere.set_visible(self.mode == ATMOSPHERE)

        self.ship.attach_to(self._mode_root())
        self.ship.set_visible(True)

        # Traffic lifecycle.
        if getattr(self, "traffic", None) is not None:
            try:
                self.traffic.destroy()
            except Exception as exc:
                log.debug("[traffic] cleanup failed: %s", exc)
            self.traffic = None
        self._swap_hold = 0.0
        self._swap_target = None
        self.selected_ship = None
        if self.mode in (GALAXY, SYSTEM):
            try:
                traffic_seed = self.seed + (
                    0 if self.mode == GALAXY else 1000 + (self.system_index or 0))
                self.traffic = TrafficManager(
                    self, self._mode_root(), self.mode, self.galaxy,
                    self.system_scene, anchor=self._controller().position,
                    seed=traffic_seed)
            except Exception as exc:
                log.warning("traffic init failed: %s", exc)
                self.traffic = None
        self._swap_hold = 0.0
        self._swap_target = None
        self.selected_ship = None
        controller = self._controller()
        self.ship.set_transform(controller.position, controller.quat)
        self.chase_cam.snap(controller.position, controller.quat, self.mode,
                         self._height_sampler())
        self._apply_background()
        self.hud.set_prompt("")
        self._save_timer = 0.0

    def _height_sampler(self):
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            return self.atmosphere.height_at
        return None

    def _apply_background(self):
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            fog = self.atmosphere.palette["fog"]
            self.setBackgroundColor(fog[0] * 0.72, fog[1] * 0.72,
                                    fog[2] * 0.72, 1.0)
        elif self.mode == SYSTEM:
            self.setBackgroundColor(0.002, 0.003, 0.008, 1.0)
        else:
            self.setBackgroundColor(0.004, 0.006, 0.016, 1.0)

    # ------------------------------------------------------------- transitions
    def _begin_transition(self, swap):
        if self.transition.active:
            return False
        self.hud.set_prompt("")
        self.audio.play_transition()
        self.transition.play(swap)
        return True

    def _activate_system(self, index, save=True):
        info = self.galaxy.system_at(index)
        if not self._build_system(index):
            return
        spawn = Vec3(0.0, -(info.outer_radius + config.SYSTEM_ENTRY_PAD), 0.0)
        self.space_flight.reset(spawn, (0.0, 0.0, 0.0), 0.0)
        self.mode = SYSTEM
        self.planet_index = None
        self._apply_mode()
        if save:
            self._save_now()
        self.hud.show_message(f"{info.name}  -  {info.star_type}")

    def _activate_atmosphere(self, system_index, planet):
        if not self._build_atmosphere(planet):
            return
        position = Vec3(*self.atmosphere.entry_position())
        self.atmo_flight.reset(position, (0.0, 0.0, 0.0), 0.55)
        self.mode = ATMOSPHERE
        self.planet_index = planet.index
        self.discovered.add(planet.key(system_index))
        self._apply_mode()
        self._save_now()
        self.hud.show_message(
            f"Descending to {planet.name}  -  {planet.archetype['label']}")

    def _activate_orbit(self, planet):
        if self.system_scene is None or planet is None:
            return
        position = self.system_scene.planet_position(planet)
        offset = Vec3(0.0, -(planet.radius + config.SYSTEM_INTERACT_DIST
                             + 45.0), 0.0)
        if self.atmosphere is not None:
            self._detach_ship()
            self.atmosphere.destroy()
            self.atmosphere = None
        self.space_flight.reset(position + offset, (0.0, 0.0, 0.0), 0.45)
        self.mode = SYSTEM
        self.planet_index = None
        self._apply_mode()
        self._save_now()
        self.hud.show_message("Orbit achieved")

    def _activate_galaxy(self, use_return=True, fallback_system=None):
        self._detach_ship()
        if self.atmosphere is not None:
            self.atmosphere.destroy()
            self.atmosphere = None
        if self.system_scene is not None:
            self.system_scene.destroy()
            self.system_scene = None

        position = self.galaxy.start_position
        hpr = (0.0, 0.0, 0.0)
        throttle = 0.0
        if use_return and self.galaxy_return:
            state = self.galaxy_return
            position = _vec(state.get("pos"), position)
            hpr = _hpr(state.get("hpr"))
            throttle = float(state.get("throttle", 0.0) or 0.0)
        elif fallback_system is not None:
            position = fallback_system.position + Vec3(
                0.0, -(config.GALAXY_ENTRY_DIST + 320.0), 0.0)
        self.galaxy_return = None

        self.space_flight.reset(position, hpr, throttle)
        self.mode = GALAXY
        self.system_index = None
        self.planet_index = None
        self._apply_mode()

    def _enter_system_from_galaxy(self, system):
        if self.mode != GALAXY or self.transition.active:
            return
        self.galaxy_return = {
            "pos": _plain_vec(self.space_flight.position),
            "hpr": list(self.space_flight.quat.getHpr()),
            "throttle": self.space_flight.throttle,
        }
        self._begin_transition(
            lambda: self._activate_system(system.index, save=True))

    def _descend(self, planet):
        if self.mode != SYSTEM or self.system_scene is None:
            return
        index = self.system_index
        self._begin_transition(
            lambda: self._activate_atmosphere(index, planet))

    def _leave_atmosphere(self):
        if self.mode != ATMOSPHERE or self.transition.active:
            return
        planet = self._current_planet()
        self._begin_transition(lambda: self._activate_orbit(planet))

    def _leave_system(self):
        if self.mode != SYSTEM or self.transition.active:
            return
        info = self.galaxy.system_at(self.system_index)
        self._begin_transition(
            lambda: self._activate_galaxy(use_return=True,
                                          fallback_system=info))

    def _warp(self, index):
        target = self.galaxy.system_at(index)
        if target is None or self.transition.active:
            return
        self.galaxy_return = {
            "pos": _plain_vec(target.position + Vec3(
                0.0, -(config.GALAXY_ENTRY_DIST + 320.0), 0.0)),
            "hpr": [0.0, 0.0, 0.0],
            "throttle": 0.0,
        }
        self._begin_transition(
            lambda: self._activate_system(index, save=True))

    # ------------------------------------------------------------------- map
    def _open_map(self):
        if self.map.visible or self.transition.active:
            return
        if self.mode == ATMOSPHERE:
            self.hud.show_message("Galaxy map unavailable in atmosphere")
            return
        current = self.system_index if self.mode == SYSTEM else None
        selected = current if current is not None else 0
        visited_names = {
            self.galaxy.systems[index].name
            for index in self.visited
            if 0 <= index < len(self.galaxy.systems)
        }
        self.map.open(selected, current, self._controller().position,
                      visited_names, self.discovered)
        self.hud.set_visible(False)

    def _close_map(self):
        if self.map.visible:
            self.map.close()
            self.hud.set_visible(True)
            if not self.paused:
                self.input.set_mouse_look(True)

    def _open_inventory(self):
        if self.inventory_ui.visible:
            return
        self.inventory_ui.set_visible(True, self.inventory)
        self.hud.set_visible(False)
        self.input.set_mouse_look(False)

    def _close_inventory(self):
        if self.inventory_ui.visible:
            self.inventory_ui.set_visible(False)
            self.hud.set_visible(True)
            if not self.paused:
                self.input.set_mouse_look(True)

    def _cycle_ship(self):
        from core.ship_profiles import SHIP_PROFILES, get_profile
        ids = [p["id"] for p in SHIP_PROFILES]
        try:
            idx = ids.index(self.selected_ship)
        except ValueError:
            idx = 0
        idx = (idx + 1) % len(ids)
        new_id = ids[idx]
        profile = get_profile(new_id)
        self.ship.set_profile(profile)
        self.audio.set_ship_profile(profile)
        if getattr(self, "trail", None) is not None:
            try:
                self.trail.destroy()
            except Exception as exc:
                log.debug("[trail] cleanup failed: %s", exc)
        # Trails store world positions, so parent them to render, not the ship.
        self.trail = Trail(self.render, self.ship.trail_color)
        self.selected_ship = new_id
        self.hud.show_message(f"Ship: {profile['name']}")
        self._save_now()

    def _update_map(self, dt, inp):
        self.map.update_ship(self._controller().position)
        # While typing a search query, let MapOverlay own the keyboard.
        if self.map.search_mode:
            return
        if inp.select_prev:
            self.map.move(-1)
        if inp.select_next:
            self.map.move(1)
        if inp.cancel or inp.toggle_map:
            self._close_map()
            return
        if inp.confirm:
            index = self.map.selected_index
            current = self.system_index if self.mode == SYSTEM else None
            self._close_map()
            if current is not None and index == current:
                self.hud.show_message("Already inside this system")
            else:
                self._warp(index)

    # ----------------------------------------------------------------- input
    def _capture_mouse(self):
        if self.map.visible or self.transition.active:
            return
        self.input.set_mouse_look(True)

    def _handle_common_input(self, inp):
        if inp.toggle_help:
            self.hud.set_help_visible(not self.hud.help_visible)
            self.audio.play_ui_click()
        # Esc is context sensitive: close whatever is open, otherwise release
        # the mouse, and finally open the pause menu when the cursor is free.
        if inp.release_mouse:
            if self.map.visible:
                if not self.map.handle_escape():
                    self._close_map()
            elif self.inventory_ui.visible:
                self._close_inventory()
            elif self.paused:
                # Esc walks back one menu level, then resumes from the main page.
                if self.pause.back():
                    self._resume_from_pause()
            elif self.input.mouse_look:
                self.input.set_mouse_look(False)
            else:
                self._open_pause()
        if inp.pause and not self.transition.active:
            self._toggle_pause()
        if (inp.toggle_inventory and not self.transition.active
                and not self.paused and not self.map.visible):
            if self.inventory_ui.visible:
                self._close_inventory()
            else:
                self._open_inventory()
        if self.inventory_ui.visible and inp.cancel:
            self._close_inventory()
        if inp.toggle_map and not self.transition.active:
            if self.map.visible:
                self._close_map()
            elif not self.paused and not self.inventory_ui.visible:
                self._open_map()

    # ---------------------------------------------------------------- settings/pause
    def _apply_settings(self, persist=False):
        settings = self.pause.settings()
        self.audio.set_volume_settings(
            settings.get("master_volume"), settings.get("effects_volume"))
        self.input.apply_settings(settings)
        if persist:
            save_settings(settings)

    def _open_pause(self):
        if (self.paused or self.transition.active or self.map.visible
                or self.inventory_ui.visible):
            return
        self.paused = True
        self.pause.set_visible(True)
        self.hud.set_visible(False)
        # Free the cursor so menu navigation is visible and clickable.
        self.input.set_mouse_look(False)

    def _toggle_pause(self):
        if self.transition.active:
            return
        if self.paused:
            # Start/P acts as "back" inside the options page.
            if self.pause.in_options:
                self.pause.back()
            else:
                self._resume_from_pause()
        else:
            self._open_pause()

    def _resume_from_pause(self):
        self.paused = False
        self.pause.set_visible(False)
        self.hud.set_visible(True)
        if not self.map.visible and not self.inventory_ui.visible:
            self.input.set_mouse_look(True)

    def _update_pause(self, dt, inp):
        moved = False
        if self.pause.in_options:
            if inp.select_down:
                self.pause.move(1)
                moved = True
            if inp.select_up:
                self.pause.move(-1)
                moved = True
            if inp.select_prev and self.pause.adjust(-1):
                self._apply_settings(persist=True)
                moved = True
            if inp.select_next and self.pause.adjust(1):
                self._apply_settings(persist=True)
                moved = True
        else:
            if inp.select_down or inp.select_next:
                self.pause.move(1)
                moved = True
            if inp.select_up or inp.select_prev:
                self.pause.move(-1)
                moved = True
        if moved:
            self.audio.play_ui_click()
        if inp.confirm:
            self.audio.play_ui_click()
            self._pause_activate()
        # Esc is handled by _handle_common_input (which resumes or pops a
        # menu level); other cancel buttons (Select/B) go back one level here.
        # Excluding release_mouse avoids closing the menu on the same frame
        # that Esc opened it.
        if inp.cancel and not inp.release_mouse and self.pause.back():
            self._resume_from_pause()

    def _pause_activate(self):
        action = self.pause.activate()
        if action == PAUSE_CONTINUE:
            self._resume_from_pause()
        elif action == PAUSE_INVENTORY:
            self._resume_from_pause()
            self._open_inventory()
        elif action == PAUSE_SHIP:
            self._cycle_ship()
        elif action == PAUSE_EXIT:
            self._quit()

    # ------------------------------------------------------------------ frame
    def _update(self, task):
        dt = min(self.clock.getDt(), 0.05)
        if dt <= 0.0:
            dt = 1.0 / 60.0
        self.play_time += dt
        self._layout_overlays()

        self.transition.update(dt)
        inp = self.input.poll()
        map_was_visible = self.map.visible
        self._handle_common_input(inp)

        if self.transition.active:
            self._update_scene(dt)
        elif self.paused:
            self._update_pause(dt, inp)
        elif self.inventory_ui.visible:
            self.mining.target = None
            if inp.select_down:
                self.inventory_ui.move(0, 1)
            if inp.select_up:
                self.inventory_ui.move(0, -1)
            if inp.select_next:
                self.inventory_ui.move(1, 0)
            if inp.select_prev:
                self.inventory_ui.move(-1, 0)
            self.inventory_ui.refresh(self.inventory)
        elif self.map.visible:
            # Keep simulating flight so the ship keeps thrusting forward.
            if self.map.search_mode:
                # Typing a query must not also steer the ship.
                for axis in ("throttle_axis", "pitch_axis", "yaw_axis",
                             "lateral_axis", "pad_roll_axis"):
                    setattr(inp, axis, 0.0)
                inp.boost = False
                inp.brake = False
            self._update_flight(dt, inp)
            if self.traffic is not None:
                try:
                    self.traffic.set_anchor(self._controller().position)
                    self.traffic.update(dt)
                except Exception as exc:
                    log.debug("[traffic] update failed: %s", exc)
            # Skip map navigation on the frame the overlay was just opened;
            # the same key edge would otherwise close it immediately.
            if map_was_visible:
                self._update_map(dt, inp)
        else:
            if self.on_foot:
                self._update_foot_scene(dt)
                self._update_foot(dt, inp)
                self._update_foot_mining(dt, inp)
                self._update_foot_interactions(inp)
            else:
                self._update_flight(dt, inp)
                if self.traffic is not None:
                    try:
                        self.traffic.set_anchor(self._controller().position)
                        self.traffic.update(dt)
                    except Exception as exc:
                        log.debug("[traffic] update failed: %s", exc)
                self._update_interactions(inp)
                if self.mode == ATMOSPHERE:
                    self._update_mining(dt, inp)
                    self._update_ship_swap(dt, inp)

        if not self.paused:
            self.hud.update(dt, self._telemetry())
        self._update_audio(dt)
        self._update_autosave(dt)
        return task.cont

    def _update_scene(self, dt):
        controller = self._controller()
        self.ship.update(dt, controller.throttle, False)
        self.ship.set_transform(controller.position, controller.quat)
        if getattr(self, "trail", None) is not None:
            try:
                self.trail.update(controller.position)
                self.trail.set_color(self.ship.trail_color)
            except Exception as exc:
                log.debug("[trail] update failed: %s", exc)
        self.chase_cam.update(dt, controller.position, controller.quat,
                           self.mode, self._height_sampler(), False,
                           speed=controller.speed)
        if self.mode == SYSTEM and self.system_scene is not None:
            self.system_scene.update(dt)
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            self.atmosphere.update(dt, self.chase_cam.position,
                                   controller.position)

    def _update_flight(self, dt, inp):
        controller = self._controller()
        sens = float(self.pause.settings().get("control_sensitivity", 1.0))
        # Mouse sensitivity and Y inversion are applied by InputManager so
        # they also affect any other consumer of mouse deltas.
        dx, dy = inp.mouse_dx, inp.mouse_dy
        if self._dock_state is not None:
            # Hangar autopilot owns the ship; manual flight and pulse are off.
            self._update_docking(dt, inp, controller)
            return
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            height_sampler = self.atmosphere.height_at
        else:
            height_sampler = None

        can_pulse = self.mode in (GALAXY, SYSTEM)
        # Pulse update first so countdown/engage state is current; it also
        # handles Y-hold detection via inp.boost.
        self.pulse.update(dt, inp, controller, height_sampler, self.audio,
                          self.pulse_fx, can_pulse)

        # Suppress turbo while counting down or pulsing.
        flight_boost_flag = inp.boost
        if self.pulse.state in ("countdown", "active"):
            flight_boost_flag = False

        if self.pulse.state == "active":
            boost = False
            self._boost = False
        else:
            inp.boost = flight_boost_flag
            if self.mode == ATMOSPHERE and self.atmosphere is not None:
                boost = controller.update(dt, inp, dy, dx, height_sampler,
                                         sensitivity=sens)
            else:
                boost = controller.update(dt, inp, dy, dx, sensitivity=sens)
            self._boost = bool(boost)

        self.ship.update(dt, controller.throttle,
                         boost or self.pulse.state == "active")
        self.ship.set_transform(controller.position, controller.quat)
        if getattr(self, "trail", None) is not None:
            try:
                self.trail.update(controller.position)
                self.trail.set_color(self.ship.trail_color)
            except Exception as exc:
                log.debug("[trail] update failed: %s", exc)
        self.chase_cam.update(dt, controller.position, controller.quat,
                           self.mode, self._height_sampler(), boost,
                           speed=controller.speed)

        if self.pulse.state == "active":
            try:
                self.base.camLens.setFov(config.PULSE_FOV)
            except Exception as exc:
                log.debug("[camera] pulse FOV failed: %s", exc)
        elif getattr(self.pulse, "fov_rebound", 0.0) > 0:
            t = self.pulse.fov_rebound / (config.PULSE_PULLOUT_FLASH + 0.35)
            fov = (config.PULSE_PULLOUT_FOV_REBOUND
                   + (70.0 - config.PULSE_PULLOUT_FOV_REBOUND) * (1.0 - t))
            try:
                self.base.camLens.setFov(fov)
            except Exception as exc:
                log.debug("[camera] FOV rebound failed: %s", exc)

        self.pulse_fx.update(dt)

        if self.mode == SYSTEM and self.system_scene is not None:
            self.system_scene.update(dt)
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            self.atmosphere.update(dt, self.chase_cam.position,
                                   controller.position)
            self._check_atmosphere_exit()

    # ---------------------------------------------------- hangar docking flow
    def _begin_hangar_landing(self, traffic_ship):
        """Freeze the freighter and hand the ship to the landing autopilot."""
        traffic_ship.docked = True
        traffic_ship.velocity = Vec3(0.0)
        traffic_ship.hangar.begin_docking()
        self._dock_hangar = traffic_ship.hangar
        self._dock_ship = traffic_ship
        self._dock_state = "landing"
        self.audio.play_transition()
        self.hud.show_message("Landing autopilot engaged")

    def _end_hangar_docking(self):
        """Takeoff autopilot finished: release the freighter and give back control."""
        if self._dock_ship is not None:
            self._dock_ship.docked = False
        self._dock_state = None
        self._dock_hangar = None
        self._dock_ship = None
        self.hud.show_message("Autopilot complete  -  you have control")

    # ---------------------------------------------------------------- on-foot
    def _can_exit_to_foot(self):
        if self.on_foot:
            return False
        if self.mode == ATMOSPHERE:
            if not bool(getattr(self.atmo_flight, "landed", False)):
                return False
            # Validate exit spot: not underwater, not too steep.
            if self.atmosphere is not None:
                try:
                    from core.flight import _surface_normal
                    pos = self.atmo_flight.position
                    terrain = self.atmosphere.terrain
                    wl = terrain.water_level
                    ground = self.atmosphere.height_at(pos.x, pos.y)
                    if wl is not None and ground < wl + 1.0:
                        return False
                    normal = _surface_normal(self.atmosphere.height_at, pos.x, pos.y)
                    if normal.z < config.FOOT_EXIT_MAX_SLOPE:
                        return False
                except Exception:
                    pass
            return True
        return self.mode == GALAXY and self._dock_state == "docked"

    def _exit_to_foot(self):
        if not self._can_exit_to_foot():
            self.hud.show_message("Can't exit here  -  move to flatter ground")
            return
        if self.mode == ATMOSPHERE:
            controller = self.atmo_flight
            pos = Vec3(controller.position)
            # spawn beside ship
            pos.x += 3.0
            sampler = self.atmosphere.height_at if self.atmosphere else None
            self._foot_hangar = None
        else:
            hangar = self._dock_hangar
            if hangar is None:
                return
            pad_pos = hangar.pad.world_position(self.render)
            pos = pad_pos + Vec3(3.0, 0.0, config.FOOT_FOOT_HEIGHT)

            def sampler(x, y):
                return self._hangar_floor_z(hangar, x, y)

            self._foot_hangar = hangar
        self._foot_ship_pos = Vec3(
            self.atmo_flight.position if self.mode == ATMOSPHERE
            else self.space_flight.position)

        self.foot = FootController()
        self.foot.reset(pos, yaw=0.0)
        self.foot_avatar = FootAvatar(self)
        self.foot_avatar.attach_to(self.render)
        self.foot_avatar.set_transform(self.foot.position, self.foot.quat)
        # ground clamp once
        if sampler is not None:
            try:
                g = float(sampler(self.foot.position.x, self.foot.position.y))
                self.foot.position.z = g + config.FOOT_FOOT_HEIGHT
            except Exception:
                pass
        self.ship.set_visible(False)
        self.on_foot = True
        self._foot_cam_pitch = 0.0
        if self.chase_cam is not None:
            self.chase_cam.foot_pitch = 0.0
        self.chase_cam.snap(self.foot.position, self.foot.quat, "onfoot",
                            lambda x, y: (sampler(x, y) if sampler else 0.0))
        self.hud.show_message("On foot  -  E to re-enter ship")

    def _hangar_floor_z(self, hangar, x, y):
        try:
            local = hangar.to_local(Vec3(x, y, 0.0))
            floor_local = Vec3(local.x, local.y, -hangar.half_h)
            origin = hangar.root.getPos(self.render)
            quat = hangar.root.getQuat(self.render)
            scale = max(hangar.root.getScale().x, 1e-6)
            world = origin + quat.xform(floor_local * scale)
            return float(world.z)
        except Exception:
            return 0.0

    def _enter_ship_from_foot(self):
        if not self.on_foot:
            return
        if self.foot_avatar is not None:
            self.foot_avatar.destroy()
            self.foot_avatar = None
        self.foot = None
        self.on_foot = False
        self.ship.set_visible(True)
        try:
            self.ship.root.setColorScale(1.0, 1.0, 1.0, 1.0)
        except Exception:
            pass
        if self.mode == ATMOSPHERE:
            # ship hasn't moved; keep landed state
            self.atmo_flight.landed = True
        self._foot_hangar = None
        self.hud.show_message("Back in ship")

    def _foot_height_sampler(self):
        if self._foot_hangar is not None:
            hg = self._foot_hangar
            return lambda x, y: self._hangar_floor_z(hg, x, y)
        if self.atmosphere is not None:
            return self.atmosphere.height_at
        return None

    def _foot_bounds(self, pos):
        if self._foot_hangar is not None:
            try:
                return self._foot_hangar.is_inside(pos)
            except Exception:
                return True
        return True

    def _update_foot(self, dt, inp):
        sampler = self._foot_height_sampler()
        hs = sampler if sampler is not None else (lambda x, y: 0.0)
        bounds_fn = None
        if self._foot_hangar is not None:
            hg = self._foot_hangar

            def bounds_fn(p):  # noqa: E306
                return hg.is_inside(p)
        mouse_dx, mouse_dy = inp.mouse_dx, inp.mouse_dy
        # Camera pitch orbit via mouse up/down (only when not turning? always accumulate).
        self._foot_cam_pitch = clamp(
            self._foot_cam_pitch - mouse_dy * 2.0, -0.7, 0.7)
        if self.chase_cam is not None:
            self.chase_cam.foot_pitch = self._foot_cam_pitch
        if self.foot is None:
            return
        prev_pos = Vec3(self.foot.position)
        self.foot.update(dt, inp, mouse_dx, mouse_dy, hs,
                         bounds=bounds_fn)
        # If bounds rejected, revert
        if bounds_fn is not None and not bounds_fn(self.foot.position):
            self.foot.position = prev_pos
            self.foot.velocity.x = 0.0
            self.foot.velocity.y = 0.0
        moving = self.foot.speed > 0.1
        if self.foot_avatar is not None:
            self.foot_avatar.set_transform(self.foot.position, self.foot.quat)
            self.foot_avatar.update(dt, moving, self.foot.speed,
                                    turn_rate=getattr(self.foot, "turn_rate", 0.0),
                                    landed=getattr(self.foot, "_landed_this_frame", False))
        self.chase_cam.update(dt, self.foot.position, self.foot.quat, "onfoot",
                              hs, False, speed=self.foot.speed)

    def _update_foot_mining(self, dt, inp):
        if self.mode != ATMOSPHERE or self.atmosphere is None:
            if self.foot_mining is not None:
                self.foot_mining.target = None
            return
        if self.foot is None:
            return
        veins = self.atmosphere.veins
        level = 1 + min(2, self.inventory.total_items()
                        // config.MINING_TOOL_EVERY)
        tool_speed = 1.0 + config.MINING_TOOL_STEP * (level - 1)
        depleted = self.foot_mining.update(
            self.foot.position, self.foot.quat, veins, dt,
            getattr(inp, "mine_held", False), tool_speed=tool_speed)
        if self.foot_mining.target is None:
            self.foot_mining.progress = 0.0
        if depleted is not None:
            mid = depleted.material["id"]
            amount = getattr(depleted, "last_yield", 1) or 1
            tier = depleted.material.get("tier", "common")
            if self.inventory.add(mid, amount):
                self.hud.show_message(
                    f"Mined {depleted.material['name']} x{amount}  ({tier})",
                    color=depleted.material["color"])
                self.audio.play_ui_click()
            else:
                self.hud.show_message("Inventory full")

    def _update_foot_interactions(self, inp):
        if self.foot is None:
            return
        # Re-enter ship.
        ship_pos = (self._foot_ship_pos if self._foot_ship_pos is not None
                    else self.foot.position)
        dist = (ship_pos - self.foot.position).length()
        if dist <= config.FOOT_REENTER_RANGE:
            self.hud.set_prompt("E  -  enter ship")
            # Highlight the ship while re-enter is available.
            try:
                self.ship.root.setColorScale(1.25, 1.25, 0.7, 1.0)
            except Exception:
                pass
            if inp.interact:
                try:
                    self.ship.root.setColorScale(1.0, 1.0, 1.0, 1.0)
                except Exception:
                    pass
                self._enter_ship_from_foot()
                return
        else:
            self.hud.set_prompt("")
            try:
                self.ship.root.setColorScale(1.0, 1.0, 1.0, 1.0)
            except Exception:
                pass
        # Mining prompt handled by mining lock.
        locked = (self.foot_mining.target if self.foot_mining is not None
                  else None)
        if locked is not None:
            self.hud.set_prompt(
                f"Lock: {locked.material['name']}  Hold R to mine")
            self.hud.set_crosshair_color((1.0, 0.85, 0.3, 1.0))
        elif self.on_foot and self.foot is not None and self.foot.jetpack_active:
            self.hud.set_crosshair_color((1.0, 0.4, 0.1, 1.0))
        else:
            self.hud.set_crosshair_color((0.7, 0.94, 1.0, 0.85))

    def _update_foot_scene(self, dt):
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            self.atmosphere.update(dt, self.chase_cam.position,
                                    self.foot.position if self.foot else self.chase_cam.position)
        # hangar update
        if self._foot_hangar is not None:
            import contextlib
            with contextlib.suppress(Exception):
                self._foot_hangar.update(
                    dt, self.foot.position if self.foot
                    else self.chase_cam.position)

    def _foot_teardown(self):
        if self.foot_avatar is not None:
            self.foot_avatar.destroy()
            self.foot_avatar = None
        self.foot = None
        self.on_foot = False

    def _update_docking(self, dt, inp, controller):
        """Run the hangar landing/takeoff autopilot and sync the visuals."""
        hangar = self._dock_hangar
        if hangar is None:
            self._dock_state = None
            return
        if self._dock_state in ("landing", "docked"):
            phase = hangar.dock_update(controller, dt)
            if phase == "docked" and self._dock_state == "landing":
                self._dock_state = "docked"
                self.hud.show_message(
                    "Docked  -  hold throttle to launch")
            if self._dock_state == "docked" and inp.throttle_axis > 0.05:
                hangar.begin_takeoff()
                self._dock_state = "takeoff"
                self.audio.play_transition()
                self.hud.show_message("Launching  -  autopilot has the ship")
        else:
            if hangar.takeoff_update(controller, dt):
                self._end_hangar_docking()

        # Keep the ship model, trail and camera in sync with the autopilot.
        self.ship.update(dt, controller.throttle, False)
        self.ship.set_transform(controller.position, controller.quat)
        if getattr(self, "trail", None) is not None:
            try:
                self.trail.update(controller.position)
                self.trail.set_color(self.ship.trail_color)
            except Exception as exc:
                log.debug("[trail] update failed: %s", exc)
        self.chase_cam.update(dt, controller.position, controller.quat,
                              self.mode, None, False,
                              speed=controller.speed)

    def _update_mining(self, dt, inp):
        if self.mode != ATMOSPHERE or self.atmosphere is None:
            self.mining.target = None
            return
        veins = self.atmosphere.veins
        # Mining tool upgrades automatically as the haul grows.
        level = 1 + min(2, self.inventory.total_items()
                        // config.MINING_TOOL_EVERY)
        tool_speed = 1.0 + config.MINING_TOOL_STEP * (level - 1)
        depleted = self.mining.update(
            self.atmo_flight.position, self.atmo_flight.quat, veins, dt,
            getattr(inp, "mine_held", False), tool_speed=tool_speed)
        if depleted is not None:
            mid = depleted.material["id"]
            amount = getattr(depleted, "last_yield", 1) or 1
            tier = depleted.material.get("tier", "common")
            if self.inventory.add(mid, amount):
                self.hud.show_message(
                    f"Mined {depleted.material['name']} x{amount}  ({tier})",
                    color=depleted.material["color"])
                self.audio.play_ui_click()
            else:
                self.hud.show_message("Inventory full")

    def _update_ship_swap(self, dt, inp):
        target = getattr(self, "_swap_target", None)
        if target is None:
            self._swap_hold = 0.0
            return
        held = (self.input._down("interact")
                or (self.input.pad.connected and self.input.pad.a_button))
        if held:
            self._swap_hold += dt
        else:
            self._swap_hold = 0.0
        if self._swap_hold >= config.LANDED_SHIP_SWAP_HOLD:
            profile = target["profile"]
            self.ship.set_profile(profile)
            self.audio.set_ship_profile(profile)
            # recreate trail with new color
            try:
                if getattr(self, "trail", None) is not None:
                    self.trail.destroy()
            except Exception as exc:
                log.debug("[trail] swap cleanup failed: %s", exc)
            self.trail = Trail(self.render, self.ship.trail_color)
            self.selected_ship = profile["id"]
            # remove landed ship
            try:
                target["ship"].root.removeNode()
            except Exception as exc:
                log.debug("[swap] landed ship removal failed: %s", exc)
            if target in self.atmosphere.landed_ships:
                self.atmosphere.landed_ships.remove(target)
            self._swap_target = None
            self._swap_hold = 0.0
            self.hud.show_message(f"Swapped to {profile['name']}")
            self._save_now()

    def _check_atmosphere_exit(self):
        if self.transition.active or self.atmosphere is None:
            return
        if self.atmo_flight.altitude >= config.ATMO_EXIT_ALTITUDE:
            self._leave_atmosphere()

    def _update_interactions(self, inp):
        if self.transition.active:
            return
        if self.mode == GALAXY:
            if self._dock_state is not None:
                # Prompts are handled by the docking messages.
                self.hud.set_prompt("")
                if self._dock_state == "docked" and inp.interact:
                    self._exit_to_foot()
                return
            hangar_ts = (self.traffic.nearest_hangar(self.space_flight.position)
                         if self.traffic is not None else None)
            if (hangar_ts is not None
                    and hangar_ts.state not in ("warp_in", "warp_out")):
                prompt = hangar_ts.hangar.prompt_state(
                    self.space_flight.position)
                if prompt is not None:
                    self.hud.set_prompt(prompt)
                    if (hangar_ts.hangar.is_inside(self.space_flight.position)
                            and inp.land):
                        self._begin_hangar_landing(hangar_ts)
                    return
            system, distance = self.galaxy.nearest_system(
                self.space_flight.position)
            if system is not None and distance <= config.GALAXY_ENTRY_DIST:
                self.hud.set_prompt(f"E  -  enter {system.name}")
                if inp.interact:
                    self._enter_system_from_galaxy(system)
            else:
                self.hud.set_prompt("")
        elif self.mode == SYSTEM and self.system_scene is not None:
            planet, surface, _pos = self.system_scene.nearest_planet(
                self.space_flight.position)
            if planet is not None and surface <= config.SYSTEM_INTERACT_DIST:
                self.hud.set_prompt(f"E  -  descend to {planet.name}")
                if inp.interact:
                    self._descend(planet)
            elif self.space_flight.position.length() >= config.SYSTEM_EXIT_DIST:
                self.hud.set_prompt(
                    f"E  -  leave {self.system_scene.info.name}")
                if inp.interact:
                    self._leave_system()
            else:
                self.hud.set_prompt("")
        elif self.mode == ATMOSPHERE:
            altitude = self.atmo_flight.altitude
            if self.atmosphere is not None:
                ground = self.atmosphere.height_at(
                    self.atmo_flight.position.x, self.atmo_flight.position.y)
                agl = altitude - ground
            else:
                agl = altitude
            # Landed ship swap check.
            swap_target = None
            if self.atmosphere is not None:
                for entry in self.atmosphere.landed_ships:
                    pos = entry["position"]
                    dist = (pos - self.atmo_flight.position).length()
                    if dist <= config.LANDED_SHIP_SWAP_RANGE:
                        swap_target = entry
                        break
            self._swap_target = swap_target
            if swap_target is not None:
                locked = self.mining.target
                if locked is not None:
                    self.hud.set_prompt(
                        f"Lock: {locked.material['name']}  Hold R/A to mine")
                else:
                    self.hud.set_prompt("Hold E/A to swap ship")
            else:
                self._swap_hold = 0.0
                self._swap_target = None
                locked = self.mining.target
                if locked is not None:
                    self.hud.set_prompt(
                        f"Lock: {locked.material['name']}  Hold R/A to mine")
                elif self.atmo_flight.landed:
                    self.hud.set_prompt("E - exit ship   Hold throttle - lift off")
                    if inp.interact:
                        self._exit_to_foot()
                elif agl <= config.LAND_MAX_ALTITUDE:
                    self.hud.set_prompt("X/F - land   Hold throttle - lift off")
                elif altitude >= config.ATMO_EXIT_ALTITUDE:
                    self.hud.set_prompt("Climbing to orbit")
                elif altitude >= config.ATMO_EXIT_ALTITUDE * 0.84:
                    self.hud.set_prompt("Climb higher to reach orbit")
                else:
                    self.hud.set_prompt("")

    # ------------------------------------------------------------- telemetry
    def _location_name(self):
        if self.mode == ATMOSPHERE:
            planet = self._current_planet()
            if planet is not None:
                return f"{planet.name}  -  {planet.archetype['label']}"
            return "Atmosphere"
        if self.mode == SYSTEM and self.system_scene is not None:
            return self.system_scene.info.name
        system, distance = self.galaxy.nearest_system(
            self.space_flight.position)
        if system is not None:
            return f"Deep Space  -  {system.name} {distance:.0f} u"
        return "Deep Space"

    def _telemetry(self):
        controller = self._controller()
        altitude = 0.0
        distance = 0.0
        if self.mode == ATMOSPHERE and self.atmosphere is not None:
            ground = self.atmosphere.height_at(controller.position.x,
                                               controller.position.y)
            altitude = controller.position.z - ground
        elif self.mode == SYSTEM and self.system_scene is not None:
            distance = self.system_scene.nearest_planet(
                controller.position)[1]
        else:
            distance = self.galaxy.nearest_system(controller.position)[1]
        position = controller.position
        mining_target = None
        if self.mode == ATMOSPHERE and self.mining is not None:
            locked = self.mining.target
            if locked is not None:
                mining_target = locked.material["name"]
        pulse_state = getattr(self.pulse, "state", "idle")
        return {
            "mode": self.mode,
            "location": self._location_name(),
            "speed": controller.speed,
            "throttle": controller.throttle,
            "boost": self._boost,
            "seed": self.seed,
            "gamepad": self.input.pad.connected,
            "altitude": max(0.0, altitude),
            "distance": max(0.0, distance),
            "position": (position.x, position.y, position.z),
            "mining_progress": (self.mining.progress
                                if self.mining.target is not None else 0.0),
            "mining_target": mining_target,
            "on_foot": bool(self.on_foot),
            "jetpack_fuel": float(self.foot.jetpack_fuel) if self.foot else 0.0,
            "fuel": self.fuel.fuel,
            "pulse_fuel": self.fuel.pulse_fuel,
            "pulse_state": pulse_state,
            "pulse_countdown": getattr(self.pulse, "countdown", 0.0),
        }

    def _update_audio(self, dt):
        controller = self._controller()
        if self.mode == ATMOSPHERE:
            ratio = controller.speed / max(config.ATMO_MAX_SPEED, 1.0)
        else:
            ratio = controller.speed / max(config.SPACE_MAX_SPEED, 1.0)
        self.audio.update(dt, self.mode, controller.throttle, ratio,
                          self._boost)
        # Freighter hangar proximity hum.
        hum = 0.0
        if self.mode == GALAXY and self.traffic is not None:
            for traffic_ship in self.traffic.ships:
                if traffic_ship.hangar is not None:
                    hum = max(hum, traffic_ship.hangar.hum_level(
                        controller.position))
        self.audio.set_hangar_hum(hum)

    # ------------------------------------------------------------------ save
    def _snapshot(self):
        # If walking, snapshot the ship (landed) pose and foot pose separately.
        ship_controller = (self.atmo_flight if self.mode == ATMOSPHERE
                          else self.space_flight)
        hpr = ship_controller.quat.getHpr()
        snap = {
            "version": config.SAVE_VERSION,
            "seed": self.seed,
            "mode": self.mode,
            "system_index": self.system_index,
            "planet_index": self.planet_index,
            "ship": {
                "pos": _plain_vec(ship_controller.position),
                "hpr": [float(hpr.x), float(hpr.y), float(hpr.z)],
                "throttle": float(ship_controller.throttle),
            },
            "visited_systems": sorted(int(i) for i in self.visited),
            "discovered_planets": sorted(str(k) for k in self.discovered),
            "play_time": round(self.play_time, 2),
            "galaxy_return": self.galaxy_return,
            "settings": self.pause.settings(),
            "inventory": self.inventory.to_dict(),
            "fuel": self.fuel.to_dict(),
            "selected_ship": self.selected_ship,
            "on_foot": bool(self.on_foot),
        }
        if self.on_foot and self.foot is not None:
            snap["foot"] = {
                "pos": _plain_vec(self.foot.position),
                "yaw": float(self.foot.yaw),
            }
        return snap

    def _save_now(self):
        self._save_timer = 0.0
        if self.save_store is not None:
            self.save_store.save(self._snapshot())

    def _update_autosave(self, dt):
        if self.save_store is None:
            return
        self._save_timer += dt
        if self._save_timer >= config.AUTOSAVE_SECONDS:
            self._save_now()

    # ------------------------------------------------------------------ exit
    def _on_exit(self):
        self._save_now()
        try:
            self.audio.stop_all()
        except Exception as exc:
            log.debug("[audio] shutdown failed: %s", exc)

    def _quit(self):
        self._on_exit()
        self.userExit()

    # ----------------------------------------------------------- smoke tests
    def run_smoke(self, folder=None):
        """Render one frame of each scene for automated visual checks."""
        folder = Path(folder or config.SAVE_DIR)
        folder.mkdir(parents=True, exist_ok=True)
        results = []

        self._activate_galaxy(use_return=False)
        results.append(self._capture(folder / "smoke_galaxy.png", "galaxy"))

        info = self.galaxy.system_at(0)
        self._activate_system(0, save=False)
        results.append(self._capture(folder / "smoke_system.png", "system"))

        planet = info.planets[0] if info is not None and info.planets else None
        if planet is not None:
            self._activate_atmosphere(0, planet)
            results.append(
                self._capture(folder / "smoke_atmosphere.png", "atmosphere"))

        self._activate_galaxy(use_return=False)
        return results

    def _capture(self, path, label):
        for _ in range(3):
            self.graphicsEngine.renderFrame()
        image = PNMImage()
        self.win.getScreenshot(image)
        image.write(Filename.fromOsSpecific(str(path)))
        return path, label

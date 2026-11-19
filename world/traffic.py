"""Space traffic: AI ships cruising, navigating to systems, warping in/out."""

import logging
import math
import random

from panda3d.core import Quat, Vec3

import config
from core.mathutil import clamp, damp_vec
from core.ship import Spaceship
from core.ship_profiles import random_traffic_profile
from core.trails import Trail
from world.structures.hangar import Hangar

log = logging.getLogger("galaxy_explorer.traffic")


def _align_quat_to_vector(quat, direction, max_angle=None):
    """Rotate quat toward direction, limited to max_angle radians this step."""
    if direction.length() < 1e-6:
        return quat
    d = Vec3(direction)
    d.normalize()
    fwd = quat.getForward()
    if fwd.length() < 1e-6:
        fwd = Vec3(0, 1, 0)
    dot = clamp(fwd.dot(d), -1.0, 1.0)
    angle = math.acos(dot)
    if max_angle is not None:
        angle = min(angle, max(0.0, max_angle))
    if angle < 1e-9:
        return Quat(quat)
    axis = fwd.cross(d)
    if axis.length() < 1e-6:
        axis = quat.getRight()
    axis.normalize()
    delta = Quat()
    delta.setFromAxisAngleRad(angle, axis)
    result = delta * quat
    result.normalize()
    return result


class TrafficShip:
    def __init__(self, base, parent, profile, position, freighter=False,
                 rng=None):
        self.base = base
        self.parent = parent
        self.profile = profile
        self.freighter = freighter
        self.rng = rng if rng is not None else random.Random()
        self.ship = Spaceship(base, profile)
        self.ship.attach_to(parent)
        self.position = Vec3(position)
        self.anchor = Vec3(position)
        self.quat = Quat()
        self.quat.setHpr((self.rng.uniform(0, 360), 0, 0))
        speed = (config.TRAFFIC_FREIGHTER_SPEED if freighter
                 else config.TRAFFIC_CRUISE_SPEED)
        self.speed = speed * self.rng.uniform(0.85, 1.15)
        self.steer = (config.TRAFFIC_FREIGHTER_STEER if freighter
                      else config.TRAFFIC_STEER)
        self.max_turn = (config.TRAFFIC_FREIGHTER_MAX_TURN if freighter
                         else 2.0)
        self.velocity = self.quat.getForward() * self.speed
        self.state = "warp_in"
        self.warp_timer = config.TRAFFIC_WARP_TIME
        self.target = None
        self.repath_timer = self.rng.uniform(0, config.TRAFFIC_REPATH_INTERVAL)
        self.trail = Trail(parent, profile["trail"])
        self.removed = False
        self.visible = True
        self._scale = profile["scale"]
        self.ship.root.setScale(self._scale * 0.2)
        # Freighters carry a dockable hangar bay; while a player is docking
        # or launching the freighter holds position (station-keeping).
        self.docked = False
        self.hangar = Hangar(base, self.ship.root, self) if freighter else None

    def set_lod_visible(self, visible):
        """Hide distant meshes/trails and reset the trail on re-entry."""
        visible = bool(visible)
        if visible == self.visible:
            return
        self.visible = visible
        self.ship.set_visible(visible)
        if visible:
            self.trail.node.show()
            # Avoid one long segment spanning the time we were hidden.
            self.trail.points.clear()
            self.trail.last = None
            self.trail._rebuild()
        else:
            self.trail.node.hide()

    def update(self, dt, systems, galaxy_systems, player_pos=None):
        if self.removed:
            return
        if self.hangar is not None and player_pos is not None:
            self.hangar.update(dt, player_pos)
        if self.docked:
            # Station-keeping: freeze movement while a ship docks/launches.
            self.velocity = Vec3(0.0)
            return
        if self.state == "warp_in":
            self.warp_timer -= dt
            self.ship.root.setScale(min(
                self._scale,
                self._scale * (1.0 - self.warp_timer
                               / config.TRAFFIC_WARP_TIME)))
            self.velocity = damp_vec(self.velocity,
                                     self.quat.getForward() * self.speed,
                                     self.steer, dt)
            if self.warp_timer <= 0:
                self.state = "cruise"
                self.ship.root.setScale(self._scale)
        elif self.state == "warp_out":
            self.warp_timer -= dt
            self.ship.root.setScale(max(
                0.05, self._scale * (self.warp_timer
                                     / config.TRAFFIC_WARP_TIME)))
            self.velocity = damp_vec(self.velocity,
                                     self.quat.getForward()
                                     * config.TRAFFIC_WARP_SPEED,
                                     self.steer * 2, dt)
            if self.warp_timer <= 0:
                self.removed = True
                return
        else:
            # cruise / goto
            self.repath_timer -= dt
            if self.repath_timer <= 0:
                self._pick_intent(systems, galaxy_systems)
                self.repath_timer = (config.TRAFFIC_REPATH_INTERVAL
                                     + self.rng.uniform(-2, 2))
            if self.state == "goto" and self.target is not None:
                desired = self.target - self.position
                if desired.length() < config.TRAFFIC_WAYPOINT_REACH:
                    self.state = "cruise"
                    self.target = None
                else:
                    desired.normalize()
                    self.velocity = damp_vec(self.velocity,
                                             desired * self.speed,
                                             self.steer, dt)
            else:
                # aimless: gentle wander
                if self.target is None:
                    self.target = self.position + Vec3(
                        self.rng.uniform(-200, 200),
                        self.rng.uniform(-200, 200), 0)
                desired = self.target - self.position
                if desired.length() < 60:
                    self.target = None
                else:
                    desired.normalize()
                    self.velocity = damp_vec(self.velocity,
                                             desired * self.speed,
                                             self.steer * 0.5, dt)

            # bounds check -> warp out (relative to the player's area)
            if ((self.position - self.anchor).length()
                    > config.TRAFFIC_SPAWN_RADIUS * 1.3):
                self.state = "warp_out"
                self.warp_timer = config.TRAFFIC_WARP_TIME

        # face velocity
        if self.velocity.length() > 1e-3:
            self.quat = _align_quat_to_vector(self.quat, self.velocity,
                                              self.max_turn * dt)

        self.position += self.velocity * dt
        self.ship.set_transform(self.position, self.quat)
        if self.visible:
            self.trail.update(self.position)

    def _pick_intent(self, systems, galaxy_systems):
        r = self.rng.random()
        if r < 0.35 and systems:
            sys = self.rng.choice(systems)
            pos = getattr(sys, "position", None)
            if pos is not None:
                self.target = Vec3(pos)
                self.state = "goto"
                return
        self.target = self.position + Vec3(
            self.rng.uniform(-300, 300), self.rng.uniform(-300, 300), 0)
        self.state = "cruise"

    def destroy(self):
        if self.hangar is not None:
            self.hangar.destroy()
            self.hangar = None
        try:
            self.trail.destroy()
        except Exception as exc:
            log.debug("[traffic] trail removal failed: %s", exc)
        try:
            self.ship.root.removeNode()
        except Exception as exc:
            log.debug("[traffic] ship removal failed: %s", exc)
        self.removed = True


class TrafficManager:
    def __init__(self, base, parent, mode, galaxy, system_scene, anchor=None,
                 seed=None):
        self.base = base
        self.parent = parent
        self.mode = mode
        self.galaxy = galaxy
        self.system_scene = system_scene
        self.ships = []
        self.anchor = Vec3(anchor) if anchor is not None else Vec3(0.0)
        self.rng = random.Random(seed)
        self._spawn_initial()

    def set_anchor(self, position):
        """Keep spawning/despawning traffic around the player, not the origin."""
        self.anchor = Vec3(position)

    def _spawn_position(self):
        return self.anchor + Vec3(
            self.rng.uniform(-config.TRAFFIC_SPAWN_RADIUS,
                             config.TRAFFIC_SPAWN_RADIUS),
            self.rng.uniform(-config.TRAFFIC_SPAWN_RADIUS,
                             config.TRAFFIC_SPAWN_RADIUS),
            self.rng.uniform(-config.TRAFFIC_SPAWN_RADIUS * 0.35,
                             config.TRAFFIC_SPAWN_RADIUS * 0.35))

    def _spawn_initial(self):
        if self.mode == "galaxy":
            n_ship = config.TRAFFIC_GALAXY_COUNT
            n_freight = config.TRAFFIC_FREIGHTER_COUNT
        else:
            n_ship = config.TRAFFIC_SYSTEM_COUNT
            n_freight = 0
        for _ in range(n_ship):
            prof = random_traffic_profile(self.rng, freighter=False)
            pos = self._spawn_position()
            self.ships.append(TrafficShip(self.base, self.parent, prof, pos,
                                          freighter=False, rng=self.rng))
        for _ in range(n_freight):
            prof = random_traffic_profile(self.rng, freighter=True)
            pos = self._spawn_position()
            self.ships.append(TrafficShip(self.base, self.parent, prof, pos,
                                          freighter=True, rng=self.rng))

    def update(self, dt):
        systems = []
        if self.mode == "galaxy" and self.galaxy is not None:
            systems = list(getattr(self.galaxy, "systems", []))
        sys_positions = []
        for s in systems:
            p = getattr(s, "position", None)
            if p is not None:
                sys_positions.append(p)
        if self.mode != "galaxy" and self.system_scene is not None:
            sys_positions = [Vec3(0, 0, 0)]

        for ts in list(self.ships):
            ts.anchor = self.anchor
            distance = (ts.position - self.anchor).length()
            # A docked freighter must stay visible even if LOD would hide it.
            ts.set_lod_visible(distance <= config.TRAFFIC_LOD_DIST
                               or ts.docked)
            ts.update(dt, sys_positions, systems, player_pos=self.anchor)
            if ts.removed:
                self.ships.remove(ts)
                ts.destroy()
        # maintain counts
        if self.mode == "galaxy":
            target = config.TRAFFIC_GALAXY_COUNT + config.TRAFFIC_FREIGHTER_COUNT
        else:
            target = config.TRAFFIC_SYSTEM_COUNT
        while len(self.ships) < target:
            freighter = (self.mode == "galaxy"
                         and self.rng.random() < 0.2
                         and sum(1 for s in self.ships if s.freighter)
                         < config.TRAFFIC_FREIGHTER_COUNT)
            prof = random_traffic_profile(self.rng, freighter=freighter)
            pos = self._spawn_position()
            ts = TrafficShip(self.base, self.parent, prof, pos,
                             freighter=freighter, rng=self.rng)
            ts.state = "warp_in"
            self.ships.append(ts)

    def nearest_hangar(self, position):
        """The freighter with the closest hangar entry, or None if out of range."""
        best = None
        best_dist = config.HANGAR_ENTRY_RANGE * 1.6
        for ts in self.ships:
            if ts.hangar is None or ts.removed:
                continue
            dist = ts.hangar.entry_distance(position)
            if dist < best_dist:
                best_dist = dist
                best = ts
        return best

    def destroy(self):
        for ts in list(self.ships):
            ts.destroy()
        self.ships.clear()

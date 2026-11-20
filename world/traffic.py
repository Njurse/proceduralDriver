"""Basic traffic: kinematic cars on world.roads.lanes paths.

Up to config.TRAFFIC_MAX cars appear on lanes a ring round the player,
drive at a cruising speed, ease off for sharp turns at junctions, slow and
stop behind the car ahead in their lane, take a random exit at each node
and vanish once far away.  No physics: they follow the lane exactly.
"""

import logging
import math
import random
from dataclasses import dataclass

import numpy as np

import config
from world.roads.lanes import LaneMap, connector

log = logging.getLogger(__name__)

TINTS = ((0.85, 0.2, 0.2), (0.2, 0.4, 0.85), (0.9, 0.85, 0.3), (0.25, 0.7, 0.35),
         (0.9, 0.9, 0.9), (0.3, 0.3, 0.32), (0.8, 0.45, 0.2))


@dataclass(eq=False)
class Car:
    id: int
    lane: object
    s: float
    speed: float
    cruise: float
    tint: tuple
    next: object = None        # the lane picked for after this one

    def pose(self):
        """(position (3,), rotation (3, 3)): columns right, up, forward."""
        pos, fwd = self.lane.at(self.s)
        right = np.array([fwd[2], 0.0, -fwd[0]])
        right /= max(float(np.linalg.norm(right)), 1e-9)
        return pos, np.stack([right, np.cross(fwd, right), fwd], axis=1)


def _turn(a, b):
    """Heading change (radians, 0..pi) from the end of lane a into lane b."""
    da, db = a.pts[-1] - a.pts[-2], b.pts[1] - b.pts[0]
    t = math.atan2(db[2], db[0]) - math.atan2(da[2], da[0])
    return abs((t + math.pi) % (2.0 * math.pi) - math.pi)


class Traffic:
    """The traffic round the player on a world.world.World."""

    def __init__(self, world, seed=0, max_cars=None):
        self.world = world
        self.lanes = LaneMap(world.roads)
        self.rng = random.Random(seed)
        self.max_cars = config.TRAFFIC_MAX if max_cars is None else int(max_cars)
        self.cars = []
        self._ids = 0

    # ------------------------------------------------------------ spawning
    def _spawn(self, px, pz):
        lanes = [ln for ln in self.lanes.lanes_near(px, pz, config.TRAFFIC_SPAWN_M)
                 if ln.length > 2.0 * config.TRAFFIC_GAP_M]
        if not lanes:
            return None
        lane = self.rng.choice(lanes)
        s = self.rng.uniform(config.TRAFFIC_GAP_M, lane.length - config.TRAFFIC_GAP_M)
        pos, _ = lane.at(s)
        d = math.hypot(pos[0] - px, pos[2] - pz)
        if not config.TRAFFIC_SPAWN_MIN_M <= d <= config.TRAFFIC_SPAWN_M:
            return None
        if any(c.lane is lane and abs(c.s - s) < 3.0 * config.TRAFFIC_GAP_M for c in self.cars):
            return None
        self._ids += 1
        cruise = config.TRAFFIC_SPEED_MS * self.rng.uniform(0.8, 1.1)
        car = Car(self._ids, lane, s, cruise, cruise, self.rng.choice(TINTS))
        self.cars.append(car)
        log.debug("[traffic] car %d on %s at %.0f m, %d cars", car.id, lane.key, d,
                  len(self.cars))
        return car

    # ------------------------------------------------------------ driving
    def _gap(self, car):
        """Free road ahead of ``car`` in its lane and the next (metres)."""
        gap = math.inf
        ahead = car.lane.length - car.s
        nxt = car.next or car.lane.to
        for o in self.cars:
            if o is car:
                continue
            if o.lane is car.lane and o.s > car.s:
                gap = min(gap, o.s - car.s)
            elif nxt is not None and o.lane is nxt:
                gap = min(gap, ahead + o.s)
        return gap

    def _pick_next(self, car):
        exits = self.lanes.exits(car.lane)
        if not exits:
            return None
        back = [ln for ln in exits if ln.key[0] == car.lane.key[0]]
        choice = self.rng.choice([ln for ln in exits if ln not in back] or exits)
        return connector(car.lane, choice)

    def step(self, dt, px, pz):
        """Advance every car by dt; despawn far ones and spawn up to the cap."""
        keep = []
        for c in self.cars:
            pos, _ = c.lane.at(c.s)
            if math.hypot(pos[0] - px, pos[2] - pz) <= config.TRAFFIC_DESPAWN_M:
                keep.append(c)
        self.cars = keep
        if len(self.cars) < self.max_cars:
            self._spawn(px, pz)
        for c in self.cars:
            self._drive(c, dt)

    def _drive(self, c, dt):
        lane = c.lane
        if c.next is None and lane.end_node is not None and lane.length - c.s < 30.0:
            c.next = self._pick_next(c)
        want = c.cruise
        if c.next is not None and c.next.to is not None:
            turn = _turn(lane, c.next.to)
            slow = c.cruise * (1.0 - 0.7 * turn / math.pi)
            ahead = lane.length - c.s
            want = min(want, slow + max(ahead - 10.0, 0.0) * 0.5)
        gap = self._gap(c)
        want = min(want, max(0.0, (gap - config.TRAFFIC_GAP_M) * 0.8))
        rate = config.TRAFFIC_ACCEL if want > c.speed else config.TRAFFIC_BRAKE
        c.speed += max(-rate * dt, min(rate * dt, want - c.speed))
        c.s += c.speed * dt
        while c.s > c.lane.length:
            if c.lane.to is not None:            # leaving a junction
                c.s -= c.lane.length
                c.lane, c.next = c.lane.to, None
            elif c.next is not None:
                c.s -= c.lane.length
                c.lane, c.next = c.next, None
            else:                                # nowhere to go: wait at the end
                c.s, c.speed = c.lane.length, 0.0

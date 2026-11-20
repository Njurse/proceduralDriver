"""Draws world.traffic: instances of the player's car model, tinted."""

import logging

from panda3d.core import Mat4, NodePath

from core.mathutil import clamp
from world.traffic import Traffic

log = logging.getLogger(__name__)


class TrafficView:
    """One instanced, colour-scaled copy of the car per traffic car."""

    def __init__(self, parent, car_model, wheel_mounts, ride_m):
        self.root = parent.attachNewNode("traffic")
        self.root.setLightOff(1)
        self.template = NodePath("traffic_car")
        car_model.instanceTo(self.template)
        for mount in wheel_mounts:          # wheels frozen where they are now
            mount.copyTo(self.template)
        self.ride_m = ride_m
        self.nodes = {}

    def sync(self, cars):
        live = set()
        for c in cars:
            node = self.nodes.get(c.id)
            if node is None:
                node = self.nodes[c.id] = self.root.attachNewNode(f"traffic_{c.id}")
                self.template.instanceTo(node)
                node.setColorScale(*c.tint, 1.0)
            pos, rot = c.pose()
            (r0, u0, f0), (r1, u1, f1), (r2, u2, f2) = rot.tolist()
            x, y, z = pos.tolist()
            k = self.ride_m
            # _mat4(PHYS_TO_PANDA @ rot, ...) written out: physics (x, y up, z) -> Panda (x, z, y)
            node.setMat(Mat4(r0, r2, r1, 0.0, u0, u2, u1, 0.0, f0, f2, f1, 0.0,
                             x + u0 * k, z + u2 * k, y + u1 * k, 1.0))
            live.add(c.id)
        for cid in [k for k in self.nodes if k not in live]:
            self.nodes.pop(cid).removeNode()

    def destroy(self):
        self.root.removeNode()
        self.nodes = {}


class TrafficDraw:
    """DriveSim mixin: runs and draws the traffic, remade with the world."""

    traffic = None
    traffic_view = None

    def _update_traffic(self, dt):
        world, v = self.session.world, self.vehicle
        if self.traffic is None or self.traffic.world is not world:
            if self.traffic_view is not None:
                self.traffic_view.destroy()
            self.traffic = Traffic(world, seed=self.session.seed)
            ride = clamp(v.pos[1] - world.height(v.pos[0], v.pos[2]), 0.3, 1.5)
            self.traffic_view = TrafficView(self.render, self.car, self.wheel_mounts, ride)
            log.info("[traffic] up to %d cars, ride height %.2f m", self.traffic.max_cars, ride)
        self.traffic.step(dt, v.pos[0], v.pos[2])
        self.traffic_view.sync(self.traffic.cars)

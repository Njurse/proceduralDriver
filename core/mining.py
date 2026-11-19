"""Lock-on mining: find the nearest vein inside a forward cone."""

import math

import config
from core.mathutil import clamp


class MiningLock:
    """Tracks the currently locked vein and mining progress."""

    def __init__(self, mining_range=None, cone_deg=None):
        self.target = None
        self.last_target = None
        self.progress = 0.0
        self.range = (config.MINING_RANGE if mining_range is None
                      else mining_range)
        self.half_angle = math.radians(
            config.MINING_CONE_DEG if cone_deg is None else cone_deg)

    def update(self, ship_pos, ship_quat, veins, dt, mining_held,
               tool_speed=1.0):
        forward = ship_quat.getForward()
        best = None
        best_dist = self.range
        for v in veins:
            if v.depleted:
                continue
            to = v.position - ship_pos
            dist = to.length()
            if dist > self.range or dist < 1e-3:
                continue
            cosang = (to.dot(forward)) / (dist * forward.length())
            if cosang < math.cos(self.half_angle):
                continue
            if dist < best_dist:
                best_dist = dist
                best = v
        # Reset progress when the locked vein changes.
        if best is not self.last_target:
            self.progress = 0.0
        self.last_target = best
        self.target = best

        if mining_held and self.target is not None:
            hardness = max(1, int(getattr(self.target, "hardness", 1)))
            self.progress += dt * tool_speed / (config.MINING_TIME * hardness)
            if self.progress >= 1.0:
                depleted = self.target.damage(1.0)
                self.progress = 0.0
                if depleted:
                    return self.target
        else:
            self.progress = clamp(self.progress - dt * 2.0, 0.0, 1.0)
        return None

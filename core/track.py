"""Track / ground / collision primitives for the driving demo.

Flat ground by default; OBB wall primitives + 2D SAT (bcollide.c style) for
walls/buildings.  Car-car collision is stubbed.
"""

import math


def surface_at(x, z):
    """Return (height, normal_x, normal_z, surface_type). Flat ground."""
    return (0.0, 0.0, 1.0, 0)  # height=0, normal up, concrete


class OBBWall:
    def __init__(self, cx, cz, hx, hz, angle=0.0):
        self.cx = cx
        self.cz = cz
        self.hx = hx  # half-extents
        self.hz = hz
        self.angle = angle

    def axes(self):
        c = math.cos(self.angle)
        s = math.sin(self.angle)
        return ((c, s), (-s, c))

    def corners(self):
        ax, az = self.axes()[0]
        bx, bz = self.axes()[1]
        pts = []
        for sx in (-1, 1):
            for sz in (-1, 1):
                px = self.cx + (ax * self.hx * sx + bx * self.hz * sz)
                pz = self.cz + (az * self.hx * sx + bz * self.hz * sz)
                pts.append((px, pz))
        return pts


def obb_overlap(a, b):
    """2D SAT overlap test between two OBBs. Returns (overlap, nx, nz, depth)."""
    axes = list(a.axes()) + list(b.axes())
    min_overlap = float("inf")
    min_axis = (1.0, 0.0)
    for ax, az in axes:
        # project a
        amin = min(p[0] * ax + p[1] * az for p in a.corners())
        amax = max(p[0] * ax + p[1] * az for p in a.corners())
        bmin = min(p[0] * ax + p[1] * az for p in b.corners())
        bmax = max(p[0] * ax + p[1] * az for p in b.corners())
        if amax < bmin or bmax < amin:
            return False, 0.0, 0.0, 0.0
        overlap = min(amax, bmax) - max(amin, bmin)
        if overlap < min_overlap:
            min_overlap = overlap
            min_axis = (ax, az)
    return True, min_axis[0], min_axis[1], min_overlap


class Track:
    def __init__(self):
        self.walls = []

    def add_wall(self, wall):
        self.walls.append(wall)

    def collide_car(self, car_x, car_z, car_hx, car_hz, car_angle):
        """Return list of (nx, nz, depth) corrections."""
        cb = OBBWall(car_x, car_z, car_hx, car_hz, car_angle)
        corrections = []
        for w in self.walls:
            hit, nx, nz, depth = obb_overlap(cb, w)
            if hit and depth > 0:
                corrections.append((nx, nz, depth))
        return corrections


if __name__ == "__main__":
    t = Track()
    t.add_wall(OBBWall(50, 0, 5, 20, 0))
    print(t.collide_car(40, 0, 4, 2, 0))
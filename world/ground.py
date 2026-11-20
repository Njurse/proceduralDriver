"""What the car drives on.

The physics asks the ground one question per contact point (core.car_forces):
``ground(x, z) -> (height, unit normal, surface)`` in the physics frame
(metres, +y up).  ``Ground`` names that contract and splits it into three
overridable parts, so a later layer (roads, a 70s/80s city) can wrap the
terrain and replace just the height, normal or surface where it builds.
``SampledGround`` derives all of it from one analytic height field.
"""

import math
from typing import Protocol, runtime_checkable

import numpy as np

from core.car_forces import SURFACE_ROAD


@runtime_checkable
class Ground(Protocol):
    def height(self, x: float, z: float) -> float:
        """Ground height (m) under the point (x, z)."""

    def normal(self, x: float, z: float) -> np.ndarray:
        """Unit surface normal (3,) at (x, z), +y up."""

    def surface(self, x: float, z: float) -> int:
        """core.car_forces SURFACE_* code at (x, z)."""

    def __call__(self, x: float, z: float) -> tuple[float, np.ndarray, int]:
        """The physics query: (height, normal, surface)."""


class GroundBase:
    """Builds the physics query from height/normal/surface.

    Subclasses override those three; a subclass that can produce height and
    normal together more cheaply overrides ``__call__`` as well.
    """

    def height(self, x: float, z: float) -> float:
        raise NotImplementedError

    def normal(self, x: float, z: float) -> np.ndarray:
        raise NotImplementedError

    def surface(self, x: float, z: float) -> int:
        return SURFACE_ROAD

    def __call__(self, x: float, z: float) -> tuple[float, np.ndarray, int]:
        return self.height(x, z), self.normal(x, z), self.surface(x, z)


class SampledGround(GroundBase):
    """Ground from ``sample(x, z) -> (height, dh/dx, dh/dz)``.

    ``sample`` takes scalars (the physics path) or numpy arrays (the mesh
    path); height and normal both come from it, so they always agree.
    """

    def sample(self, x, z):
        raise NotImplementedError

    def height(self, x, z):
        return self.sample(x, z)[0]

    def normal(self, x, z):
        """Unit normal; shape (3,) for scalars, (..., 3) for arrays."""
        _, dhx, dhz = self.sample(x, z)
        n = np.stack(np.broadcast_arrays(-np.asarray(dhx), 1.0, -np.asarray(dhz)), axis=-1)
        return n / np.linalg.norm(n, axis=-1, keepdims=True)

    def __call__(self, x, z):
        h, dhx, dhz = self.sample(x, z)
        inv = 1.0 / math.sqrt(dhx * dhx + 1.0 + dhz * dhz)
        return h, np.array([-dhx * inv, inv, -dhz * inv]), self.surface(x, z)

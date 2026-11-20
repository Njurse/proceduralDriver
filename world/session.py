"""One play session without any rendering: world, car, discovery, save file.

``Session`` owns what the pause menu acts on (pause, restart, a new seed,
saving) so it can be tested without a window; main.DriveSim draws it.

Save policy:
  * no save path (offscreen runs, tests): nothing is read or written;
  * otherwise saves/world.json is loaded when it exists and the seed agrees;
  * ``--seed N`` / ``--new`` that disagree with an existing save start fresh
    and write to a new file beside it (world_2.json, ...), never over it;
  * a seed change from the menu (after the player confirms) keeps a copy of
    the old file as world_<old seed>.json before the new world takes over.
"""

import logging
import math
import random
import shutil
from pathlib import Path

import config
from core.car_profile import load_cosmetics
from core.redriver2_physics import PhysicsWorld, Vehicle
from world.chunks import chunks_around
from world.save import SaveState, free_path, read_save, write_save
from world.world import World

log = logging.getLogger(__name__)


class Explorer:
    """The chunks seen so far: everything within ``radius`` chunks of the
    car (the streamed draw distance), and the biomes in them."""

    def __init__(self, world, radius=None, discovered=None, biomes=None):
        self.world = world
        self.radius = int(config.TERRAIN_VIEW_CHUNKS if radius is None else radius)
        self.discovered = set(discovered or ())
        self.biomes = set(biomes or ())
        self._last = None

    def visit(self, x, z):
        """Mark the chunks round (x, z); returns the newly discovered ones."""
        size = config.TERRAIN_CHUNK_SIZE
        here = (math.floor(x / size), math.floor(z / size))
        if here == self._last:
            return []
        self._last = here
        new = [k for k in chunks_around(x, z, self.radius) if k not in self.discovered]
        for ci, ck in new:
            self.discovered.add((ci, ck))
            self.biomes.add(self.world.regions.biome_at((ci + 0.5) * size,
                                                        (ck + 0.5) * size).name)
        return new


class Session:
    def __init__(self, save_path=None, seed=None, new=False, cosmetics=None):
        self.save_path = None if save_path is None else Path(save_path)
        state = None
        if self.save_path is not None and not new:
            state = read_save(self.save_path)
        if state is not None and seed is not None and int(seed) != state.seed:
            state = None
        if state is None and self.save_path is not None and self.save_path.exists():
            self.save_path = free_path(self.save_path)  # keep the other world's save
        if seed is None:
            seed = (state.seed if state is not None
                    else random.randrange(1 << 31) if new else config.TERRAIN_SEED)
        self.cos = load_cosmetics() if cosmetics is None else cosmetics
        self.vehicle = Vehicle(self.cos)
        self.physics = PhysicsWorld()
        self.physics.add(self.vehicle)
        self.paused = False
        self._build(int(seed), state)
        log.info("seed %d: %s (save file: %s)", self.seed,
                 "save loaded" if state is not None else "new world", self.save_path or "off")

    def _build(self, seed, state=None):
        self.seed = seed
        self.world = World(seed=seed)
        self.explorer = Explorer(self.world,
                                 discovered=state.discovered if state else None,
                                 biomes=state.biomes if state else None)
        if state is not None and state.car is not None:
            self.vehicle.reset(*state.car, ground=self.world)
        else:
            self.vehicle.reset(*self.world.spawn(), ground=self.world)
        self.physics._accumulator = 0.0
        self.explorer.visit(self.vehicle.pos[0], self.vehicle.pos[2])

    # ---------------------------------------------------------------- play
    def step(self, inp, dt):
        """Advance the car; does nothing while paused.  Returns the ticks run."""
        if self.paused:
            return 0
        self.vehicle.process_input(inp)
        ticks = self.physics.step(dt)
        self.explorer.visit(self.vehicle.pos[0], self.vehicle.pos[2])
        return ticks

    def set_paused(self, paused):
        self.paused = bool(paused)
        if self.paused:
            self.save()

    def restart(self):
        """Respawn on the road nearest the world's spawn point, same seed."""
        self.vehicle.reset(*self.world.spawn(), ground=self.world)
        self.physics._accumulator = 0.0

    def change_seed(self, seed):
        """Rebuild the world for ``seed`` with nothing discovered.  The menu
        confirms first; the old save is copied to world_<old seed>.json."""
        if self.save_path is not None and self.save_path.exists():
            backup = free_path(self.save_path.with_name(
                f"{self.save_path.stem}_{self.seed}{self.save_path.suffix}"))
            shutil.copy2(self.save_path, backup)
        self._build(int(seed))
        self.save()

    # ---------------------------------------------------------------- save
    def state(self):
        v = self.vehicle
        return SaveState(seed=self.seed, discovered=set(self.explorer.discovered),
                         biomes=set(self.explorer.biomes),
                         car=(float(v.pos[0]), float(v.pos[2]), float(v.heading)))

    def save(self):
        """Write the save file (atomically); False when saving is off."""
        if self.save_path is None:
            return False
        write_save(self.save_path, self.state())
        log.info("saved %s (%d chunks discovered)", self.save_path, len(self.explorer.discovered))
        return True

"""--shots: camera test screenshots of a running DriveSim, offscreen."""

import logging
import math
import os

import numpy as np
from panda3d.core import (
    Filename,
    Point3,
    Vec3,
)

import config
from core.orbit_camera import OrbitCamera
from game.car_view import PHYS_TO_PANDA
from game.chase_camera import CAMERA_NEAR_M
from world.worldmap import export_world

log = logging.getLogger(__name__)


# --shots: (file name, orbit angle in degrees) after a short run-up
SHOTS = (("world_chase", 0.0), ("world_left_side", -90.0), ("world_right_side", 90.0),
         ("world_front_looking_back", 180.0))


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

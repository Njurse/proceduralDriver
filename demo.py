"""Pygame driving demo for the REDRIVER2 handling model.

Controls:
  W/S        throttle / brake
  A/D        steer left/right
  LShift     fast steer
  Space      handbrake
  Esc        quit
"""

import math
import os
import sys

import config

try:
    import pygame
except Exception as e:
    print("pygame required:", e)
    sys.exit(1)


from core.car_profile import load_cosmetics
from core.redriver2_physics import PhysicsWorld, Vehicle
from core.track import OBBWall, Track
from core.vehicle_audio import AudioManager, sound_input


def make_input(keys):
    inp = type("I", (), {})()
    throttle = 1.0 if keys[pygame.K_w] else 0.0
    brake = 1.0 if keys[pygame.K_s] else 0.0
    steer = 0.0
    if keys[pygame.K_a]:
        steer -= 1.0
    if keys[pygame.K_d]:
        steer += 1.0
    inp.steer = steer
    inp.throttle = throttle
    inp.brake = brake
    inp.handbrake = bool(keys[pygame.K_SPACE])
    inp.fast_steer = bool(keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT])
    inp.wheelspin = False
    return inp


def project_car_points(cos, scale=0.05):
    """Project the OBJ mesh to 2D (X-Z plane) for top-down rendering.

    Uses invert_vertical flag already applied during load (Y flipped).  We
    take X,Z coordinates and center/scale them.  Returns list of (x,z) for
    outline (convex-ish) by using all vertices; rendering connects them as a
    point cloud-ish polygon via a simple hull-free approach (draw lines between
    consecutive face edges).
    """
    pts = [(v[0], v[2]) for v in cos.mesh_vertices]
    if not pts:
        return []
    # center
    xs = [p[0] for p in pts]
    zs = [p[1] for p in pts]
    cx = (min(xs) + max(xs)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    out = [((p[0] - cx) * scale, (p[1] - cz) * scale) for p in pts]
    return out, (cx, cz)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="run N frames then exit (headless check)")
    ap.add_argument("--offscreen", action="store_true", help="use dummy video driver")
    args = ap.parse_args()
    if args.offscreen:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    screen = pygame.display.set_mode((config.WINDOW_WIDTH, config.WINDOW_HEIGHT))
    pygame.display.set_caption("REDRIVER2 handling demo")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 18)

    cos = load_cosmetics()
    car_pts = project_car_points(cos, scale=1.0)  # OBJ is in metres

    track = Track()
    # a few walls to bump into
    track.add_wall(OBBWall(60, 0, 4, 20, 0.0))
    track.add_wall(OBBWall(-60, 0, 4, 20, 0.0))
    track.add_wall(OBBWall(0, 60, 20, 4, 0.0))
    track.add_wall(OBBWall(0, -60, 20, 4, 0.0))

    vehicle = Vehicle(cos)
    vehicle.reset(0, 0, 0)
    world = PhysicsWorld(track)
    world.add(vehicle)

    audio = AudioManager(cos)

    running = True
    frame_count = 0
    while running:
        dt = clock.tick(config.TARGET_FPS) / 1000.0
        if dt <= 0:
            dt = 1.0 / 60.0
        dt = min(dt, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        inp = make_input(keys)
        vehicle.process_input(inp)
        world.step(dt)

        audio.update(dt, sound_input(vehicle))

        # render
        screen.fill((20, 24, 30))
        # ground grid
        cx0 = config.WINDOW_WIDTH // 2
        cy0 = config.WINDOW_HEIGHT // 2
        scale_px = 8.0  # world units -> pixels
        # draw walls
        for w in track.walls:
            corners = w.corners()
            pts = [(int(cx0 + c[0] * scale_px), int(cy0 + c[1] * scale_px)) for c in corners]
            pygame.draw.polygon(screen, (60, 60, 70), pts, 2)

        # car
        yaw = vehicle.heading  # C convention: forward = (sin, cos) in (x, z)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        vx = vehicle.pos[0]
        vz = vehicle.pos[2]
        car_screen_pts = []
        for (mx, mz) in car_pts[0]:
            lx, lz = -mx, -mz  # OBJ model space is the physics body frame negated
            rx = lx * cos_y + lz * sin_y
            rz = -lx * sin_y + lz * cos_y
            sx = int(cx0 + (vx + rx) * scale_px)
            sz = int(cy0 + (vz + rz) * scale_px)
            car_screen_pts.append((sx, sz))
        if len(car_screen_pts) > 2:
            # draw as point cloud (vertices)
            for p in car_screen_pts:
                pygame.draw.circle(screen, (200, 220, 240), p, 2)
            # draw a simple hull: connect sorted by angle? just draw edges from faces not available in pts.
            # Draw a bounding-ish outline using min/max? Instead draw center marker.
        # center + heading
        csx = int(cx0 + vx * scale_px)
        csz = int(cy0 + vz * scale_px)
        pygame.draw.circle(screen, (255, 200, 80), (csx, csz), 5)
        hx = int(csx + math.sin(yaw) * 20)
        hz = int(csz + math.cos(yaw) * 20)
        pygame.draw.line(screen, (255, 200, 80), (csx, csz), (hx, hz), 2)

        # debug overlay
        lines = [
            f"pos=({vx:.1f},{vz:.1f}) speed={vehicle.speed:.2f} heading={math.degrees(yaw):.1f}",
            f"throttle={vehicle.throttle_input:.2f} brake={vehicle.brake_input:.2f} steer={vehicle.steer_input:.2f}",
            f"wheel_angle={vehicle.wheel_angle} handbrake={vehicle.handbrake} auto_brake={vehicle.auto_brake}",
            f"thrust={vehicle.thrust} gear={vehicle.gear + 1} rpm={audio.mix.rpm:.0f}",
            f"engine={audio.mix.engine_level:.2f} idle={audio.mix.idle_level:.2f} "
            f"skid={audio.mix.skid_level:.2f} load={audio.mix.load:.2f}",
            f"fps={clock.get_fps():.1f}  verts={len(cos.mesh_vertices)} invert_vertical={cos.invert_vertical}",
            "W/S throttle/brake  A/D steer  LShift fast steer  Space handbrake  Esc quit",
        ]
        for i, ln in enumerate(lines):
            surf = font.render(ln, True, (220, 220, 220))
            screen.blit(surf, (10, 10 + i * 18))

        # per-wheel texture labels
        for i, w in enumerate(vehicle.wheels):
            lbl = "FAST" if w.texture_fast else "clean"
            surf = font.render(f"w{i}:{lbl} ws={w.wheel_speed:.1f}", True, (180, 180, 180))
            screen.blit(surf, (10, 10 + (len(lines) + i) * 18))

        pygame.display.flip()
        frame_count += 1
        if args.frames and frame_count >= args.frames:
            running = False

    audio.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()

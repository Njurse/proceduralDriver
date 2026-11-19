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
from core.redriver2_physics import Vehicle, PhysicsWorld
from core.track import Track, OBBWall
from core.vehicle_audio import AudioManager, VehicleAudioState


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
    car_pts = project_car_points(cos, scale=0.06)

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
    audio_state = VehicleAudioState()

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

        # update audio state
        audio_state.speed = vehicle.speed
        audio_state.throttle = vehicle.throttle_input
        audio_state.handbrake = vehicle.handbrake
        audio_state.wheelspin = vehicle.wheelspin
        audio_state.wheel_speeds = [w.wheel_speed for w in vehicle.wheels]
        audio_state.slip_lateral = vehicle.wheels[0].slip_lateral
        audio_state.slip_longitudinal = vehicle.wheels[0].slip_longitudinal
        audio_state.locked = [w.locked for w in vehicle.wheels]
        audio.update(dt, audio_state)

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
        yaw = vehicle.yaw_rad
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        _pm = vehicle.position_m
        vx = _pm[0]
        vz = _pm[2]
        car_screen_pts = []
        for (lx, lz) in car_pts[0]:
            # rotate local point by yaw
            rx = lx * cos_y - lz * sin_y
            rz = lx * sin_y + lz * cos_y
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
        hx = int(csx + fx * 20) if False else int(csx + math.cos(yaw) * 20)
        hz = int(csz + math.sin(yaw) * 20)
        pygame.draw.line(screen, (255, 200, 80), (csx, csz), (hx, hz), 2)

        # debug overlay
        lines = [
            f"pos=({vx:.1f},{vz:.1f}) speed={vehicle.speed:.2f} yaw={vehicle.orientation}",
            f"throttle={vehicle.throttle_input:.2f} brake={vehicle.brake_input:.2f} steer={vehicle.steer_input:.2f}",
            f"wheel_angle={vehicle.wheel_angle} handbrake={vehicle.handbrake} auto_brake={vehicle.auto_brake}",
            f"thrust={vehicle.thrust}",
            f"engine_vol={audio.engine_vol:.2f} idle_vol={audio.idle_vol:.2f} skid_vol={audio.skid_vol:.2f} skid={audio.skid_active}",
            f"fps={clock.get_fps():.1f}  verts={len(cos.mesh_vertices)} invert_vertical={cos.invert_vertical}",
            "W/S throttle/brake  A/D steer  LShift fast steer  Space handbrake  Esc quit",
        ]
        for i, ln in enumerate(lines):
            surf = font.render(ln, True, (220, 220, 220))
            screen.blit(surf, (10, 10 + i * 18))

        # per-wheel texture labels
        for i, w in enumerate(vehicle.wheels):
            lbl = "FAST" if w.texture_fast else "clean"
            surf = font.render(f"w{i}:{lbl} ws={w.wheel_speed}", True, (180, 180, 180))
            screen.blit(surf, (10, 10 + (len(lines) + i) * 18))

        pygame.display.flip()
        frame_count += 1
        if args.frames and frame_count >= args.frames:
            running = False

    audio.shutdown()
    pygame.quit()


if __name__ == "__main__":
    main()
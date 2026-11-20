# Procedural Driver

A small driving game: a float port of the REDRIVER2 (Driver 2) car handling,
rendered with Panda3D.  You drive the Havana car over endless, seeded rolling
hills (Perlin fBm), streamed in 64 m chunks round the car.

## Run

```
pip install -r requirements.txt
python main.py                          # windowed
python main.py --offscreen --frames 600 # headless smoke run
python main.py --shots screenshots      # save camera test shots (offscreen)
```

## Controls

| Keyboard | Gamepad | Action |
|---|---|---|
| W | right trigger | throttle |
| S | left trigger | brake, then reverse once stopped |
| A / D | left stick | steer |
| Left Shift | right shoulder | fast steer |
| Space | A | handbrake |
| I / J / K / L | right stick | swing the camera round the car (springs back) |
| O | R3 | latch the current camera angle |
| Esc | | quit |

## Layout

```
main.py          entry point: DriveSim (window, world, HUD, audio, frame loop), CLI
game/            DriveSim mixins: car_view.py (car model, wheels, pose sync),
                 chase_camera.py, controls.py (keys, gamepad, menu nav), shots.py (--shots),
                 traffic_view.py (draws the traffic cars), world_view.py (streams chunks)
config.py        paths and tuning (terrain, camera, gearbox, audio)
worldgen/        the world generator, numpy only (no Panda3D, no config): spec (WorldSpec),
                 hashing, biomes, height, regions, palette, ground (the Ground protocol
                 the physics queries), roads/ (layout, road, mesh, lanes), chunk (meshes),
                 query (WorldGround: height, slope, surface, road strips), traffic
world/           the game's world: spec.py (WorldSpec and TrafficSpec from config),
                 world.py (World: autopilot, spawn), session, save, sky, worldmap
core/            physics (redriver2_physics, car_forces, engine, fixedpoint, quat),
                 audio (engine_sound, vehicle_audio), camera, gamepad + xinput, textures
ui/              pause menu, minimap
tests/           pytest suite, including a golden trajectory
tools/           dr2_fixed_emu.py: integer emulator of the C handling
assets/          Havana car model, wheels, handling.json and sounds
```

World knobs live in `config.py` (`TERRAIN_SEED` as the default seed, chunk
size and resolution, view radius, region, road and traffic settings) and
reach `worldgen` only through the `WorldSpec` and `TrafficSpec` that
`world/spec.py` builds; terrain shape comes from the biome files.  `worldgen`
never reads `config.py`, so it can be lifted into another engine as it is.

## Roads and traffic

Each biome (`assets/world/definitions/*.json`) sets its road rules: `grid`
(0 organic .. 1 Manhattan), `grid_spacing_m`, `curviness_m`, `loop_odds` (how
many blocks close), `l_corner_odds` with `corner_radius_m` (0 for a sharp L
corner, more for a rounded one) and a top-level `region_size` (1-6: how
many region sites the biome claims as one block).  `metropolis` is a big
dense grid; `british_town` is winding, looping streets.  No two roads leave
a junction closer than `ROAD_MIN_JUNCTION_DEG` (45).

Traffic is kinematic: cars follow lanes (`worldgen/roads/lanes.py`, right-hand
side, junction connectors) round the player, keep their gap to the car ahead
and pick a random exit at each node.  They do not collide with the player.
Knobs are the `TRAFFIC_*` values in `config.py`.

Deviations from the C are marked `DEVIATION` in the code; see also
`PORTING_NOTES.md`.

## Tests

```
python -m pytest -q
python -m ruff check .
```

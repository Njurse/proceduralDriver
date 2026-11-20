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
                 chase_camera.py, controls.py (keys, gamepad, menu nav), shots.py (--shots)
config.py        paths and tuning (terrain, camera, gearbox, audio)
world/           ground.py: the Ground protocol the physics queries;
                 terrain.py: seeded Perlin fBm height/normal; chunks.py: chunk meshes;
                 regions/biomes, roads/ (layout, road, mesh), session, save, sky, worldmap
core/            physics (redriver2_physics, car_forces, engine, fixedpoint, quat),
                 audio (engine_sound, vehicle_audio), camera, gamepad + xinput, textures
ui/              pause menu, minimap
tests/           pytest suite, including a golden trajectory
tools/           dr2_fixed_emu.py: integer emulator of the C handling
assets/          Havana car model, wheels, handling.json and sounds
```

Terrain knobs live in `config.py` (`TERRAIN_SEED`, `_SCALE`, `_OCTAVES`,
`_AMPLITUDE`, chunk size, view radius).  A later layer (roads, a city) can
wrap `Terrain` as a `world.ground.Ground` and override height, normal or
surface where it builds.

Deviations from the C are marked `DEVIATION` in the code; see also
`PORTING_NOTES.md`.

## Tests

```
python -m pytest -q
python -m ruff check .
```

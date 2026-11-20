# Procedural Driver

A small driving game: a float port of the REDRIVER2 (Driver 2) car handling,
rendered with Panda3D.  You drive the Havana car around a flat asphalt plane;
this is the clean slate for a procedural world.

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
main.py          entry point: Panda3D scene, input, chase camera, HUD
config.py        paths and tuning (camera, gearbox, audio)
core/            physics (redriver2_physics, car_forces, engine, fixedpoint, quat),
                 audio (engine_sound, vehicle_audio), camera, gamepad, textures
tests/           pytest suite, including a golden trajectory
tools/           dr2_fixed_emu.py: integer emulator of the C handling
assets/          Havana car model, wheels, handling.json and sounds
```

Deviations from the C are marked `DEVIATION` in the code; see also
`PORTING_NOTES.md`.

## Tests

```
python -m pytest -q
python -m ruff check .
```

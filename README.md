# Procedural Galaxy Explorer

A small No Man's Sky-flavoured exploration game built with **Panda3D**. Fly one
of four ships through a seeded 512-system galaxy, enter star systems, descend
into planet atmospheres, land, swap ships, mine resources, and fast-travel with
the pulse engine. Every world transition plays a one-second rainbow-spiral /
fBm dissolve.

The project is intentionally split into small, single-purpose modules: the
entry point is thin, `game.py` only wires the pieces together, and all flight,
rendering, generation, input and UI code lives under `core/` and `world/`.

---

## Features

- **Seeded procedural galaxy**: 512 star systems on a four-arm spiral, each
  with a star, 3-4 orbiting planets, names, archetypes, palettes and weather.
- **Four flyable ships** (`sparrow`, `talon`, `wraith`, `vortex`) with unique
  models, trail colours, engine sounds and handling. Swap ships by landing next
  to one, or cycle them from the pause menu's **Ship** entry.
- **Freighter hangars**: slow-turning freighters carry a dockable hangar bay
  hanging under the hull. Fly into the humming glowing blue entry rectangle to
  enter the room, approach the landing pad (pulsing green guidance cone), and
  press `X`/`F` to engage the landing autopilot: the ship locks onto the pad,
  descends, rests for half a second, then smoothly rotates to the pad
  orientation. Hold throttle to launch — the takeoff autopilot lifts the ship
  and flies it out through the entry, handing back control in space. The
  freighter holds position (station-keeps) while you are docked.
- **Layered warp sound**: the system-transition whoosh is now a four-layer
  cinematic sound (sub-bass dive, dual-band noise riser, rising sweep, terminal
  impact) for a deeper, more dramatic feel. A looping freighter-hangar hum
  fades in by proximity.
- **Dynamic chase camera**: the camera pulls back and widens its FOV during
  boost and high-speed flight, banks slightly with the ship in atmosphere, and
  tracks more responsively during turns.
- **Two flight models**:
  - **Space**: persistent throttle with reverse, body-relative pitch/yaw/roll,
    boost, free 6DOF rotation, and a soft galaxy boundary.
  - **Atmosphere**: plane-style pitch and banked turns with rudder, automatic
    terrain clearance, a ceiling, and no strafing.
- **Pulse engine fast travel** in galaxy/system space: hold boost to start a
  3-2-1 countdown, then travel along a locked trajectory at 8x boosted speed.
  Brake to pull out instantly with an additive flash and FOV rebound.
- **Landing and liftoff assist**: brake, descend gently and align the ship's up
  axis to the local surface normal; hold throttle to take off again.
- **Procedural planets**: palette-tinted heightfield terrain, water, sky dome,
  two scrolling cloud layers, wind particles and distance fog.
- **Mining and inventory**: 12 original materials across four tiers, procedural
  ore veins scattered with Poisson-disk sampling, cone lock-on mining, tiered
  hardness and yield, an automatically upgrading mining tool, and a grid
  inventory. Depleted veins show a tier-coloured haul message.
- **Third-person chase camera** that follows the ship's orientation smoothly
  and keeps the controller in control when one is connected.
- **Fully responsive 2D UI**: HUD, map, pause menu, inventory, pulse effects and
  the transition card all reflow to the real `aspect2d` bounds, including
  ultrawide displays, and re-check their layout every frame.
- **Galaxy map tools**: name search, visited-only filter, zoom, pan and marker
  clustering at wide zoom levels.
- **Keyboard + mouse look + gamepad** (pygame-ce / XInput, optional), with
  configurable stick/mouse sensitivity, invert-Y and gamepad deadzone.
- **Crash-safe JSON saves**: validation on load, atomic writes with `fsync`,
  a `.bak` of the previous save, and autosave. Player preferences live in
  `saves/settings.json` so they survive deleting the save.
- **F1 help overlay** with the complete control reference.
- **Headless smoke mode** with pixel-statistics validation.

---

## Requirements

- Python 3.10+ (tested on 3.14)
- A GPU/OpenGL driver for normal play. Headless modes use Panda3D's offscreen
  pipe.
- The bundled assets:
  - `assets/models/spaceship/spaceship.glb` (Sparrow)
  - `assets/models/spaceship/spaceship_t2.glb` (Talon)
  - `assets/models/spaceship/spaceship_t3.glb` (Wraith)
  - `assets/models/spaceship/spaceship_t4.glb` (Vortex)
  - `assets/models/spaceship/spaceship_f1.glb` (Freighter traffic)
  - `assets/sounds/spaceship/engine_1.wav` ... `engine_4.wav`,
    `engine_freighter.wav`

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

`pygame-ce` is only used to poll a gamepad; the game runs without a controller
or even without pygame installed.

---

## Running

```powershell
# Continue the existing save, or start a new random galaxy if none exists.
python main.py

# Ignore the save and generate a fresh galaxy.
python main.py --new

# Reproduce/generate a specific galaxy.
python main.py --new --seed 424242

# Render one frame of each scene, validate pixel coverage, and exit.
python main.py --smoke --seed 424242

# Run headless for a fixed number of frames (no save is written).
python main.py --offscreen --frames 10
```

A normal window is 1280x720. The window title shows the current seed.

---

## Controls

### Keyboard and mouse

| Action | Key |
| --- | --- |
| Throttle forward / reverse | `W` / `S` |
| Pitch up / down | `Up` / `Down` or mouse up/down |
| Yaw left / right | `Left` / `Right` or mouse left/right |
| Roll | `A` / `D` |
| Boost / hold to charge pulse engine | `Shift` |
| Brake / instant pulse pullout | `Space` |
| Land near surface / lift off (hold throttle) | `X` / `F` |
| Exit ship / enter ship on foot (when landed) | `E` |
| Walk / strafe on foot | `W`/`A`/`S`/`D` |
| Sprint on foot | `Shift` |
| Jump on foot | `Space` |
| Mine aimed vein on foot (hold) | `R` or mouse left button |
| Mine locked vein (hold) | `R` or mouse left button |
| Open/close inventory | `I` |
| Interact: enter system, descend, leave system | `E` |
| Swap into a landed ship (hold ~1 second) | `E` |
| Galaxy map | `M` or `Tab` |
| Pause menu | `P` |
| Full controls reference | `F1` |
| Pause/options navigate | `W`/`S` or arrow keys, `A`/`D` or left/right to adjust |
| Confirm / select | `Enter` or `E` |
| Back / close overlay | `Esc` |
| Quit | `Ctrl+Q` or window close |

`Esc` is context sensitive: it closes the map or inventory, returns from the
options page, releases the mouse when the pointer is captured, and opens the
pause menu when the cursor is free. Click the window to recapture the pointer
for mouse look.

All flight axes are **body-relative**: pitch rotates about the ship's right
axis, roll about its forward axis, and yaw about its up axis, regardless of
which way the ship is facing. There is no Euler coupling or gimbal lock.

### Gamepad

The first connected joystick is detected automatically and re-scanned every few
seconds. When a controller is connected it takes control precedence for flight
(mouse look and keyboard flight axes are suppressed) while menu actions remain
available.

| Action | Input |
| --- | --- |
| Throttle | Right trigger / left trigger |
| Pitch | Left stick up/down |
| Yaw | Right stick left/right or `LB` / `RB` bumpers |
| Roll | Left stick left/right |
| Boost / hold to charge pulse engine | `Y` |
| Brake / instant pulse pullout | `B` |
| Land near surface / lift off (hold throttle) | `X` |
| Mine locked vein (hold) / interact / confirm | `A` |
| Swap into a landed ship (hold) | `A` |
| Galaxy map | `Select` |
| Pause menu / resume | `Start` |
| Pause/options navigate | D-pad up/down, left/right to adjust |
| Map selection | D-pad left/right |
| Back / close overlay | `B` or `Select` |

Analogue stick steering is scaled by `ANALOGUE_SENSITIVITY` in `config.py` and
by the **Stick Sensitivity** slider in Options. Mouse look has its own
**Mouse Sensitivity** slider, and **Invert Look Y** affects both mouse and stick
pitch. XInput controllers are auto-detected on Windows (with a pygame joystick
fallback). Controller detection scans every connected device and picks the
first one that initializes cleanly.

### Galaxy map

| Action | Key |
| --- | --- |
| Select next / previous system | `A`/`D` or arrow keys |
| Warp to selection | `Enter` (`E` also works when not searching) |
| Search by name | `/`, then type letters/digits; `Backspace` to delete |
| Apply search / select first match | `Enter` |
| Clear search | `Esc` |
| Show only visited systems | `V` |
| Zoom | Mouse wheel or `+` / `-` |
| Pan | `Shift` + arrow keys |
| Reset zoom and pan | `Home` |
| Close map | `M` / `Esc` |

With 512 systems, markers are collapsed into clusters when zoomed out. Labels
show for the selection, the current system, visited systems, and all results of
a search.

### Pause menu

`P` (keyboard), `Start` (gamepad) or `Esc` (with the cursor free) opens the
pause overlay. The main menu provides:

1. **Continue** - unpause and keep flying.
2. **Inventory** - open the inventory grid.
3. **Ship** - cycle to the next flyable ship (useful while testing).
4. **Options** - open the settings submenu.
5. **Exit** - save and exit.

**Options** includes:

- **Master Volume** and **Effects Volume** (adjusting plays a preview tick)
- **Stick Sensitivity** (runtime multiplier for analogue steering)
- **Mouse Sensitivity** (mouse-look multiplier)
- **Invert Look Y** (On/Off for mouse and stick pitch)
- **Stick Deadzone** (gamepad axis deadzone)
- **Back** to return to the main pause menu

Navigate with `W`/`S`, arrow keys, or D-pad up/down. Adjust option values with
`A`/`D`, left/right arrows, or D-pad left/right. `Enter`/`E`/`A` selects.
`Esc`, `B` or `Select` go back one level, and `P`/`Start` also acts as "back"
on the options page while resuming from the main page. While paused the flight
loop is frozen and the HUD is hidden behind the overlay. Settings are written
immediately to `saves/settings.json` and survive deleting `save.json`.

---

## Gameplay loop

1. Spawn in deep space near the home system and fly toward a star marker.
2. Within range the HUD prompt appears; press `E` to enter the system.
3. Inside the system, approach a planet and press `E` to descend.
4. In the atmosphere, fly over the terrain. Near the surface, press `X` / `F`
   to brake and settle gently onto the ground, aligned to the local surface
   normal. Hold throttle while landed (or press `X` / `F` again with throttle)
   to lift off and climb back to flight. Climb above the exit altitude
   (about 1280 units) to return to orbit automatically.
5. Outside the system boundary (about 1250 units from the star) press `E` to
   return to the galaxy.
6. Open the galaxy map with `M`, search or filter systems, zoom in, choose one,
   and press `Enter`/`E` to warp there.
7. Aim at an ore vein and hold `R` / mouse left / `A` until it fractures. Harder
   tiers need several mining cycles but pay more; your mining tool upgrades
   automatically as your haul grows. Open the inventory with `I` to review it.
8. In space, hold `Shift` / `Y` for a moment to charge the pulse engine. After
   the 3-2-1 countdown the ship locks onto its heading and accelerates hard.
   `Space` / left trigger / `B` pulls out instantly.

### Landed ships

Each planet spawns 0-3 landed ships aligned to the local surface normal. Fly
within about 28 units of one and hold `E` / `A` for one second to swap into it.
The landed ship is consumed, the player model, engine sound and trail update,
and the new `selected_ship` is stored in the save file. The pause menu's
**Ship** entry cycles the four flyable profiles without needing a landed ship.

---

## Architecture

```
main.py                 CLI entry point, offscreen/smoke modes, reporting
config.py               every path and tuning constant in one place
game.py                 application shell: mode state machine, scene swap,
                        transitions, overlay layout, save/restore, frame loop

core/
  atomicfile.py         fsync-backed atomic JSON writes and .bak recovery
  audio.py              engine/wind/whoosh/pulse/UI audio, ship crossfade
  camera.py             smoothed third-person chase camera
  flight.py             body-frame SpaceFlight and AtmosphereFlight controllers
  fuel.py               fuel and pulse-fuel regen/drain/serialization
  gamepad.py            optional pygame-ce / XInput joystick polling
  geometry.py           spheres, heightfields, glows, points, additive blend
  hud.py                responsive HUD text, gauges and crosshair
  input.py              keyboard/mouse/gamepad merged into InputState
  inventory.py          inventory slots, counts and value
  inventory_ui.py       grid inventory overlay
  map.py                galaxy map: search, filter, zoom/pan, clusters, warp
  materials.py          12 mineable material definitions
  mathutil.py           clamp / damp helpers
  mining.py             cone lock-on mining state machine
  noise.py              value noise, fBm, Perlin-style terrain helpers
  pause.py              pause/options overlay
  pulse.py              pulse state machine (charge, countdown, active)
  pulse_fx.py           countdown text, speed lines, pullout flash/FOV
  save.py               versioned JSON schema, validation, atomic SaveStore
  settings.py           persistent player preferences (settings.json)
  ship.py               profile-aware GLB loading, materials and thrusters
  ship_profiles.py      flyable ships and freighter profile catalog
  textures.py           numpy RGBA texture uploads and soft dots
  trails.py             fading trails backed by one reusable dynamic geometry
  transitions.py        rainbow-spiral / fBm transition frames and callback

world/
  naming.py             deterministic syllable-based names
  archetypes.py         planet archetype, palette and weather tables
  planet.py             PlanetInfo generation from a seed
  system.py             SystemInfo, sun, planets, clouds, halos and labels
  galaxy.py             seeded spiral galaxy, starfield and system markers
  terrain.py            heightfield mesh, palette tinting, water, height_at
  sky.py                sky dome, sun disc, two scrolling cloud layers
  weather.py            dynamic wind/rain particle system
  veins.py              procedural vein models, Poisson scattering, tier yields
  atmosphere.py         composes terrain + sky + weather + landed ships, LOD
  traffic.py            seeded AI shuttles/freighters with warp, trails, LOD
  structures/           procedural structures: freighter hangars, landing pads
```

### Determinism

Every world derives from the galaxy seed: system positions from a seeded
`random.Random`, planets from a per-planet seed, terrain and textures from
deterministic numpy noise. The same seed always produces the same galaxy.
Starting a new game with `--seed` is enough to reproduce a run.

### Responsive UI

Panda3D's `aspect2d` spans `[-aspect, aspect]` horizontally and `[-1, 1]`
vertically. Every overlay derives its margins from those real bounds (read from
the `aspect2d` scale, not a cached window ratio), so the HUD, map, pause menu,
inventory and pulse effects stay on screen at 4:3, 16:10, 16:9 and ultrawide
resolutions. Layouts are refreshed on window events and re-checked each frame.

---

## Saves and generated files

- Save file: `saves/save.json`, with `saves/save.json.bak` holding the previous
  good version. Writes use a `.tmp` file, `fsync`, and an atomic rename.
- Preferences: `saves/settings.json` (plus `.bak`), independent of the save.
- Autosave every 12 seconds and on exit/window close.
- Cached procedural audio: `saves/generated/*.wav` (recreated on demand).
- Smoke-test screenshots: `saves/smoke_*.png`.

The schema is version 4 and includes the seed, mode, current system/planet,
ship position/HPR/throttle, visited systems, discovered planets, play time, the
galaxy position to return to, settings (volumes, stick/mouse sensitivity,
invert-Y, deadzone), inventory contents, fuel levels and the selected ship.
Older saves are migrated in place (v1 -> v2 settings/inventory, v2 -> v3 fuel,
v3 -> v4 selected ship). Saves are validated on load; a corrupt main file falls
back to the backup, then to a fresh start, and never blocks play.

Delete `saves/save.json` to start a new galaxy; preferences in
`saves/settings.json` are kept. Delete both to reset everything.

---

## Validation

`--smoke` renders the galaxy, a star system and an atmosphere, then reports the
mean colour and the fraction of pixels that differ from the background. A scene
fails if coverage is below 2%. A successful run prints `SMOKE OK`.

The offscreen mode is useful as a quick startup/render check:

```powershell
python main.py --offscreen --frames 10
```

It does not persist a save.

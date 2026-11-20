# REDRIVER2 Vehicle Physics Port — Planning & Implementation Document

> Historical porting plan.  The space game, the pygame `demo.py` and the module
> layout in section 4.1 are gone; `README.md` describes the current project.

## Purpose
Port the core vehicle dynamics model from OpenDriver2/REDRIVER2 (a reverse-engineered port of Driver 2) to a Pygame-based driving prototype for `vibe_procedural_driver6`. The original game is a PS1-era driving title; this document records the architecture, C→Python mappings, simplifications, audio integration, and tuning notes so the port is traceable and maintainable.

---

## 1. Source Material

### 1.1 REDRIVER2 files analyzed
| File | Key contents |
|---|---|
| `src_rebuild/Game/C/handling.c` | `InitCarPhysics`, `ProcessCarPad`, `GlobalTimeStep`, `RebuildCarMatrix`, `CheckCarToCarCollisions` |
| `src_rebuild/Game/C/wheelforces.c` | `StepCars`, `StepOneCar`, `GetFrictionScalesDriver1`, `AddWheelForcesDriver1`, `ConvertTorqueToAngularAcceleration`, `handlingType[7]` |
| `src_rebuild/Game/dr2types.h` | `CAR_DATA`, `HANDLING_DATA`, `RigidBodyState`, `WHEEL`, `CAR_COSMETICS`, `HANDLING_TYPE`, `CONTROL_TYPE_*` |
| `src_rebuild/Game/dr2math.h` | `ONE=4096`, `FIXEDH`, `FIXED`, `FixHalfRound`, `RSIN`, `RCOS`, `ratan2` usage |
| `src_rebuild/Game/C/bcollide.c` | `bcollided2d` (2D OBB SAT), `bFindCollisionPoint`, `bFindCollisionTime`, `CarBuildingCollision` |
| `src_rebuild/Game/dr2roads.c` | `FindSurfaceD2`, `sdGetCell`, `sdHeightOnPlane`, `MapHeight` |
| `src_rebuild/Game/C/cosmetic.c` | `car_cosmetics[]` loading from `LEVELS\*.LCF` |
| `src_rebuild/Game/C/pad.h` | `CAR_PAD_*` button bits |
| `src_rebuild/Game/C/cars.c` | `car_data[]`, `active_car_list[]`, `DrawCar` (rendering only) |

### 1.2 Existing project state
- Panda3D space game: `main.py` → `game.py` (`GalaxyExplorer`), `core/flight.py` (`SpaceFlight`, `AtmosphereFlight`), `core/ship.py`, `core/input.py` (Panda3D keyboard/mouse).
- Modules retained from existing repo: `core/mathutil.py`, `core/atomicfile.py`, `core/gamepad.py` (needs dummy-driver removal for Pygame), `core/save.py`/`core/settings.py` (schema must change).
- Space-specific modules to remove/archival: `core/ship.py`, `core/ship_profiles.py`, `core/pulse.py`, `core/pulse_fx.py`, `core/mining.py`, `core/inventory.py`, `core/inventory_ui.py`, `core/map.py`, `core/trails.py`, `core/audio.py`, `core/foot.py`, `core/fuel.py`, `core/geometry.py`, `core/textures.py`, `core/materials.py`, `core/noise.py`, `core/transitions.py`, `core/pause.py`, and all `world/` files (`galaxy.py`, `system.py`, `atmosphere.py`, `terrain.py`, `sky.py`, `weather.py`, `traffic.py`, `veins.py`, `planet.py`, `archetypes.py`, `naming.py`, `structures/*`).
- `requirements.txt` must drop `panda3d`/`panda3d-gltf`; keep `pygame-ce`, `numpy`, `pillow`.

---

## 2. Technical Decisions

### 2.1 Fixed-point math
- **Decision:** Integer emulation with `ONE=4096` for core vehicle state/forces/steering; `float` only at render/HUD boundary.
- **Why:** The feel depends on integer truncation and rounding: `FIXEDH`, `>>12`, `>>8`, steering increments of 32/64, `wheel_angle & 0xfffc`, and thresholds like `wheel_speed > 41943`. Python ints are arbitrary precision, so arithmetic is exact and traceable to C. Floats would require re-tuning every threshold and would drift in long runs.

### 2.2 Cosmetic / profile data
- **Decision:** Load `assets/vehicle/havana_model0/handling.json` into `CarCosmetics` dataclass; keep `raw` blob for traceability; `powerRatio`, `mass`, `traction`, `susCoeff` all in `ONE` units (4096 = 1.0).
- **Why:** Real values provided; no `.LCF`/`.COS` files in repo. JSON maps directly to `CAR_COSMETICS` from `dr2types.h`.

### 2.3 Dimensionality / view
- **Decision:** Full 3D vehicle state (x,y,z, suspension, yaw/pitch/roll) but **2D top-down Pygame demo** (render x/z plane, simple height function for ground).
- **Why:** Keeps wheel/suspension logic intact while keeping the prototype readable.

### 2.4 Collision
- **Decision:** Flat ground + `surface_at(x,z)` stub; **2D OBB SAT** from `bcollide.c` (`bcollided2d`, `bFindCollisionPoint`) for walls/buildings only; car-car collision stubbed.
- **Why:** Sufficient for handling validation without the full BSP/road-cell system.

### 2.5 Integration
- **Decision:** Semi-implicit Euler per frame, configurable substeps (document deviation from REDRIVER2's 4×2 RK subframe loop).
- **Why:** Pragmatic for Pygame; can increase substeps if unstable.

### 2.6 Rendering / audio / AI
- **Constraints:** No rendering/AI port from REDRIVER2. Use provided `engine.wav`, `idle.wav`, `skid.wav` only.
- **Audio:** `pygame.mixer`, separate channels (engine/idle/skid), state-driven controller (observer pattern fed by vehicle state/events).

---

## 3. Data & Asset Integration

### 3.1 Car cosmetics profile
File: `assets/vehicle/havana_model0/handling.json`

| JSON field | Maps to | Units / Notes |
|---|---|---|
| `model` | Model slot | Not used in port |
| `powerRatio` | `CarCosmetics.power_ratio` | `4096 = 1.0` |
| `mass` | `CarCosmetics.mass` | `4096 = 1.0` |
| `traction` | `CarCosmetics.traction` | `4096 = 1.0` |
| `susCoeff` | `CarCosmetics.sus_coeff` | `4096 = 1.0` |
| `wheelDisp[4]` | `CarCosmetics.wheel_disp[4]` | SVECTOR offsets |
| `cPoints[12]` | `CarCosmetics.c_points[12]` | SVECTOR 12 collision/lift points |
| `colBox[3]` | `CarCosmetics.col_box[3]` | Bounding box extents |
| `cog[3]` | `CarCosmetics.cog[3]` | Center of gravity |
| `twistRateX/Y/Z` | `CarCosmetics.twist_rate_x/y/z` | Car-specific twist |
| `extraInfo` | Various flags | Not used in minimal port |
| `sounds` block | Audio paths | Override defaults |
| `wheels` block | Texture paths | Override defaults |

### 3.2 Sound assets (provided, NOT from REDRIVER2)
| Asset | Path |
|---|---|
| Engine | `assets/sounds/vehicle/havana_model0/engine.wav` |
| Idle | `assets/sounds/vehicle/havana_model0/idle.wav` |
| Skid | `assets/sounds/vehicle/common/skid.wav` |

- All paths configurable via config/profile; defaults overrideable.
- Existing space sounds in `assets/sounds/spaceship/*` can be removed in the driving conversion.

### 3.3 Wheel textures
- `assets/vehicle/wheels/havana/wheelclean.png` (static/low speed)
- `assets/vehicle/wheels/havana/wheelfast.png` (high speed)
- `assets/vehicle/wheels/havana/wheeldam_fixed.png` (damage state)
- Visual: low-poly diamond wheel mesh with two flat textured planes per wheel; texture swaps when per-wheel spin speed exceeds `WHEEL_FAST_TEXTURE_SPEED` threshold (hysteresis optional to avoid flicker).
- JSON references stale `BMP` filenames/sets; actual PNGs live under `wheels/havana/`. Configurable overrides allowed.

---

## 4. Core Architecture & Function Mappings

### 4.1 Module layout (final)

```
main.py                     Pygame entry point (driving demo)
config.py                   driving constants + car profile (rewritten)
game.py                     DrivingGame: loop, scene, input, HUD, save (rewritten)
core/
  fixedpoint.py             ONE, FIXEDH, FIXED, FixHalfRound, rsin/rcos, ratan2, helpers
  redriver2_physics.py      Vehicle, Wheel, CarCosmetics, HandlingType, PhysicsWorld
  vehicle_audio.py          VehicleAudioController / AudioManager
  input.py                  Pygame InputManager -> DriveInput
  camera.py                 2D top-down chase camera
  track.py                  ground plane, walls, surface types
  mathutil.py               reused
  gamepad.py                reused (remove SDL_VIDEODRIVER=dummy)
  save.py                   adapted schema
  settings.py               adapted schema
  atomicfile.py             reused
world/
  city.py                   simple procedural city / test track
```

### 4.2 Function & C-citation mapping table

| Python method | C equivalent | File / Line notes |
|---|---|---|
| `Vehicle.reset()` | `InitCarPhysics` | `handling.c:InitCarPhysics` |
| `Vehicle.process_input()` | `ProcessCarPad` | `handling.c:ProcessCarPad` — digital/analogue steer, fast/regular, auto-centering, auto-brake, handbrake/wheelspin thrust |
| `Vehicle.step()` | `StepOneCar` | `wheelforces.c:StepOneCar` — gravity, speed calc, c-points lift, friction scales, wheel forces, angular accel |
| `Vehicle._get_friction_scales()` | `GetFrictionScalesDriver1` | `wheelforces.c:GetFrictionScalesDriver1` — front/rear split, auto-brake, handbrake, grass, traction scaling |
| `Vehicle._add_wheel_forces()` | `AddWheelForcesDriver1` | `wheelforces.c:AddWheelForcesDriver1` — per-wheel: world pos, suspension, point velocity, slip/longitudinal/lateral forces, lock/handbrake/wheelspin, friction coeff, acc/aacc |
| `Vehicle._convert_torque()` | `ConvertTorqueToAngularAcceleration` | `wheelforces.c:ConvertTorqueToAngularAcceleration` — twistRateY/Z, extraangulardamping |
| `PhysicsWorld.step()` | `GlobalTimeStep` | `handling.c:GlobalTimeStep` — step active vehicles, integrate, collisions (simplified from 4×2 RK) |
| `PhysicsWorld._obb_collision()` | `bcollided2d` / `bFindCollisionPoint` | `bcollide.c:bcollided2d`, `bFindCollisionPoint` — 2D OBB SAT collision/response |
| `Vehicle._rebuild_matrix()` | `RebuildCarMatrix` | `handling.c:RebuildCarMatrix` — orientation → transformation matrix |

### 4.3 Fixed-point helpers (`core/fixedpoint.py`)

| Helper | C equivalent / Note |
|---|---|
| `ONE = 4096` | `dr2math.h:ONE` |
| `FIXEDH(a)` | `FixHalfRound(a, 12)` |
| `FIXED(a)` | `a >> 12` |
| `FixHalfRound(x, bits)` | `dr2math.h:FixHalfRound` |
| `RSIN(a)` | `rcossin_tbl[((a) & 4095) * 2]` |
| `RCOS(a)` | `rcossin_tbl[((a) & 4095) * 2 + 1]` |
| `ratan2(y, x)` | `dr2math.h:DIFF_ANGLES` style, 4096 circle |
| `clamp(a, lo, hi)` | `dr2limits.h` style |
| `ABS(a)` | `fst_abs(a)` |
| `Vec3` ops (`+`, `-`, `dot`, `length`, `lerp`) | Pure integer ops with `ONE` scaling; float conversion only at render/HUD |

### 4.4 Vehicle state structures

**`CarCosmetics`** (from `handling.json`):
- `wheel_disp: tuple[Vec3, ...]` — 4 wheel world offsets
- `power_ratio: int` — `4096 = 1.0`
- `mass: int` — `4096 = 1.0`
- `traction: int` — `4096 = 1.0`
- `wheel_size: int` — pixels/scale units
- `c_points: tuple[Vec3, ...]` — 12 collision/lift points
- `col_box: Vec3` — bounding box
- `cog: Vec3` — center of gravity
- `twist_rate_x/y/z: int` — car-specific twist

**`HandlingType`** (from `wheelforces.c` table, default index 0):
- `friction_scale_ratio: int` (default 32)
- `aggressive_braking: int` (default 1)
- `four_wheel_drive: int` (default 0)
- `auto_brake_on: int` (default 1)

**`Wheel`**:
- `sus_compression: int` — suspension compression (0–42 clamped)
- `locked: int` — wheel locked (handbrake/brake)
- `on_grass: int` — surface flag
- `surface: int` — surface type bitfield
- `world_pos: Vec3` — transformed wheel position
- `slip_lateral: float` — provisional (derived from `AddWheelForcesDriver1`; see TODO in audio)
- `slip_longitudinal: float` — provisional (see TODO)

**`Vehicle`** (mirrors `HANDLING_DATA` / `RigidBodyState` essentials):
- `position: Vec3` — 12-bit fixed world (x, y, z)
- `orientation: int` — yaw in 4096 circle units
- `angular_velocity: Vec3` — radians per frame (fixed)
- `linear_velocity: Vec3` — fixed units per frame
- `wheel_angle: int` — steer, ±511 (digital) or 0..±352 (analogue)
- `thrust: int` — `FIXEDH(powerRatio * throttle)` — forward/reverse force
- `handbrake: int` — 0/1
- `wheelspin: int` — 0/1
- `auto_brake: int` — capped 0..90
- `may_be_colliding: int` — bitfield
- `wheels: list[Wheel]` — 4 per vehicle

### 4.5 Physics update flow (high-level)

```
Input (Pygame) → DriveInput
      │
      ▼
Vehicle.process_input(dt)          ← ProcessCarPad (handling.c)
  - steering state machine: digital/analogue,
    fast/regular, auto-centering,
    auto-brake increment/cap/decay
  - throttle/brake/handbrake/wheelspin
  - thrust: brake = FIXEDH(powerRatio * f(wheel_speed)),
    accel = FIXEDH(powerRatio * 4915), wheelspin thrust
      │
      ▼
PhysicsWorld.step(dt)            ← GlobalTimeStep (handling.c)
  for each vehicle:
      Vehicle.step(dt)           ← StepOneCar (wheelforces.c)
        - gravity
        - body lift from c-points (simplified)
        - _get_friction_scales()
        - _add_wheel_forces()    ← AddWheelForcesDriver1 (wheelforces.c)
          per wheel: world pos, suspension, point velocity,
          slip, longitudinal/lateral forces, lock/handbrake,
          friction coeff, acc/aacc, wheel_speed
        - _convert_torque()      ← ConvertTorqueToAngularAcceleration
      integrate velocity → position
      integrate angular velocity → orientation
      collision resolve (ground + OBB walls)
      │
      ▼
Render (Pygame): top-down car polygon + debug overlay
```

### 4.5 Collision & surface (`core/track.py`)

- `surface_at(x, z)` → (height, normal, surface_type)
- Flat ground by default; normal = (0, 1, 0), height = 0.
- OBB wall primitives + SAT (`bcollided2d`, `bFindCollisionPoint` adapted) for a configurable number of rectangular obstacles.
- Response: correct position by penetration depth along surface normal.

---

## 5. Audio Integration (`core/vehicle_audio.py`)

### 5.1 Module design
- **`VehicleAudioController`** / `AudioManager` — decoupled from physics math.
- Receives a per-frame `VehicleAudioState` snapshot (or event list) from `Vehicle` / game loop.
- Never calls pygame mixer directly from handling-equivalent functions.
- All asset paths configurable via profile; defaults as in §3.2.

### 5.2 Channels & initialization
- `pygame.mixer.init(frequency=22050, size=-16, channels=3)` — or as needed.
- Reserved channels:
  - `engine_ch` — main driving loop, overlapping/crossfading with idle.
  - `idle_ch` — idle loop, crossfades with engine.
  - `skid_ch` — skid loop, fades when grip returns.
- Channels stay alive; only volumes/pitches/positions update each frame.

### 5.3 RPM proxy (documented mapping formula)
Since the full `AddWheelForcesDriver1` longitudinal/lateral slip isn't always available for audio probing, the RPM proxy is derived from the **available** physics outputs:

```
wheel_speed_raw = abs(cp.hd.wheel_speed)   # from AddWheelForcesDriver1 output
rpm_proxy_norm  = clamp(wheel_speed_raw / REDLINE_WHEEL_SPEED, 0.0, 1.0)
```

Where `REDLINE_WHEEL_SPEED` is a tunable constant (default: `3300` — matches `oldSpeed > 3300` clamping in `StepOneCar`).

Throttle/load modulation:
```
engine_load = throttle  (0.0 .. 1.0, from input axis)
engine_speed_factor = rpm_proxy_norm * engine_load
```

### 5.4 Engine loop behavior
- Main loop on `engine_ch`.
- Pitch: `lerp(ENGINE_PITCH_MIN, ENGINE_PITCH_MAX, engine_speed_factor)`
- Volume: `lerp(ENGINE_VOLUME_MIN, ENGINE_VOLUME_MAX, engine_load_factor)`
  - `engine_load_factor = clamp(throttle * speed_factor_norm, 0, 1)`
  - Where `speed_factor_norm = clamp(speed / SPEED_NOMINAL, 0, 1)` and `SPEED_NOMINAL` is a tunable (e.g., max forward speed of the car profile).
- Crossfade: when `engine_load_factor < IDLE_ENTER_THRESHOLD`, begin crossfade to idle; when `engine_load_factor > IDLE_EXIT_THRESHOLD`, crossfade back. Crossfade time = `AUDIO_CROSSFADE_MS` ms (exponential or linear envelope; default 200 ms).
- No abrupt pops: always crossfade; never hard-stop a channel.

### 5.5 Idle loop behavior
- Play when effectively stationary **or** at very low speed with no meaningful throttle (engine running but not under load).
- Conditions to start idle (all must hold):
  - `speed < IDLE_ENTER_SPEED` (default: 8.0 units/s)
  - `throttle < IDLE_THROTTLE_EPS` (default: 0.05)
  - `handbrake == 0` and `wheelspin == 0`
- Play when:
  - `speed > IDLE_EXIT_SPEED` (default: 12.0 units/s) **or** `throttle > IDLE_EXIT_THROTTLE` (default: 0.1) — exit idle.
- Crossfade engine ↔ idle over `AUDIO_CROSSFADE_MS` ms (no pops).
- Hysteresis: `IDLE_ENTER_SPEED < IDLE_EXIT_SPEED` to avoid flicker.

### 5.6 Skid loop behavior
- **Trigger conditions** (any one qualifies):
  - `abs(lateral_slip) > SKID_SLIP_THRESHOLD` (default: 0.4, provisional; see TODO §5.8)
  - `wheelspin == 1` and `throttle > 0.3` (throttle wheelspin)
  - `handbrake == 1` and `speed > SKID_HB_SPEED_MIN` (default: 4.0 units/s)
  - `longitudinal_slip_magnitude > SKID_LON_THRESHOLD` (default: 0.5, provisional)
- When above threshold:
  - Loop `skid_ch` if not already looping.
  - Volume: `clamp(SKID_VOLUME_MIN + slip_mag * (SKID_VOLUME_MAX - SKID_VOLUME_MIN), 0, 1)`
    - `slip_mag = min(1.0, abs(lateral_slip) / SKID_SLIP_THRESHOLD)` (or longitudinal analog).
  - Fade out when slip drops below `SKID_SLIP_RELEASE` (default: 0.2, hysteresis).
- **No per-frame retrigger:** state latch (`skid_active` bool) + `SKID_COOLDOWN_S` (default: 0.05 s) minimum between cycles, or envelope smoothing to avoid stutter.
- Fade out: exponential decay to silence over `SKID_FADE_OUT_MS` (default: 200 ms) when conditions no longer hold.

### 5.7 Tunable constants (grouped in config/profile)

| Category | Constant | Default | Notes |
|---|---|---|---|
| **Engine** | `ENGINE_PITCH_MIN` | 0.8 | Linear factor applied to base pitch |
| | `ENGINE_PITCH_MAX` | 1.5 | |
| | `ENGINE_VOLUME_MIN` | 0.3 | |
| | `ENGINE_VOLUME_MAX` | 1.0 | |
| | `SPEED_NOMINAL` | max car speed from profile | Used to normalize speed for volume/pitch |
| | `IDLE_ENTER_SPEED` | 8.0 u/s | Below this, consider idle |
| | `IDLE_EXIT_SPEED` | 12.0 u/s | Above this, exit idle |
| | `IDLE_THROTTLE_EPS` | 0.05 | Throttle below which idle may start |
| | `AUDIO_CROSSFADE_MS` | 200 | Engine↔idle crossfade time |
| **Skid** | `SKID_SLIP_THRESHOLD` | 0.4 | Lateral slip proxy; provisional (TODO) |
| | `SKID_SLIP_RELEASE` | 0.2 | Hysteresis release level |
| | `SKID_VOLUME_MIN` | 0.2 | Min skid volume |
| | `SKID_VOLUME_MAX` | 1.0 | Max skid volume |
| | `SKID_FADE_OUT_MS` | 200 | Fade time when grip returns |
| | `SKID_HB_SPEED_MIN` | 4.0 u/s | Min speed for handbrake skid |
| | `SKID_COOLDOWN_S` | 0.05 | Min time between skid trigger cycles |
| **Physics / feel** | `POWER_RATIO` | from `handling.json` `powerRatio` | Car-specific |
| | `MASS` | from `handling.json` `mass` | |
| | `TRACTION` | from `handling.json` `traction` | |
| | `FRICTION_SCALE_RATIO` | from `handlingType` | |
| | `TWIST_RATE_Y` | from `handling.json` `twistRateY` | |
| | `WHEEL_FAST_TEXTURE_SPEED` | 3300 | Threshold for wheel-texture swap |
| | `WHEEL_SLOW_TEXTURE_SPEED` | 2500 | Hysteresis release for texture |
| | `REDLINE_WHEEL_SPEED` | 3300 | RPM proxy denominator |
| | `SPEED_UNITS_PER_S` | depends on profile scaling | Fixed-point → float conversion |
| **Collision** | `OBB_WALL_COUNT` | 0 (no static walls in minimal demo) | Configurable |
| **Misc** | `DT_FIXED` | 1.0/60.0 | Fixed timestep for physics (optional) |

### 5.8 TODO points (where final slip feeds in)
1. **`Vehicle._add_wheel_forces()`** — The per-wheel `slip_lateral` and `slip_longitudinal` values computed inside `AddWheelForcesDriver1` are the gold standard. In the prototype, provisional slip values are derived from velocity vs. wheel heading (see §4.3). When the final model is wired in, replace `VehicleAudioState.slip_lateral` / `slip_longitudinal` with the outputs from `_add_wheel_forces()` and remove the provisional calculation blocks. Mark in code: `TODO: wire in AddWheelForcesDriver1 slip outputs`.
2. **`SKID_SLIP_THRESHOLD`** — The default 0.4 is provisional; tune against actual `lateral_slip` from the full wheel-force model.
3. **`SKID_HB_SPEED_MIN`** — Handbrake-lockup skid threshold; may need adjustment based on `handbrake` state in `ProcessCarPad`.

### 5.9 Debug/validation hooks
- On-screen debug text / log line each frame/tick:
  ```
  speed=12.3 rpm_proxy=0.45 throttle=0.7 auto_brake=30 wheel_angle=-128
  wheel_speeds=[3300, -1200, 3200, -1000] locked=[0,0,0,0] surface=concrete
  engine_pitch=1.12 engine_vol=0.58 idle_vol=0.0 skid_active=False skid_vol=0.0
  fps=59.8
  ```
- Audio state flags clearly labeled so the behavior can be validated in the Pygame test harness.

---

## 6. Demo / Test Harness (`demo.py`)

### 6.1 Controls
| Key | Action |
|---|---|
| `W` | Throttle forward |
| `S` | Brake / reverse |
| `A` | Steer left (digital) |
| `D` | Steer right (digital) |
| `LShift` / `RShift` | Fast steer (L1 equivalent) |
| `Space` | Handbrake |
| `Esc` | Quit |

### 6.2 Scenarios to exercise
1. **Idle**: Start standing still; engine loop plays, idle fades in when speed < `IDLE_ENTER_SPEED` and throttle ~0; crossfade to engine when throttle > 0 or speed > `IDLE_EXIT_SPEED`.
2. **Acceleration**: Apply throttle; engine pitch/volume increase with speed; car accelerates forward; wheel texture swaps to `wheelfast` when any wheel speed > `WHEEL_FAST_TEXTURE_SPEED`.
3. **Turning**: While moving, apply steer; car turns; per-wheel compression visible; skid sound triggers when lateral slip > `SKID_SLIP_THRESHOLD` (hysteresis visible).
4. **Handbrake turn**: Apply handbrake + steer; rear wheels lock; car pivots; skid sound from handbrake condition; texture may stay clean if speed low.
5. **Braking from speed**: Release throttle, apply brake; engine volume decreases, pitch drops; car slows; skid sound may trigger under heavy braking (longitudinal slip condition).

### 6.3 Debug overlay (Pygame)
- Top-left: `position (x, z)`, `velocity magnitude`, `yaw`, `wheel_angle`
- Center: `throttle`, `brake`, `handbrake`, `wheelspin`, `auto_brake`
- Right column: `engine_pitch`, `engine_vol`, `idle_vol`, `skid_active`, `skid_vol`
- Bottom: `fps`, `dt`
- Per-wheel: `compression`, `locked`, `wheel_speed`, `wheel_texture` (clean/fast label)

### 6.4 Validation checklist
- [ ] Unit tests: `fixedpoint` math/trig helpers; steering increments/centering/auto-brake; friction scale branches (throttle/brake/handbrake/wheelspin).
- [ ] Harness checks: straight accel time/distance, braking distance, steady turn radius (measure vs expected), handbrake response latency, skid trigger/release hysteresis (no stutter), idle↔engine crossfade smoothness (no pops/stutter), wheel texture swap at threshold (clean ↔ fast).
- [ ] Tuning pass: expose all constants in `config.py`/`car_profile.py` section; iterate against expected Driver 2 feel (responsive, rear-biased/drifty, skidy under throttle/handbrake).

---

## 7. Validation Plan (summary)

### 7.1 Unit tests (logic-level)
- `fixedpoint` math/trig helpers: `FIXEDH`, `FIXED`, `rsin`, `rcos`, `ratan2`, `clamp`, `abs`.
- Steering state machine: digital fast/regular increments, auto-centering toward zero, auto-brake cap at 90, handbrake/wheelspin thrust logic.
- Friction scale branches: thrust < 0 vs ≥ 0; auto-brake contribution; lock conditions; handbrake/wheelspin adjustments; `frictionScaleRatio` and traction scaling; surface grass/sand modifiers.
- Integration: Euler step with/without substeps; position/velocity drift over 60 s.

### 7.2 Demo scenario checks
- Straight-line accel from 0 to max forward speed; measure time and distance.
- Coast from max speed; measure deceleration (engine brake vs real brake).
- Steady-state circle at fixed steer; measure turn radius and understeer/oversteer feel.
- Slalom (alternating steer left/right); measure response latency.
- Handbrake turn from speed; measure yaw response and skid sound trigger.
- Skid/spin from throttle on loose surface; verify skid volume/slip correlation.

### 7.3 Known deviations / assumptions
- **Fixed-point vs C exact behavior:** Python ints are exact, but C `>>12` rounding may differ slightly from Python `//4096` for negative numbers. The `FIXEDH` helper (`FixHalfRound`) mitigates this. Documented in `fixedpoint.py` comments.
- **Wheel texture swap threshold:** `WHEEL_FAST_TEXTURE_SPEED` default 3300 matches `oldSpeed > 3300` clamping in `StepOneCar` but is provisional; tune against actual `wheel_speed` from the full model.
- **Simplified collision:** Flat ground + OBB walls only; no multi-level roads/ramps, no car-car collision. The `c_points` lift logic is partially modeled (see `StepOneCar` lift loop in `wheelforces.c`); full BSP/road-cell system not ported.
- **Audio RPM proxy:** Derived from `wheel_speed` clamp; not the exact engine-RPM model from REDRIVER2. When final `AddWheelForcesDriver1` slip outputs are available, swap in per-wheel `slip_longitudinal` for more accurate engine braking sound.
- **Integration method:** Single Euler step per frame instead of REDRIVER2's 4×2 subframe RK loop. Configurable substeps available if stability issues arise.

---

## 8. Timeline / Implementation Order (post-confirmation)

1. **Week 1** — `core/fixedpoint.py` + basic tests/helpers; profile/cosmetics loader (`handling.json`).
2. **Week 2** — `core/redriver2_physics.py`: data structures, `reset`, steering/thrust (`ProcessCarPad`), wheel/friction/forces, integration, minimal surface/collision hooks.
3. **Week 3** — `core/vehicle_audio.py`: mixer setup, `rpm_proxy`/slip proxies, engine/idle crossfade, skid latch+hysteresis; debug overlay hooks.
4. **Week 4** — `demo.py`: Pygame loop, input, rendering (car + runtime wheels), HUD/debug, audio wiring; scenario testing.
5. **Week 5** — `PORTING_NOTES.md` finalization, tuning pass, known deviations documented.
6. **Optional Week 6** — Phase 2: full conversion of `main.py`/`game.py`/`config.py`/`core/`/`world/` to driving game (replace spaceflight with driving, adapt save/schema, update `requirements.txt`).

---

## 9. References & Citations

### 9.1 C source citations
All Python methods cite the original C function and file for traceability:
- `handling.c:InitCarPhysics` — initialization, direction/orientation, suspension start values, matrix build.
- `handling.c:ProcessCarPad` — digital/analogue steer, fast/regular, auto-centering, auto-brake, handbrake/wheelspin thrust logic.
- `handling.c:GlobalTimeStep` — world step loop, integration order, collision resolution outline.
- `handling.c:RebuildCarMatrix` — orientation → transformation matrix.
- `wheelforces.c:StepOneCar` — gravity, speed calc, c-points lift, friction scales, wheel forces, angular accel.
- `wheelforces.c:GetFrictionScalesDriver1` — front/rear split, auto-brake, handbrake, grass, traction scaling.
- `wheelforces.c:AddWheelForcesDriver1` — per-wheel: world pos, suspension, point velocity, slip/longitudinal/lateral forces, lock/handbrake/wheelspin, friction coeff, acc/aacc, wheel_speed.
- `wheelforces.c:ConvertTorqueToAngularAcceleration` — twistRateY/Z, damping.
- `bcollide.c:bcollided2d` — 2D OBB SAT collision test.
- `bcollide.c:bFindCollisionPoint` — collision point/normal computation.
- `dr2math.h:ONE`, `FIXEDH`, `FixHalfRound`, `RSIN`, `RCOS`, `ratan2` — math constants and trig.
- `dr2types.h:CAR_DATA`, `HANDLING_DATA`, `RigidBodyState`, `WHEEL`, `CAR_COSMETICS`, `HANDLING_TYPE`, `CONTROL_TYPE_*` — data structures.
- `pad.h:CAR_PAD_*` — input button/axis definitions.

### 9.2 Asset provenance
- `assets/vehicle/havana_model0/handling.json` — extracted from REDRIVER2 `.LCF` via `ProcessCosmeticsLump`; `4096 = 1.0` scaling noted.
- `assets/vehicle/wheels/havana/wheelclean.png`, `wheelfast.png`, `wheeldam_fixed.png` — actual PNG files; JSON references stale `BMP` paths/sets.
- `assets/sounds/vehicle/havana_model0/engine.wav`, `idle.wav`, `common/skid.wav` — provided for the Pygame prototype; not extracted from REDRIVER2.
- All other assets from the original `vibe_procedural_driver6` project (Panda3D space game) are either retained (math, save, settings) or archived for Phase 2 conversion.

---
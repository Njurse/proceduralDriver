"""Central configuration: paths and tuning constants for the driving game."""

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"

# Window / loop -------------------------------------------------------------
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# Car model loading ---------------------------------------------------------
# The havana OBJs are in metres (Driver 2 units / core.units.DR2_UNITS_PER_METRE)
# but keep Driver 2 model axes: Y points DOWN (roof at -Y) and the nose faces
# -Z.  main.py maps them onto the physics body with a -1 scale.
CAR_MODEL_INVERT_VERTICAL = False
HAVANA_HANDLING_JSON = (ASSETS / "vehicle" / "havana_model0" / "handling.json").as_posix()
HAVANA_MODEL_OBJ = (ASSETS / "vehicle" / "havana_model0" / "HAVANA_model0_clean.obj").as_posix()

# Procedural terrain (world/terrain.py, world/chunks.py) --------------------
# Rolling hills: Perlin fBm, height = AMPLITUDE * fbm(x / SCALE, z / SCALE).
TERRAIN_SEED = 1977
TERRAIN_SCALE = 140.0           # metres per noise cell of the first octave
TERRAIN_OCTAVES = 4
TERRAIN_AMPLITUDE = 10.0        # metres; hills reach about +-5 m, slopes under 10 deg
TERRAIN_CHUNK_SIZE = 64.0       # metres per square chunk
TERRAIN_CHUNK_RES = 32          # quads per chunk side (2 m spacing)
DRAW_DISTANCE_M = 290.0         # camera far plane (was ~190 m: 3 chunks of terrain)
# chunks kept loaded in each direction round the car: enough to reach the far plane
TERRAIN_VIEW_CHUNKS = math.ceil(DRAW_DISTANCE_M / TERRAIN_CHUNK_SIZE)
TERRAIN_BUILDS_PER_FRAME = 2    # chunk meshes built per frame while streaming
SKYBOX_PNG = (ASSETS / "world" / "landscape" / "skybox.png").as_posix()
SKY_RADIUS_M = 250.0            # sky dome radius, inside the far plane
DRIVE_FOG_DENSITY = 0.0073      # exp-squared fog: ~1% visibility at the far plane

# Biomes, regions and roads (world/biomes.py, world/regions.py, world/roads.py)
WORLD_BIOMES = ("plains", "desert")  # assets/world/definitions/<name>.json
REGION_SCALE_M = 900.0          # wavelength of the noise that splotches the biomes
REGION_BAND = 0.08              # blend half-width in region-noise units (~200 m)
ROAD_LIFT_M = 0.04              # road strips sit this far above the flattened ground
ROAD_DETAIL_TILE_M = 2.5        # metres per repeat of the road detail texture

# Chase camera (main.py).  Heights are above the car body's origin (~0.77 m
# off the road).  The aim point sits LOOK_AHEAD metres in front of the car in
# the chase view and slides back onto the car as the right stick orbits round.
CHASE_CAM_DIST = 7.5            # metres behind the car (was 10)
CHASE_CAM_HEIGHT = 2.0          # was 3.6
CHASE_CAM_LOOK_AHEAD = 6.0      # was 0: aimed at the car itself
CHASE_CAM_LOOK_HEIGHT = -0.45   # below body origin: keeps the car in frame
CHASE_CAM_FOV = 60.0            # horizontal degrees (Panda default 40)
# Rough ground / slides: the camera hangs back from the body's vertical and
# sideways motion instead of being welded to it.  Gains are metres of camera
# offset per m/s of body velocity; the signal is clamped to +-2 m/s so a big
# jump or landing cannot fling the view.
CHASE_CAM_BOB_TAU = 0.12        # seconds; smoothing of the motion signals
CHASE_CAM_BOB = 0.08            # vertical dip per m/s of body rise/fall
CHASE_CAM_SWAY = 0.05           # sideways lean per m/s of lateral slip
CHASE_CAM_KICK_MAX = 2.0        # clamp on the motion signal, m/s
CHASE_CAM_ORBIT_LOOK_RISE = 1.2 # aim point rises this much at a full side view

# Wheels -------------------------------------------------------------------
# Wheel speeds are rad/s x 100 (0.327 m tyre radius): 3300 ~ 11 m/s.
WHEEL_FAST_TEXTURE_SPEED = 3300

# Gearbox and top speed (core/engine.py, core/car_forces.py) ----------------
# DEVIATION: the C's wheel drag caps the car at ~147 km/h and its gear table
# tops out first gear at ~59 km/h.  These stretch both; launch thrust is the C's.
CAR_TOP_SPEED_KMH = 215.0       # flat-road terminal speed at full throttle
TYRE_RADIUS_M = 0.327
GEAR_RATIOS = (2.50, 1.65, 1.25, 1.00)
FINAL_DRIVE = 3.33
IDLE_RPM = 850.0
SHIFT_UP_RPM = 5600.0           # 1st-3rd top out at ~83 / 126 / 166 km/h
SHIFT_DOWN_RPM = 2600.0         # drop a gear below this
REDLINE_RPM = 6200.0            # 4th reaches ~5800 rpm at 215 km/h
SHIFT_TIME_S = 0.2              # thrust cut per upshift (the C's changingGear)

# Thrust per gear as a fraction of the C's, easing off in the higher gears.
# 0.78 in 1st gives 0-60 mph in ~3.1 s (stock C: ~3.1 s); 0.70 in 4th
# pulls 190->200 km/h in under 3 s.
GEAR_THRUST_SCALES = (0.78, 0.74, 0.72, 0.70)

# Audio (driving) -----------------------------------------------------------
# core/engine_sound.py (model) and core/vehicle_audio.py (output).
AUDIO_RATE = 22050
AUDIO_BLOCK = 1024              # samples per queued block (~46 ms)
AUDIO_MASTER_VOLUME = 0.8
ENGINE_SAMPLE_RPM = 3000.0      # rpm engine.wav sounds like at its own pitch
IDLE_SAMPLE_RPM = 850.0         # rpm idle.wav sounds like at its own pitch
FREE_REV_RPM = 4500.0           # clutch-out revs at full throttle (launch, spin)
CLUTCH_FULL_SPEED = 5.0         # m/s where the clutch is fully in
SPIN_CLUTCH = 0.4               # clutch share while spinning or airborne
RPM_RISE_TIME = 0.10            # s, smoothing time constants
RPM_FALL_TIME = 0.25
RPM_SHIFT_DROP_TIME = 0.05      # quick drop to the next gear's rpm
LOAD_TIME = 0.08
LEVEL_TIME = 0.06               # idle/engine volume smoothing
ENGINE_GAIN_ON_LOAD = 1.0
ENGINE_GAIN_OFF_LOAD = 0.45
IDLE_GAIN = 0.6

# Slip is the sideways (or locked/spinning) speed at a tyre's contact, m/s.
# Normal hard cornering slides ~0.6 m/s; handbrake turns 0.8-1.5 m/s.
SKID_SLIP_ON = 1.0
SKID_SLIP_OFF = 0.5
SKID_SLIP_FULL = 3.0
SKID_GAIN = 0.7
SKID_ATTACK_TIME = 0.04
SKID_RELEASE_TIME = 0.2
SKID_HB_SPEED_MIN = 4.0         # m/s

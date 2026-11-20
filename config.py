"""Central configuration: paths and tuning constants for the driving game."""

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

# Driving world -------------------------------------------------------------
DRIVE_FOG_DENSITY = 0.005       # exponential fog density for the ground plane
DRIVE_BOUNDARY_MARGIN = 12.0    # keep the car this far inside the ground quad

# Chase camera (main.py).  Heights are above the car body's origin (~0.77 m
# off the road).  The aim point sits LOOK_AHEAD metres in front of the car in
# the chase view and slides back onto the car as the right stick orbits round.
CHASE_CAM_DIST = 7.5            # metres behind the car (was 10)
CHASE_CAM_HEIGHT = 2.0          # was 3.6
CHASE_CAM_LOOK_AHEAD = 6.0      # was 0: aimed at the car itself
CHASE_CAM_LOOK_HEIGHT = -0.7    # below body origin: ~11 deg down, keeps the car in frame
CHASE_CAM_FOV = 60.0            # horizontal degrees (Panda default 40)

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

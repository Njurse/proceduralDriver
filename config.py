"""Central configuration: paths, tuning constants and gameplay scales."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SAVE_DIR = ROOT / "saves"
SAVE_PATH = SAVE_DIR / "save.json"
SETTINGS_PATH = SAVE_DIR / "settings.json"
CACHE_DIR = SAVE_DIR / "generated"

# Window / loop -------------------------------------------------------------
WINDOW_TITLE = "Procedural Galaxy Explorer"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
TARGET_FPS = 60
SAVE_VERSION = 5

# Assets --------------------------------------------------------------------
SHIP_GLB = (ASSETS / "models" / "spaceship"
            / "spaceship.glb").as_posix()
SHIP_SCALE = 1.2

# Ship profiles --------------------------------------------------------------
SHIP_T2_GLB = (ASSETS / "models" / "spaceship" / "spaceship_t2.glb").as_posix()
SHIP_T3_GLB = (ASSETS / "models" / "spaceship" / "spaceship_t3.glb").as_posix()
SHIP_T4_GLB = (ASSETS / "models" / "spaceship" / "spaceship_t4.glb").as_posix()
SHIP_F1_GLB = (ASSETS / "models" / "spaceship" / "spaceship_f1.glb").as_posix()

# Traffic -------------------------------------------------------------------
TRAFFIC_GALAXY_COUNT = 8
TRAFFIC_SYSTEM_COUNT = 4
TRAFFIC_FREIGHTER_COUNT = 2
TRAFFIC_CRUISE_SPEED = 60.0
TRAFFIC_FREIGHTER_SPEED = 18.0
TRAFFIC_WARP_SPEED = 700.0
TRAFFIC_WARP_TIME = 1.2
TRAFFIC_WAYPOINT_REACH = 40.0
TRAFFIC_REPATH_INTERVAL = 6.0
TRAFFIC_SPAWN_RADIUS = 4000.0
TRAFFIC_STEER = 2.5
TRAFFIC_FREIGHTER_STEER = 0.4
TRAFFIC_FREIGHTER_MAX_TURN = 0.25
TRAFFIC_LOD_DIST = 2400.0      # hide distant traffic meshes/trails

# Freighter hangar / landing pads --------------------------------------------
# Local dimensions are in freighter-model units (the freighter root scales
# these up to world size).  The hangar runs along the freighter's Y axis with
# the glowing entry opening toward +Y (the bow).
HANGAR_WIDTH = 34.0
HANGAR_HEIGHT = 24.0
HANGAR_LENGTH = 92.0
# The hangar mounts as a bay pod hanging under the freighter hull so the
# glowing entry is visible from outside (hull half-height in model units).
FREIGHTER_HULL_HALF_H = 51.0
HANGAR_ENTRY_RANGE = 780.0     # prompt/notice range from the entry glow
HANGAR_HUM_RANGE = 420.0       # hum falloff radius around the entry
HANGAR_EXIT_CLEAR = 300.0      # distance past the entry before control returns
HANGAR_LOD_DIST = 2600.0       # hide the hangar room beyond this range
PAD_LOCAL = (0.0, -24.0)       # pad centre (x, y) on the hangar floor
PAD_RADIUS = 9.0
PAD_CONE_HEIGHT = 20.0
PAD_CONE_RADIUS = 7.0
PAD_PROMPT_RANGE = 90.0        # show the land prompt within this pad range
PAD_HOVER_HEIGHT = 12.0        # autopilot hover height above the pad
PAD_DESCENT_RATE = 2.6         # damp rate onto the pad coordinates
PAD_BRAKE_RATE = 2.4           # velocity kill rate while locking on
PAD_ALIGN_RATE = 1.8           # lerp rate toward the pad orientation
PAD_SETTLE_SECONDS = 0.5       # rest on the pad before rotating to align
PAD_ROTATE_SECONDS = 1.4       # duration of the smooth orientation lerp
TAKEOFF_RISE_HEIGHT = 18.0     # vertical rise off the pad before the exit run
TAKEOFF_CLIMB_RATE = 13.0
TAKEOFF_EXIT_SPEED = 95.0      # cruise speed on the exit leg
TAKEOFF_ACCEL = 1.4

# Landed ships --------------------------------------------------------------
LANDED_SHIP_MIN = 0
LANDED_SHIP_MAX = 3
LANDED_SHIP_SWAP_HOLD = 1.0
LANDED_SHIP_SWAP_RANGE = 28.0
LANDED_SHIP_LOD_DIST = 900.0
VEIN_LOD_DIST = 1300.0

# Trails --------------------------------------------------------------------
TRAIL_MAX_POINTS = 18
TRAIL_POINT_SPACING = 4.0

# Space flight --------------------------------------------------------------
SPACE_MAX_SPEED = 150.0
SPACE_BOOST_MULT = 3.4
SPACE_REVERSE = 0.35
SPACE_THROTTLE_RATE = 1.6       # throttle units / second
SPACE_SPEED_SMOOTH = 2.2
SPACE_PITCH_RATE = 1.35         # radians / second
SPACE_YAW_RATE = 1.00
SPACE_ROLL_RATE = 1.80
# Mouse movement is a direct body-frame angle step.  These are radians per
# normalized mouse-delta unit (the values Panda reports per frame).
MOUSE_GAIN_SPACE = 2.4
MOUSE_GAIN_ATMO = 1.9
CONTROL_SENSITIVITY = 1.0        # runtime multiplier for analogue steering
ANALOGUE_SENSITIVITY = 1.0       # gamepad-stick steering gain multiplier
MASTER_VOLUME = 1.0
EFFECTS_VOLUME = 1.0
MOUSE_DELTA_LIMIT = 0.28
MOUSE_SENSITIVITY = 1.0          # runtime multiplier for mouse look
INVERT_Y = False                 # invert mouse/stick look vertically
GAMEPAD_DEADZONE = 0.16
UI_CLICK_VOLUME = 0.35
ENGINE_FADE_SECONDS = 0.35       # crossfade between ships' engine loops

# Atmospheric (plane) flight ------------------------------------------------
ATMO_MIN_SPEED = 38.0
ATMO_MAX_SPEED = 175.0
ATMO_CRUISE = 100.0
ATMO_THROTTLE_RATE = 1.2
ATMO_SPEED_SMOOTH = 1.5
ATMO_PITCH_RATE = 1.05
ATMO_YAW_RATE = 0.85
ATMO_ROLL_RATE = 1.65
ATMO_BANK_TURN = 0.60           # extra yaw produced by banking
ATMO_SINK = 0.05
ATMO_TERRAIN_CLEARANCE = 7.0
ATMO_CEILING = 1500.0
ATMO_AUTO_LEVEL = 0.55

# Landing assist (X button near the surface) ---------------------------------
LAND_MAX_ALTITUDE = 70.0        # max AGL at which landing/takeoff assist works

# Pulse engine (fast travel) -------------------------------------------------
PULSE_COUNTDOWN = 3.0
PULSE_SPEED_MULT = 8.0          # relative to boosted max speed
PULSE_FUEL_MAX = 100.0
PULSE_FUEL_DRAIN = 6.7          # per second while pulsing (~15s full tank)
PULSE_FUEL_REGEN = 4.0          # per second while idle
PULSE_FUEL_MIN_START = 20.0     # min fuel to engage
PULSE_MIN_ALTITUDE = 80.0       # min AGL to pulse in atmosphere
PULSE_SURFACE_COS = 0.25        # min ship_forward·surface_normal
PULSE_PULLOUT_SPEED = 40.0      # speed after instant pullout
PULSE_PULLOUT_FLASH = 0.25      # seconds
PULSE_PULLOUT_FOV_REBOUND = 55.0
PULSE_FOV = 95.0
LAND_CLEARANCE = 1.6            # final height above the sampled ground
LAND_DESCENT_SPEED = 1.4        # gentle vertical damp rate on approach
LAND_ALIGN_SPEED = 2.4          # rate for aligning ship up to surface normal
LAND_BRAKE_SPEED = 2.2          # deceleration rate during landing
TAKEOFF_ALTITUDE = 95.0         # AGL at which liftoff hands back to flight
TAKEOFF_CLIMB_SPEED = 16.0      # upward speed while lifting off
TAKEOFF_ACCEL = 1.6             # speed ramp back toward cruise

# Chase camera --------------------------------------------------------------
CAM_SPACE_DIST = 15.0
CAM_SPACE_HEIGHT = 4.5
CAM_ATMO_DIST = 19.0
CAM_ATMO_HEIGHT = 6.5
CAM_SMOOTH = 12.0
CAM_LOOK_AHEAD = 10.0

# Galaxy --------------------------------------------------------------------
GALAXY_RADIUS = 12000.0
GALAXY_ARM_COUNT = 4
GALAXY_ARM_TWIST = 2.35
GALAXY_SYSTEM_COUNT = 512
GALAXY_ENTRY_DIST = 95.0
GALAXY_START_DIST = 430.0
GALAXY_BOUNDARY = 30000.0
STARFIELD_COUNT = 2600
STARFIELD_RADIUS = 30000.0

# Star systems --------------------------------------------------------------
SYSTEM_PLANET_MIN = 3
SYSTEM_PLANET_MAX = 4
SUN_RADIUS = 34.0
PLANET_ORBIT_BASE = 140.0
PLANET_ORBIT_STEP = 100.0
PLANET_ORBIT_JITTER = 40.0
SYSTEM_ENTRY_PAD = 150.0
SYSTEM_INTERACT_DIST = 65.0
SYSTEM_EXIT_DIST = 1250.0

# Atmosphere / terrain ------------------------------------------------------
ATMO_WORLD_SIZE = 5200.0
ATMO_GRID = 176
ATMO_FOG_DENSITY = 0.00030
ATMO_WATER_FRACTION = 0.42
ATMO_ENTRY_ALTITUDE = 640.0
ATMO_EXIT_ALTITUDE = 1280.0
ATMO_PARTICLE_COUNT = 900
ATMO_PARTICLE_RADIUS = 340.0
ATMO_CLOUD_ALTITUDE = 560.0
ATMO_SKY_RADIUS = 9000.0
PLANET_TEXTURE_WIDTH = 512
PLANET_TEXTURE_HEIGHT = 256
CLOUD_TEXTURE_WIDTH = 512
CLOUD_TEXTURE_HEIGHT = 256
ATMO_TEX_GRID = 256

# Transition ----------------------------------------------------------------
TRANSITION_SECONDS = 1.05
TRANSITION_WIDTH = 384
TRANSITION_HEIGHT = 216

# Misc ----------------------------------------------------------------------
AUTOSAVE_SECONDS = 12.0
MESSAGE_SECONDS = 3.5

# On-foot (walking) ---------------------------------------------------------
FOOT_WALK_SPEED = 7.5
FOOT_STRAFE_MULT = 0.85
FOOT_SPRINT_MULT = 1.85
FOOT_JUMP_SPEED = 7.0
FOOT_GRAVITY = 24.0
FOOT_COYOTE_TIME = 0.12
FOOT_JUMP_BUFFER = 0.12
FOOT_TURN_SENSITIVITY = 2.0
FOOT_CAM_DIST = 6.5
FOOT_CAM_HEIGHT = 3.2
FOOT_AVATAR_SIZE = 1.2
FOOT_MINING_RANGE = 90.0
FOOT_MINING_CONE_DEG = 4.0
FOOT_REENTER_RANGE = 6.0
FOOT_FOOT_HEIGHT = 0.6
FOOT_EXIT_MAX_SLOPE = 0.55
FOOT_LAND_SQUASH = 0.35

# Jetpack ---------------------------------------------------------------
JETPACK_FUEL_MAX = 100.0
JETPACK_DRAIN = 35.0        # per second while thrusting
JETPACK_RECHARGE = 25.0     # per second while grounded, 0 in air
JETPACK_THRUST = 18.0       # upward accel
JETPACK_FORWARD_ASSIST = 6.0

# Mining / resources ---------------------------------------------------------
VEIN_COUNT = 55
VEIN_MIN_DIST = 95.0
MINING_RANGE = 130.0
MINING_CONE_DEG = 12.0
MINING_TIME = 2.0
# Harder tiers need several mining cycles; higher tiers yield fewer units.
MINING_HARDNESS = {"common": 1, "uncommon": 2, "rare": 3, "legendary": 4}
MINING_YIELD = {"common": (1, 3), "uncommon": (1, 2),
                "rare": (1, 1), "legendary": (1, 1)}
MINING_TOOL_STEP = 0.4           # speed bonus per automatic tool level
MINING_TOOL_EVERY = 25           # units mined per tool upgrade
MAP_CLUSTER_ZOOM = 0.75          # below this zoom, adjacent markers cluster
INVENTORY_COLS = 6
INVENTORY_ROWS = 4

# ---------------------------------------------------------------------------
# Driving / REDRIVER2 handling model
# ---------------------------------------------------------------------------
# Car model loading ---------------------------------------------------------
# The havana OBJs are in metres (Driver 2 units / core.units.DR2_UNITS_PER_METRE)
# but keep Driver 2 model axes: Y points DOWN (roof at -Y) and the nose faces
# -Z.  drive.py maps them onto the physics body with a -1 scale.
CAR_MODEL_INVERT_VERTICAL = False
HAVANA_MODEL_DIR = (ASSETS / "vehicle" / "havana_model0").as_posix()
HAVANA_HANDLING_JSON = (ASSETS / "vehicle" / "havana_model0" / "handling.json").as_posix()
HAVANA_MODEL_OBJ = (ASSETS / "vehicle" / "havana_model0" / "HAVANA_model0_clean.obj").as_posix()
# Space-mode ship scale only (core/ship_profiles.py).  It was 0.01 on the old
# Driver 2 unit OBJ; the OBJ is now 150x smaller, so 1.5 keeps the ship's size.
# The driving sim draws the car at 1:1 metres.
HAVANA_SHIP_SCALE = 0.01 * 150.0
HAVANA_WHEEL_TEXTURES = (ASSETS / "vehicle" / "wheels" / "havana").as_posix()
# Driving world -------------------------------------------------------------
DRIVE_FOG_DENSITY = 0.005       # exponential fog density for the ground plane
DRIVE_BOUNDARY_MARGIN = 12.0    # keep the car this far inside the ground quad

# Wheels -------------------------------------------------------------------
# Wheel speeds are rad/s x 100 (0.327 m tyre radius): 3300 ~ 11 m/s.
WHEEL_FAST_TEXTURE_SPEED = 3300

# Audio (driving) -----------------------------------------------------------
# Engine and idle follow the gearbox revs (core/vehicle_audio.py).
# Slip is the sideways (or locked/spinning) speed at a tyre's contact, m/s.
# Normal hard cornering slides ~0.6 m/s; handbrake turns 0.8-1.5 m/s.
SKID_SLIP_THRESHOLD = 1.0
SKID_SLIP_RELEASE = 0.5
SKID_VOLUME_MIN = 0.2
SKID_VOLUME_MAX = 1.0
SKID_FADE_OUT_MS = 200
SKID_HB_SPEED_MIN = 4.0         # m/s

"""Golden-trajectory test: behavioral identity guard for the handling model.

Runs a scripted input sequence through PhysicsWorld and compares the resulting
position/speed/yaw/wheel_angle trajectory against a committed golden file.
A missing golden file fails the test.  To (re)generate it after an intentional
physics change, run with REGEN_GOLDEN=1 and commit the new file.
"""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.car_profile import load_cosmetics  # noqa: E402
from core.redriver2_physics import PhysicsWorld, Vehicle  # noqa: E402
from core.track import Track  # noqa: E402

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "trajectory.json")

# Scripted 5-second scenario exercising accel, steer, brake, handbrake, coast:
# (end time s, steer, throttle, brake, handbrake)
SCRIPT = (
    (1.0, 0.0, 1.0, 0.0, False),
    (2.0, 0.6, 0.8, 0.0, False),
    (2.5, 0.6, 0.0, 0.0, False),
    (3.0, -0.8, 0.0, 1.0, False),
    (3.5, -0.8, 0.0, 0.0, True),
    (float("inf"), 0.0, 0.5, 0.0, False),
)


def _inputs_for_frame(i):
    t = i / 60.0
    for end, steer, throttle, brake, handbrake in SCRIPT:
        if t < end:
            return SimpleNamespace(steer=steer, throttle=throttle, brake=brake,
                                   handbrake=handbrake, fast_steer=False, wheelspin=False)
    raise AssertionError("unreachable")


def run_scenario(frames=300):
    v = Vehicle(load_cosmetics())
    v.reset(0.0, 0.0, 0.0)
    world = PhysicsWorld(Track())
    world.add(v)
    out = []
    for i in range(frames):
        v.process_input(_inputs_for_frame(i))
        world.step(1.0 / 60.0)
        out.append({
            "frame": i,
            "x": round(float(v.pos[0]), 6),
            "y": round(float(v.pos[1]), 6),
            "z": round(float(v.pos[2]), 6),
            "speed": round(v.speed, 6),
            "heading": round(v.heading, 9),
            "wheel_angle": round(v.wheel_angle, 9),
        })
    return out


def test_golden_trajectory():
    traj = run_scenario()
    if os.environ.get("REGEN_GOLDEN") == "1":
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w", newline="\n") as f:
            json.dump(traj, f, indent=1)
    assert os.path.exists(GOLDEN_PATH), (
        f"golden file missing: {GOLDEN_PATH}; regenerate with REGEN_GOLDEN=1")
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert len(golden) == len(traj), "trajectory length mismatch"
    for g, t in zip(golden, traj, strict=True):
        assert g == t, f"frame {g['frame']} mismatch:\n  golden={g}\n  actual={t}"

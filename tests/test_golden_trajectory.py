"""Golden-trajectory test: behavioral identity guard for the handling refactor.

Runs a scripted input sequence through PhysicsWorld and compares the resulting
position/speed/yaw/wheel_angle trajectory against a committed golden file.
If the golden file is missing it is generated (and the test passes); re-run
after generation to enforce equality.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.car_profile import load_cosmetics
from core.redriver2_physics import Vehicle, PhysicsWorld
from core.track import Track

GOLDEN_PATH = os.path.join(os.path.dirname(__file__), "golden", "trajectory.json")


def _inputs_for_frame(i):
    # Scripted 5-second scenario exercising accel, steer, handbrake, coast.
    t = i / 60.0
    inp = type("I", (), {})()
    if t < 1.0:
        inp.steer = 0.0; inp.throttle = 1.0; inp.brake = 0.0
    elif t < 2.0:
        inp.steer = 0.6; inp.throttle = 0.8; inp.brake = 0.0
    elif t < 2.5:
        inp.steer = 0.6; inp.throttle = 0.0; inp.brake = 0.0
    elif t < 3.0:
        inp.steer = -0.8; inp.throttle = 0.0; inp.brake = 1.0
    elif t < 3.5:
        inp.steer = -0.8; inp.throttle = 0.0; inp.brake = 0.0
        inp.handbrake = True
    else:
        inp.steer = 0.0; inp.throttle = 0.5; inp.brake = 0.0
        inp.handbrake = False
    inp.fast_steer = False
    inp.wheelspin = False
    inp.handbrake = getattr(inp, "handbrake", False)
    return inp


def run_scenario(frames=300):
    cos = load_cosmetics()
    v = Vehicle(cos)
    v.reset(0.0, 0.0, 0.0)
    world = PhysicsWorld(Track())
    world.add(v)
    out = []
    for i in range(frames):
        inp = _inputs_for_frame(i)
        v.process_input(inp)
        world.step(1.0 / 60.0)
        out.append({
            "frame": i,
            "x": v.position.x,
            "z": v.position.z,
            "speed": round(v.speed, 6),
            "yaw": v.orientation,
            "wheel_angle": v.wheel_angle,
        })
    return out


def test_golden_trajectory():
    traj = run_scenario()
    os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
    if not os.path.exists(GOLDEN_PATH):
        with open(GOLDEN_PATH, "w") as f:
            json.dump(traj, f, indent=1)
        # first run generates; pass
        return
    with open(GOLDEN_PATH) as f:
        golden = json.load(f)
    assert len(golden) == len(traj), "trajectory length mismatch"
    for g, t in zip(golden, traj):
        assert g == t, f"frame {g['frame']} mismatch:\n  golden={g}\n  actual={t}"
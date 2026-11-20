"""Pinned digests of what the world generator makes, so moving code round
(worldgen/ out of world/) provably changes nothing.  A digest changes only
when generation does; then re-pin on purpose (python tests/test_worldgen_pins.py)."""

import hashlib
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from world.world import World  # noqa: E402
from worldgen.chunk import chunk_mesh  # noqa: E402
from worldgen.roads.lanes import LaneMap  # noqa: E402

SEEDS = (42, 1977)
PINS = {
    42: {
        "region_weights": "19c522eb6d58c642",
        "region_heights": "1eeff7674946a3f1",
        "world_heights": "e2ae6e64170147f7",
        "chunk_meshes": "38fe428565cae69a",
        "road_meshes": "e1483bf848e9e608",
        "road_graph": "a9aac12c035d54ec",
        "roads_near": "99d474ca054dcf81",
        "lanes": "342670a55ddce161",
        "ground_query": "de7e884ebfa37d2c",
    },
    1977: {
        "region_weights": "63337432d861a2ad",
        "region_heights": "59f7adc18b4f5f0a",
        "world_heights": "a1c5b90bf952f2a7",
        "chunk_meshes": "92b1785098af5fa0",
        "road_meshes": "0dc3c2c68367703d",
        "road_graph": "655af21de532fb74",
        "roads_near": "60e5e86e452a15d1",
        "lanes": "599e1b84efe222ff",
        "ground_query": "96725d0cd61712f4",
    },
}


def _digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        if p is None:
            h.update(b"none")
            continue
        a = np.asarray(p, float)
        h.update(str(a.shape).encode())
        h.update(np.round(a, 6).tobytes())
    return h.hexdigest()[:16]


def digests(seed):
    """{name: digest} of one seed's world, from the regions to the lanes."""
    w = World(seed=seed)
    out = {}
    g = np.linspace(-6000.0, 6000.0, 41)
    gx, gz = (a.ravel() for a in np.meshgrid(g, g))
    out["region_weights"] = _digest(*w.regions.weights(gx, gz))
    out["region_heights"] = _digest(*w.regions.sample(gx, gz))
    x0, z0, _ = w.spawn(0.0, 0.0)
    g = np.linspace(-300.0, 300.0, 31)
    gx, gz = (a.ravel() for a in np.meshgrid(g + x0, g + z0))
    out["world_heights"] = _digest(*w.query(gx, gz))
    ci, ck = int(x0 // 64), int(z0 // 64)
    keys = [(ci, ck), (ci + 1, ck - 2), (ci - 3, ck + 1), (0, 0), (-5, 7)]
    meshes = [chunk_mesh(w, *k, w.spec) for k in keys]
    out["chunk_meshes"] = _digest(*[a for m in meshes
                                    for a in (m.pos, m.normal, m.color, m.uv, m.tris)])
    roads = [w.road_mesh(*k) for k in keys]
    out["road_meshes"] = _digest(*[a for m in roads for a in
                                   ((None,) if m is None else (m.pos, m.normal, m.color, m.uv,
                                                               m.tris))])
    gr = w.roads.graph(0, 0)
    out["road_graph"] = _digest(gr.x, gr.z, gr.h, np.array(gr.edges))
    near = w.roads.roads_near(x0 - 600.0, z0 - 600.0, x0 + 600.0, z0 + 600.0)
    out["roads_near"] = _digest(np.array([r.key for r in near]), *[r.pts for r in near])
    lm = LaneMap(w.roads)
    lanes = lm.lanes_near(x0, z0, 300.0)
    out["lanes"] = _digest(*[ln.pts for ln in lanes],
                           np.array([[len(lm.exits(ln))] for ln in lanes]))
    rng = np.random.default_rng(seed)
    pts = np.stack([x0 + rng.uniform(-200, 200, 60), z0 + rng.uniform(-200, 200, 60)], 1)
    q = [w(float(x), float(z)) for x, z in pts]
    out["ground_query"] = _digest(np.array([[h, *n, s] for h, n, s in q]))
    return out


@pytest.mark.parametrize("seed", SEEDS)
def test_world_generation_matches_the_pins(seed):
    assert digests(seed) == PINS[seed]


if __name__ == "__main__":
    print({seed: digests(seed) for seed in SEEDS})

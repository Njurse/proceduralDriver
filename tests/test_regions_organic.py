"""Organic region borders (worldgen.regions): domain warp and encroach."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from worldgen.biomes import parse_biome  # noqa: E402
from worldgen.regions import Regions  # noqa: E402
from worldgen.spec import WorldSpec  # noqa: E402

BIOMES = (parse_biome({"name": "a", "terrain": {"amplitude_m": 20.0}}),
          parse_biome({"name": "b", "encroach": 0.6, "terrain": {"base_height_m": 10.0}}),
          parse_biome({"name": "c", "terrain": {"amplitude_m": 5.0, "base_height_m": -5.0}}))
PTS = np.random.default_rng(2).uniform(-6000.0, 6000.0, (400, 2))


def _regions(warp=300.0, biomes=BIOMES):
    return Regions(WorldSpec(seed=11, biomes=biomes, region_warp_m=warp))


def test_warped_weights_sum_to_one_and_carry_their_gradient():
    r = _regions()
    x, z = PTS[:, 0], PTS[:, 1]
    w = r.weights(x, z)[0]
    assert np.allclose(w.sum(axis=1), 1.0) and (w >= 0.0).all()
    h, hx, hz = r.sample(x, z)
    e = 1e-3
    assert np.allclose((r.sample(x + e, z)[0] - r.sample(x - e, z)[0]) / (2 * e), hx, atol=1e-5)
    assert np.allclose((r.sample(x, z + e)[0] - r.sample(x, z - e)[0]) / (2 * e), hz, atol=1e-5)
    for i in range(40):   # the physics path agrees with the mesh path
        assert np.allclose(r.sample(float(x[i]), float(z[i])), (h[i], hx[i], hz[i]))


def test_warp_moves_the_borders_and_is_deterministic():
    flat, warped = _regions(0.0), _regions()
    g = np.linspace(-4000.0, 4000.0, 120)
    gx, gz = (a.ravel() for a in np.meshgrid(g, g))
    a, b = flat.biome_index(gx, gz), warped.biome_index(gx, gz)
    assert 0.05 < np.mean(a != b) < 0.5
    assert np.array_equal(b, _regions().biome_index(gx, gz))


def test_encroach_grows_a_biome():
    plain = tuple(parse_biome({"name": n}) for n in "abc")
    pushy = (plain[0], parse_biome({"name": "b", "encroach": 1.0}), plain[2])
    g = np.linspace(-8000.0, 8000.0, 160)
    gx, gz = (a.ravel() for a in np.meshgrid(g, g))
    share = [np.mean(_regions(biomes=bs).biome_index(gx, gz) == 1) for bs in (plain, pushy)]
    assert share[1] > 1.2 * share[0]


def test_site_of_lands_in_its_biome():
    r = _regions()
    for name in "abc":
        assert r.biome_at(*r.site_of(name)).name == name

"""Engine-free world generation: pure Python + numpy, no Panda3D.

    hashing   splitmix hashes: the only randomness
    spec      WorldSpec: everything a world is generated from
    biomes    biome definitions (JSON) and their rules
    height    a biome's height field (fbm / ridged / terraced noise)
    palette   vertex colours from height and slope
    regions   biome regions, blended heights and colours
    ground    the ground contract the physics queries, and its surface codes

The game (world/, game/) adapts this to Panda3D and config.py; nothing in
here may import either (tests/test_worldgen_pure.py).
"""

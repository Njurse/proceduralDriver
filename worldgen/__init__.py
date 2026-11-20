"""Engine-free world generation: pure Python + numpy, no Panda3D.

    hashing   splitmix hashes: the only randomness
    spec      WorldSpec: everything a world is generated from
    biomes    biome definitions (JSON) and their rules
    height    a biome's height field (fbm / ridged / terraced noise)
    palette   vertex colours from height and slope
    regions   biome regions, blended heights and colours
    ground    the ground contract the physics queries, and its surface codes
    roads/    road graph (layout), polylines and profiles (road), strip and
              marking meshes (mesh), lane paths (lanes); RoadNetwork caches
              and prefetches them
    chunk     terrain chunk meshes (ChunkMesh) and chunk keys
    query     WorldGround(spec): the whole generated world as one Ground
    traffic   kinematic lane-following traffic (TrafficSpec, Traffic)

Every setting arrives in a WorldSpec (or a TrafficSpec), seed included.  The
game (world/, game/) builds those from config.py and adapts the output to
Panda3D; nothing in here may import either (tests/test_worldgen_pure.py).
"""

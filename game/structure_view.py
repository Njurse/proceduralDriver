"""Draws worldgen structures: one merged GeomNode per chunk (game/world_view
builds it from WorldGround.structure_mesh), unlit, its shading and neon trims
already in the vertex colours."""


def style_structures(node):
    """Panda state for a chunk's structure node."""
    node.setLightOff(1)        # flat-shaded in the colours; emissive trims stay bright
    node.setTextureOff(1)      # not the terrain's detail texture
    return node

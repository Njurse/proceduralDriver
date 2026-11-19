"""Car cosmetics profile loader for the REDRIVER2 handling model.

Loads ``assets/vehicle/havana_model0/handling.json`` into a ``CarCosmetics``
dataclass, remapping the stale BMP/sound paths to the real PNG/WAV assets, and
loads the OBJ mesh with an ``invert_vertical`` flag (see config).
"""

import json
import os
from dataclasses import dataclass, field

try:
    import config
except ModuleNotFoundError:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config


@dataclass
class CarCosmetics:
    model: int = 0
    wheel_disp: list = field(default_factory=list)   # 4 (x,y,z) offsets
    power_ratio: int = 4096
    mass: int = 4096
    traction: int = 4096
    sus_coeff: int = 4096
    wheel_size: int = 49
    c_points: list = field(default_factory=list)     # 12 (x,y,z)
    col_box: tuple = (0, 0, 0)
    cog: tuple = (0, 0, 0)
    twist_rate_x: int = 0
    twist_rate_y: int = 0
    twist_rate_z: int = 0
    extra_info: int = 0
    cb_y_offset: int = 0
    # resolved asset paths
    wheel_textures: dict = field(default_factory=dict)
    sounds: dict = field(default_factory=dict)
    # mesh
    mesh_vertices: list = field(default_factory=list)  # list of (x,y,z)
    mesh_faces: list = field(default_factory=list)     # list of vertex-index tuples
    invert_vertical: bool = False
    raw: str = ""


def _remap_wheel_textures(base_dir):
    # Real PNGs live under assets/vehicle/wheels/havana/
    wdir = os.path.join(base_dir, "..", "wheels", "havana")
    wdir = os.path.normpath(wdir)
    return {
        "clean": os.path.join(wdir, "wheelclean.png"),
        "damaged": os.path.join(wdir, "wheeldam_fixed.png"),
        "fast": os.path.join(wdir, "wheelfast.png"),
    }


def _remap_sounds(base_dir):
    sdir = os.path.join(base_dir, "..", "..", "sounds", "vehicle", "havana_model0")
    sdir = os.path.normpath(sdir)
    common = os.path.join(os.path.dirname(sdir), "common")
    return {
        "engine": os.path.join(sdir, "engine.wav"),
        "idle": os.path.join(sdir, "idle.wav"),
        "skid": os.path.join(common, "skid.wav"),
    }


def load_obj(path, invert_vertical=False):
    """Parse a minimal Wavefront OBJ into (vertices, faces)."""
    verts = []
    faces = []
    if not os.path.exists(path):
        return verts, faces
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] == "v":
                x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                if invert_vertical:
                    y = -y
                verts.append((x, y, z))
            elif parts[0] == "f":
                idx = []
                for p in parts[1:]:
                    # handle v/vt/vn
                    v = p.split("/")[0]
                    if v:
                        idx.append(int(v) - 1)
                if len(idx) >= 3:
                    faces.append(tuple(idx))
    return verts, faces


def load_cosmetics(path=None, invert_vertical=None):
    if path is None:
        path = config.HAVANA_HANDLING_JSON
    if invert_vertical is None:
        invert_vertical = getattr(config, "CAR_MODEL_INVERT_VERTICAL", False)
    with open(path, "r") as f:
        data = json.load(f)

    h = data.get("handling", {})
    base_dir = os.path.dirname(path)

    cos = CarCosmetics()
    cos.model = data.get("model", 0)
    cos.wheel_disp = [tuple(w) for w in h.get("wheelDisp", [])]
    cos.power_ratio = h.get("powerRatio", 4096)
    cos.mass = h.get("mass", 4096)
    cos.traction = h.get("traction", 4096)
    cos.sus_coeff = h.get("susCoeff", 4096)
    cos.wheel_size = h.get("wheelSize", 49)
    cos.c_points = [tuple(c) for c in h.get("cPoints", [])]
    cos.col_box = tuple(h.get("colBox", [0, 0, 0]))
    cos.cog = tuple(h.get("cog", [0, 0, 0]))
    cos.twist_rate_x = h.get("twistRateX", 0)
    cos.twist_rate_y = h.get("twistRateY", 0)
    cos.twist_rate_z = h.get("twistRateZ", 0)
    cos.extra_info = h.get("extraInfo", 0)
    cos.cb_y_offset = h.get("cbYoffset", 0)
    cos.wheel_textures = _remap_wheel_textures(base_dir)
    cos.sounds = _remap_sounds(base_dir)
    cos.invert_vertical = bool(invert_vertical)
    cos.raw = data.get("raw", "")

    # load mesh (clean)
    obj_path = getattr(config, "HAVANA_MODEL_OBJ", None)
    if obj_path:
        v, fc = load_obj(obj_path, invert_vertical=cos.invert_vertical)
        cos.mesh_vertices = v
        cos.mesh_faces = fc

    return cos


if __name__ == "__main__":
    c = load_cosmetics()
    print("power_ratio", c.power_ratio, "mass", c.mass, "traction", c.traction)
    print("wheel_disp", c.wheel_disp)
    print("mesh verts", len(c.mesh_vertices), "faces", len(c.mesh_faces))
    print("invert_vertical", c.invert_vertical)
    print("sounds", c.sounds)
    print("wheels", c.wheel_textures)
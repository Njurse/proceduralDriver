"""worldgen/ stays engine-free: no Panda3D, no game modules, no config.py,
so it can be lifted into another engine as it is."""

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANNED = ("panda3d", "direct", "pygame", "config", "core", "world", "game", "ui", "main")


def test_worldgen_imports_nothing_from_the_engine_or_the_game():
    bad = []
    for path in sorted((ROOT / "worldgen").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            bad += [f"{path.name}: {n}" for n in names if n.split(".")[0] in BANNED]
    assert not bad


def test_importing_worldgen_loads_no_engine():
    code = ("import sys, pkgutil, importlib, worldgen\n"
            "for m in pkgutil.walk_packages(worldgen.__path__, 'worldgen.'):\n"
            "    importlib.import_module(m.name)\n"
            "print(sorted(k for k in sys.modules if k.split('.')[0] in "
            "('panda3d', 'direct', 'pygame', 'config')))")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True,
                         text=True, check=True, env={**os.environ, "PYTHONPATH": str(ROOT)})
    assert out.stdout.strip() == "[]"

"""Entry point for the procedural galaxy explorer.

Normal play:      python main.py
New galaxy:       python main.py --new
Fixed seed:       python main.py --seed 12345
Headless check:   python main.py --smoke
"""

import argparse
import logging
import sys
import tempfile
from pathlib import Path

log = logging.getLogger("galaxy_explorer")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fly a spaceship through a procedural galaxy.")
    parser.add_argument("--seed", type=int, default=None,
                        help="galaxy seed (defaults to the saved seed)")
    parser.add_argument("--new", action="store_true",
                        help="ignore the existing save and start over")
    parser.add_argument("--offscreen", action="store_true",
                        help="run without a visible window for benchmarking")
    parser.add_argument("--smoke", action="store_true",
                        help="render one frame of each scene, report, and exit")
    parser.add_argument("--frames", type=int, default=180,
                        help="frame count for --offscreen (default 180)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args(argv)


def report(results):
    """Print pixel statistics so headless runs can be validated."""
    import numpy as np
    from PIL import Image

    passed = True
    for path, label in results:
        path = Path(path)
        if not path.exists():
            log.warning("%s MISSING %s", label, path)
            passed = False
            continue
        array = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        flat = array.reshape(-1, 3)
        background = np.median(flat, axis=0)
        difference = np.abs(array - background).sum(axis=2)
        coverage = float((difference > 20).mean())
        mean = flat.mean(axis=0).astype(int).tolist()
        log.info("%s %s mean=%s coverage=%.3f", label, path.name, mean,
                 coverage)
        if coverage < 0.02:
            passed = False
    return passed


def main(argv=None):
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s")

    from game import GalaxyExplorer

    options = {
        "seed": args.seed,
        "new_game": args.new,
        "window_type": "offscreen" if (args.offscreen or args.smoke)
        else "onscreen",
    }
    if args.offscreen:
        options["persist"] = False

    if args.smoke:
        options["save_path"] = (Path(tempfile.gettempdir())
                                / "galaxy_explorer_smoke.json")
        game = GalaxyExplorer(options)
        results = game.run_smoke()
        passed = report(results)
        game.destroy()
        # Keep this on stdout for CI/smoke harnesses that grep for it.
        print("SMOKE OK" if passed else "SMOKE FAILED")
        log.info("smoke %s", "ok" if passed else "failed")
        return 0 if passed else 1

    game = GalaxyExplorer(options)
    if args.offscreen:
        frames = max(1, args.frames)
        for _ in range(frames):
            game.taskMgr.step()
            game.graphicsEngine.renderFrame()
        game.destroy()
        log.info("Rendered %d offscreen frames.", frames)
        return 0

    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

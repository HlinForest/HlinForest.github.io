"""Run with the pinned ManimGL environment's Python: python render.py.

Pass --groups heads,linear-normalizer for a focused rerender.
Every layout contains complete mathematical objects. The only raster crop
removes the outside white margin; mobile never splices fragments of diagrams.
"""
import argparse
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--groups", default="")
parser.add_argument("--layout", choices=["desktop","mobile","both"], default="both")
args = parser.parse_args()
assert importlib.metadata.version("manimgl") == "1.7.2"
for layout in (["desktop","mobile"] if args.layout=="both" else [args.layout]):
    env=os.environ.copy()
    env.update(ATTENTION_LAYOUT=layout, ATTENTION_GROUPS=args.groups,
               PYTHONUNBUFFERED="1")
    subprocess.run(
        [sys.executable, "-m", "manimlib", "storyboard.py", "Storyboard",
         "-w", "-s", "-r", "2100x1200" if layout=="desktop" else "1050x2100",
         "-c", "white", "--video_dir", "../../tmp/attention-manim-render",
         "--file_name", f"last-{layout}"],
        cwd=HERE, env=env, check=True)

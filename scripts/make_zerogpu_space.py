#!/usr/bin/env python
"""Assemble the ZeroGPU Space in build/zerogpu-space/: spaces/zerogpu/* plus the bench modules it
imports (flat, next to app.py). Upload that folder's files to a Gradio Space with ZeroGPU hardware."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "build" / "zerogpu-space"
shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
for f in (ROOT / "spaces" / "zerogpu").iterdir():
    shutil.copy(f, OUT / f.name)
for name in ("measure.py", "model_list.py", "run_direct.py"):
    shutil.copy(ROOT / "bench" / name, OUT / name)
print("wrote", OUT, sorted(p.name for p in OUT.iterdir()))

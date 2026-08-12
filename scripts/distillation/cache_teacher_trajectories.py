#!/usr/bin/env python3
"""CLI bootstrap for :mod:`cache_teacher_trajectories_impl`."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.cache_teacher_trajectories_impl import *  # noqa: F401,F403,E402
from scripts.distillation.cache_teacher_trajectories_impl import _generate_one  # noqa: F401,E402

if __name__ == "__main__":
    raise SystemExit(main())

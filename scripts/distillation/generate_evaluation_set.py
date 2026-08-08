#!/usr/bin/env python3
"""CLI bootstrap for fair matched teacher/student evaluation generation."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.generate_evaluation_set_v2 import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    raise SystemExit(main())

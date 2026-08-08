#!/usr/bin/env python3
"""CLI bootstrap for canonical phased LoRA training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.distillation.train_phased_distill_lora_v2 import *  # noqa: F401,F403,E402

if __name__ == "__main__":
    raise SystemExit(main())

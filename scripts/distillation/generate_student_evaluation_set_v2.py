"""Cached-prompt compatibility layer for student-only evaluation generation."""

from pathlib import Path

from scripts.distillation import generate_student_evaluation_set_v1 as _v1
from scripts.distillation.evaluation_prompt_cache import DEFAULT_CACHE, load_evaluation_prompt_cache
from scripts.distillation.generate_student_evaluation_set_v1 import *  # noqa: F401,F403

_original_build_parser = _v1.build_parser


def build_parser():
    parser = _original_build_parser()
    parser.add_argument("--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE)
    return parser


def _cached_prompt_features(records, args):
    return load_evaluation_prompt_cache(
        args.evaluation_prompt_cache, records, component_model=args.component_model
    )


_v1.build_parser = build_parser
_v1._encode_unique_prompts = _cached_prompt_features
generate_student_set = _v1.generate_student_set
main = _v1.main

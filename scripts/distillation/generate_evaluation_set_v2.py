"""Cached-prompt layer over the fair teacher/student evaluation generator."""

from pathlib import Path

from scripts.distillation import generate_evaluation_set_v2_uncached as _uncached
from scripts.distillation.evaluation_prompt_cache import DEFAULT_CACHE, load_evaluation_prompt_cache
from scripts.distillation.generate_evaluation_set_v2_uncached import *  # noqa: F401,F403

_impl = _uncached._impl
_original_build_parser = _impl.build_parser


def build_parser():
    parser = _original_build_parser()
    parser.add_argument("--evaluation-prompt-cache", type=Path, default=DEFAULT_CACHE)
    return parser


def _cached_prompt_features(records, args):
    return load_evaluation_prompt_cache(
        args.evaluation_prompt_cache, records, component_model=args.component_model
    )


_impl.build_parser = build_parser
_impl._encode_unique_prompts = _cached_prompt_features
generate_sets = _impl.generate_sets
main = _impl.main

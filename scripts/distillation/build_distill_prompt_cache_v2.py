"""Compatibility layer for canonical manifests without a category field."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from scripts.distillation import build_distill_prompt_cache_impl as _impl
from scripts.distillation.build_distill_prompt_cache_impl import *  # noqa: F401,F403


_load_original_bundle = _impl.load_latent_bundle


def _load_bundle_with_categories(path):
    bundle = _load_original_bundle(path)
    manifest = tuple(
        {
            **row,
            "category": row.get(
                "category", str(row["sample_id"]).split("/", 1)[0]
            ),
        }
        for row in bundle.manifest
    )
    return replace(bundle, manifest=manifest)


_impl.load_latent_bundle = _load_bundle_with_categories
build_assets = _impl.build_assets


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_assets(args)
    return 0


from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath

import pytest
import torch

from scripts.training.train_local_latent_lora import (
    AssetValidationError,
    EXPECTED_MANIFEST_FINGERPRINT,
    audit_image_archive,
    deterministic_subset_ids,
    load_latent_bundle,
    load_prompt_cache,
    main,
)


ROOT = Path(__file__).resolve().parents[1]
LATENT_BUNDLE = ROOT / "data" / "archives" / "clean_latents_512.zip"
IMAGE_ARCHIVE = ROOT / "data" / "archives" / "ink.zip"


@pytest.fixture(scope="session")
def latent_bundle():
    return load_latent_bundle(LATENT_BUNDLE)


def _prompt_cache(sample_ids: list[str]) -> dict[str, object]:
    rows = len(sample_ids)
    embeds = torch.empty((rows, 300, 4096), dtype=torch.float16)
    for index in range(rows):
        embeds[index].fill_(index + 1)
    return {
        "format_version": 1,
        "sample_ids": sample_ids,
        "prompt_embeds": embeds,
        "attention_masks": torch.ones((rows, 300), dtype=torch.bool),
        "max_sequence_length": 300,
        "text_encoder_model": "test/t5",
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
    }


def _save_cache(tmp_path: Path, cache: dict[str, object]) -> Path:
    path = tmp_path / "prompt_cache.pt"
    torch.save(cache, path)
    return path


def test_archives_are_safe_and_uncorrupted() -> None:
    for path in (IMAGE_ARCHIVE, LATENT_BUNDLE):
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
            assert all(
                not PurePosixPath(name).is_absolute()
                and ".." not in PurePosixPath(name).parts
                for name in archive.namelist()
            )


def test_latent_bundle_contract(latent_bundle) -> None:
    assert latent_bundle.latents.shape == (260, 4, 64, 64)
    assert latent_bundle.latents.dtype == torch.float16
    assert torch.isfinite(latent_bundle.latents).all()
    assert len(latent_bundle.sample_ids) == 260
    assert latent_bundle.metadata["resolution"] == 512
    assert latent_bundle.metadata["scaling_factor"] == pytest.approx(0.13025)
    assert (
        latent_bundle.metadata["manifest_fingerprint"]
        == EXPECTED_MANIFEST_FINGERPRINT
    )


def test_image_archive_matches_latent_manifest(latent_bundle) -> None:
    audit = audit_image_archive(IMAGE_ARCHIVE, latent_bundle)
    assert audit == {
        "num_images": 260,
        "num_captions": 260,
        "categories": {
            "animal": 30,
            "others": 10,
            "plant": 209,
            "web": 11,
        },
        "manifest_fingerprint": EXPECTED_MANIFEST_FINGERPRINT,
    }


def test_deterministic_subsets_are_reproducible_and_nested(
    latent_bundle,
) -> None:
    ids_50 = deterministic_subset_ids(latent_bundle.sample_ids, 50, 42)
    ids_100 = deterministic_subset_ids(latent_bundle.sample_ids, 100, 42)
    ids_260 = deterministic_subset_ids(latent_bundle.sample_ids, 260, 42)
    assert ids_50[:5] == (
        "plant/220",
        "plant/154",
        "plant/80",
        "plant/132",
        "animal/184",
    )
    assert ids_50 == ids_100[:50]
    assert ids_100 == ids_260[:100]
    assert ids_50 == deterministic_subset_ids(
        latent_bundle.sample_ids, 50, 42
    )
    assert ids_50 != deterministic_subset_ids(
        latent_bundle.sample_ids, 50, 43
    )


def test_prompt_cache_is_reordered_by_sample_id(
    tmp_path: Path,
    latent_bundle,
) -> None:
    selected = list(latent_bundle.sample_ids[:3])
    cache_order = [selected[2], selected[0], selected[1]]
    path = _save_cache(tmp_path, _prompt_cache(cache_order))
    features = load_prompt_cache(path, selected)
    assert features.sample_ids == tuple(selected)
    assert features.prompt_embeds[:, 0, 0].tolist() == [2.0, 3.0, 1.0]
    assert features.attention_masks.dtype == torch.bool


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_ids", "duplicate IDs"),
        ("missing_id", "does not cover"),
        ("wrong_fingerprint", "fingerprint"),
        ("wrong_shape", "shape"),
        ("wrong_dtype", "float16"),
        ("non_finite", "NaN or infinite"),
        ("bad_mask", "0/1"),
    ],
)
def test_invalid_prompt_cache_fails_fast(
    tmp_path: Path,
    latent_bundle,
    mutation: str,
    message: str,
) -> None:
    selected = list(latent_bundle.sample_ids[:2])
    cache = _prompt_cache(selected)
    if mutation == "duplicate_ids":
        cache["sample_ids"] = [selected[0], selected[0]]
    elif mutation == "missing_id":
        cache = _prompt_cache([selected[0]])
    elif mutation == "wrong_fingerprint":
        cache["manifest_fingerprint"] = "wrong"
    elif mutation == "wrong_shape":
        cache["prompt_embeds"] = torch.zeros(
            (2, 299, 4096), dtype=torch.float16
        )
    elif mutation == "wrong_dtype":
        cache["prompt_embeds"] = torch.zeros(
            (2, 300, 4096), dtype=torch.float32
        )
    elif mutation == "non_finite":
        cache["prompt_embeds"][0, 0, 0] = float("nan")
    elif mutation == "bad_mask":
        cache["attention_masks"] = torch.full(
            (2, 300), 2, dtype=torch.int64
        )
    path = _save_cache(tmp_path, cache)
    with pytest.raises(AssetValidationError, match=message):
        load_prompt_cache(path, selected)


def test_validate_assets_only_passes_without_prompt_cache(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing_prompt = tmp_path / "not_created.pt"
    result = main(
        [
            "--latent-bundle",
            str(LATENT_BUNDLE),
            "--image-archive",
            str(IMAGE_ARCHIVE),
            "--prompt-cache",
            str(missing_prompt),
            "--num-images",
            "50",
            "--validate-assets-only",
        ]
    )
    output = capsys.readouterr().out
    assert result == 0
    assert "PASS: local image and latent assets" in output
    assert "PENDING: prompt embedding cache is not present yet" in output

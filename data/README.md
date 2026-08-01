# Local data assets

This directory separates tracked source archives, future precomputed features, and ignored extracted or generated data. Training should consume the archives and feature caches through their validation contracts instead of assuming filesystem row order.

## Tracked archives

### `archives/ink.zip`

- 260 RGB JPG images and 260 same-name UTF-8 captions.
- Canonical categories: `plant=209`, `animal=30`, `web=11`, `others=10`.
- No missing caption, orphan caption, corrupt file, or unsafe ZIP path.
- After normalizing caption newlines to LF, its canonical manifest fingerprint is `b9d3c2d1d404`.

### `archives/clean_latents_512.zip`

- Produced by `notebooks/preprocessing/pixart_clean_latents_colab.ipynb`.
- Contains a tensor cache, `manifest.jsonl`, `validation_summary.json`, and reconstruction samples.
- Clean scaled VAE latents: shape `[260, 4, 64, 64]`, dtype `torch.float16`, all finite.
- Image resolution: 512 x 512.
- VAE scaling factor: `0.13025`.
- Component repository: `PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers`.
- Transformer checkpoint: `PixArt-alpha/PixArt-Sigma-XL-2-512-MS`.
- The manifest and sample ID order exactly match `ink.zip`; fingerprint `b9d3c2d1d404`.

The local trainer reads the latent `.pt` member in memory. It does not extract the source images and does not rerun the VAE.

## Prompt embedding cache contract

The validated local prompt cache is stored at:

```text
data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt
data/features/validation_summary.json
```

Both files are intentionally Git-ignored; transfer them through the project shared storage after cloning or pulling the repository. The validation summary must report `PASS` and match the cache filename, shapes, dtypes, models, paired latent cache, and manifest fingerprint. The cache must be loadable with `torch.load(path, map_location="cpu", weights_only=True)` and contain exactly this public contract (additional keys are allowed):

```python
{
    "format_version": 1,
    "sample_ids": list[str],
    "prompt_embeds": Tensor[N, 300, 4096],  # float16, CPU
    "attention_masks": Tensor[N, 300],      # bool or int64, CPU
    "empty_prompt_embeds": Tensor[1, 300, 4096],  # float16, CPU
    "empty_prompt_attention_mask": Tensor[1, 300], # bool or int64
    "max_sequence_length": 300,
    "text_encoder_model": str,
    "manifest_fingerprint": "b9d3c2d1d404",
}
```

Rules enforced by `scripts/training/train_local_latent_lora.py`:

- `sample_ids` may be in any order; features are reordered by ID.
- A cache may cover only a planned smoke subset, but it must contain every selected ID.
- IDs must be non-empty and unique.
- `prompt_embeds` must be float16, CPU-loadable, finite, and exactly `[N, 300, 4096]`.
- `attention_masks` must be bool or int64, exactly `[N, 300]`, and contain only 0/1.
- `empty_prompt_embeds` must be finite float16 with shape `[1, 300, 4096]`.
- `empty_prompt_attention_mask` must be bool or int64 with shape `[1, 300]` and only 0/1; these tensors provide the unconditional CFG branch when guidance is greater than 1.0.
- The manifest fingerprint must be `b9d3c2d1d404`.
- Missing/duplicate IDs, missing keys, invalid shapes or dtypes, non-finite tensors, or a wrong fingerprint fail before any base model download.

Prompt embeddings must be generated from the captions represented by the canonical manifest. The teammate producing the cache owns T5/tokenizer loading and records the actual text encoder model in `text_encoder_model`. The local training code owns validation, ID alignment, LoRA training, adapter save/reload, and image generation. It never substitutes random or zero embeddings.

## Validation and training

Validate the source archive, latent bundle, and any prompt cache currently present:

```powershell
python scripts/training/train_local_latent_lora.py --validate-assets-only
```

With both feature files present, validation succeeds only after the prompt cache and its validation summary agree with the image and latent contracts.

Run the local training smoke test:

```powershell
python scripts/training/train_local_latent_lora.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --num-images 50 `
  --rank 8 `
  --max-train-steps 100 `
  --output-dir outputs/local_smoke/r8_n50_steps100
```

The deterministic subset is ranked by `sha256(seed + sample_id)` semantics in the trainer, so the default seed 42 produces nested 50 / 100 / 260 experiment sets. Each run writes `subset_manifest.json`; completed training additionally writes the PEFT adapter, `run_metadata.json`, and `reload_generation.png` under the ignored `outputs/` directory.

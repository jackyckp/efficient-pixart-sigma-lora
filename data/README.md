# Data Assets & Validation Contracts

This directory specifies the layout, schema contracts, and validation procedures for local training and distillation data assets. Raw images, latent bundles, and precomputed embeddings are intentionally excluded from the Git repository due to file size and distribution policies.

---

## 1. Directory Structure

Place required assets in the following hierarchy under `data/`:

```text
data/
├── archives/
│   ├── ink.zip                                              # 260 raw image-caption pairs
│   └── clean_latents_512.zip                                # Precomputed 512px clean SDXL VAE latents
├── features/
│   ├── t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt       # Stage 1 Style Teacher prompt embeddings
│   ├── distill_t5_plant627_len300_fp16_v1.pt                # Stage 2 Distillation prompt-bank cache
│   └── distill_eval_t5_prompts30_len300_fp16_v1.pt          # Held-out 30-prompt evaluation cache
├── distillation/
│   └── plant_prompt_bank_v1.jsonl                           # 627 generated distillation prompt variants
└── README.md                                                # Asset specification & contract guide
```

---

## 2. Asset Specification & Integrity Contracts

| Asset Path | Format / Shape | Type / Precision | Integrity Contract |
| --- | --- | --- | --- |
| `data/archives/ink.zip` | 260 image-caption pairs | JPG + JSON manifest | Canonical manifest fingerprint: `b9d3c2d1d404` |
| `data/archives/clean_latents_512.zip` | `[260, 4, 64, 64]` | FP16, SDXL VAE (`0.13025`) | Must share manifest fingerprint `b9d3c2d1d404` with `ink.zip` |
| `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` | `[260, 300, 4096]` | FP16 T5-XXL | ID-aligned with manifest; valid attention masks `[260, 300]` |
| `data/features/distill_t5_plant627_len300_fp16_v1.pt` | `[627, 300, 4096]` | FP16 T5-XXL | 3 prompt variants per plant image (`original`, `subject`, `styled`) |
| `data/features/distill_eval_t5_prompts30_len300_fp16_v1.pt` | `[30, 300, 4096]` | FP16 T5-XXL | 30 held-out evaluation prompts across 4 domain categories |

### Contract Guarantees
- **Direct ZIP Streaming**: Training scripts read latent tensors directly from `clean_latents_512.zip` into GPU memory without unpacking images to disk or re-running the VAE encoder.
- **CPU & Weights-Only Compatibility**: All `.pt` prompt caches are verified loadable via `torch.load(..., weights_only=True)`.
- **Finite Values**: Latents and prompt embeddings are strictly asserted to contain non-NaN, non-infinite values and binary (`0/1`) attention masks.

---

## 3. Asset Validation

Before launching training or distillation pipelines, validate the presence, schema, and checksums of local assets:

```powershell
py -3.11.2 scripts/training/train_local_latent_lora.py --validate-assets-only
```

For distillation prompt cache validation:

```powershell
py -3.11.2 -m pytest tests/test_local_asset_contract.py tests/test_distillation_cache_contract.py
```

---

## 4. Rebuilding Assets from Scratch (Optional)

If starting from scratch without precomputed archives:

### 4.1 Data Scraping & Preparation
Scrape raw Chinese ink-wash paintings into category folders:
```powershell
py -3.11.2 scripts/data/download_tappu.py
```

### 4.2 Automated VLM Captioning
Generate descriptive captions with domain trigger words using Florence-2 / JoyCaption:
```powershell
py -3.11.2 scripts/data/auto_caption.py `
  --dir data/ink/plant `
  --model florence-2 `
  --trigger "traditional Chinese ink wash painting style, shuimo hua"
```

### 4.3 Building Distillation Prompt Bank & T5-XXL Embeddings
Generate the 627-prompt bank and encode prompt embeddings with T5-XXL:
```powershell
py -3.11.2 scripts/distillation/build_distill_prompt_cache.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --source-prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --prompt-bank data/distillation/plant_prompt_bank_v1.jsonl `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-cache data/features/distill_t5_plant627_len300_fp16_v1.pt
```

---

## 5. Storage Policy

- **Do Not Commit**: Never commit files under `data/archives/`, `data/features/`, or `data/distillation/` to Git.
- **Verification**: Run `git status` to ensure only `data/README.md` is tracked in version control.

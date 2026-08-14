# Local data assets

The canonical 260-pair image/caption archive is versioned as `data/ink.zip`.
Large derived tensors are intentionally excluded from Git and must be obtained
from the team's shared storage or rebuilt locally.

| Expected path | Contents | Contract |
| --- | --- | --- |
| `data/ink.zip` | 260 JPG/caption pairs | Caption-manifest fingerprint `b9d3c2d1d404` |
| `data/archives/clean_latents_512.zip` | 260 clean VAE latents plus manifest | `[260, 4, 64, 64]`, FP16, 512 px, scaling factor `0.13025` |
| `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` | Original-caption T5 embeddings | FP16 `[N, 300, 4096]`, ID-aligned with the manifest |
| `data/features/distill_t5_plant627_len300_fp16_v1.pt` | 627-prompt distillation cache | FP16 embeddings and attention masks for trajectory training |
| `data/features/distill_eval_t5_prompts30_len300_fp16_v1.pt` | 30 held-out evaluation prompts | Reused by evaluations to avoid T5 re-encoding |

Validate local assets before downloading the base model or starting training:

```powershell
py -3.11 scripts/training/train_local_latent_lora.py --validate-assets-only
```

The trainer verifies IDs, fingerprints, tensor shapes, dtypes, finite values,
and the alignment between `data/ink.zip` and the latent manifest. Prompt caches
must be CPU-loadable with `torch.load(..., weights_only=True)` and contain
stable unique `sample_ids`, FP16 prompt embeddings, and valid attention masks.
See [INSTRUCTIONS.md](../INSTRUCTIONS.md) for generation and training commands.

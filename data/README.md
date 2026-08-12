# Local data assets (not in Git)

This directory is intentionally excluded from the final GitHub repository except for this file. Obtain the assets below through the team's approved shared storage; do not commit them or force-add them to Git.

| Expected path | Contents | Contract |
| --- | --- | --- |
| `data/archives/ink.zip` | 260 JPG/caption pairs | Canonical caption-manifest fingerprint: `b9d3c2d1d404` |
| `data/archives/clean_latents_512.zip` | 260 clean VAE latents plus manifest | `[260, 4, 64, 64]`, FP16, 512px, scaling factor `0.13025` |
| `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` | Original-caption T5 embeddings | FP16 `[N, 300, 4096]`, ID-aligned with the manifest |
| `data/features/distill_t5_plant627_len300_fp16_v1.pt` | 627-prompt distillation cache | FP16 embeddings and attention masks for trajectory training |
| `data/features/distill_eval_t5_prompts30_len300_fp16_v1.pt` | 30 held-out evaluation prompts | Reused by evaluations to avoid T5 re-encoding |

The local trainer validates IDs, fingerprints, tensor shapes, dtypes, and finite values before downloading the base model or starting optimization:

```powershell
py -3.11 scripts/training/train_local_latent_lora.py --validate-assets-only
```

`ink.zip` and the latent bundle must remain paired: their manifests share the fingerprint `b9d3c2d1d404`. The training scripts read latent tensors directly from the ZIP; they do not extract images or rerun the VAE.

Prompt caches must be CPU-loadable with `torch.load(..., weights_only=True)`, contain stable unique `sample_ids`, FP16 `prompt_embeds`, valid attention masks, and their stated manifest/prompt-bank fingerprints. Refer to `scripts/training/train_local_latent_lora.py` and `DISTILLATION.md` for the enforced schema.

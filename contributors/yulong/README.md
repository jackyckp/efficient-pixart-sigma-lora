# Yulong Sheng: LoRA Teacher and 20 -> 10 -> 5 Distillation

This directory contains my individual contribution to the PixArt-Sigma project. It preserves the code, run configuration, selected 5-step LoRA adapter, and matched evaluation evidence from my local RTX 4060 Laptop experiment.

## Contribution summary

1. Trained a rank-16 ink-wash style LoRA Teacher from precomputed 512 px image latents and T5 embeddings.
2. Selected the Teacher checkpoint at optimizer step 4,000 using held-out prompts and visual inspection.
3. Distilled the 20-step Teacher into a 10-step intermediate LoRA and then a 5-step Student.
4. Evaluated the 5-step checkpoints on 8 prompts and 3 seeds against both the 10-step Teacher and the original 20-step reference.
5. Selected the 8,000-update checkpoint as the canonical 5-step adapter.

## Final result

| Model | Inference steps | Guidance | Median time/image | Main role |
|---|---:|---:|---:|---|
| Style reference | 20 | 2.0 | 2.71 s | Original LoRA Teacher |
| Intermediate Teacher | 10 | 1.0 | 0.86 s | Stage-2 supervision |
| Selected Student | 5 | 1.0 | 0.51 s | Final deployable adapter |

Across 24 matched prompt/seed cases per model, the selected 5-step Student achieved:

- CLIP image similarity to the 10-step Teacher: **0.9557**
- CLIP image similarity to the 20-step reference: **0.9083**
- Median speedup over the 10-step Teacher: **1.70x**
- Median speedup over the 20-step reference: **5.35x**

Latency is hardware- and environment-dependent. CLIP similarity is used as a relative consistency metric, not as a complete measure of image quality; the experiment also used matched visual inspection for deformation and style retention.

## Directory map

```text
contributors/yulong/
|-- teacher_training/      # Local Teacher training code and selected checkpoint metadata
|-- distillation/          # 20 -> 10 -> 5 scripts, configs, and completion records
|-- artifacts/student_5step/ # Selected deployable PEFT LoRA adapter
|-- evaluation/            # Metrics, charts, prompt plan, and representative contact sheets
`-- data_manifest/         # Dataset/cache provenance without multi-GB cache files
```

## Reproduction order

1. Recreate the image-latent and T5-embedding caches described in `data_manifest/README.md`.
2. Run `teacher_training/train_pixart_sigma_lora_local.py` with rank 16, alpha 16, learning rate `1e-5`, batch size 1, and seed 42.
3. Use the selected step-4,000 Teacher with `distillation/run_stage1_v2.ps1` to train the 10-step intermediate model.
4. Use `distillation/run_stage2_10to5.ps1` to train the 5-step Student.
5. Use `distillation/evaluate_student_checkpoints.py` to reproduce the checkpoint comparison.

Large Hugging Face caches, VAE latent caches, T5 embedding caches, and base-model weights are intentionally excluded. They are reproducible inputs rather than authored model artifacts.


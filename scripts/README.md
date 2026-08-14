# Script entry points

Use the public CLI files below. Files ending in `_impl.py` contain the tested
implementation behind those CLIs and are not separate entry points.

## Data

- `data/precompute_clean_latents.py`: encode raw images from `data/ink.zip` into SDXL VAE clean latents and build `clean_latents_512.zip`.
- `data/precompute_t5_embeddings.py`: tokenize captions and precompute T5-XXL FP16 embeddings for the base dataset.
- `data/auto_caption.py`: optional tool to re-caption images using vision-language models.

## Style-teacher training and inference

- `training/train_local_latent_lora.py`: train one LoRA from validated local
  latent and prompt caches.
- `training/train_style_teacher_sweep.py`: run the four-rank 20-step teacher
  sweep.
- `inference/generate_with_prompt.py`: generate from an arbitrary prompt with
  the official model or a style-teacher adapter.

## Distillation

The complete workflow and commands are documented in
[`../INSTRUCTIONS.md`](../INSTRUCTIONS.md), with model-flow diagrams in
[`../ARCHITECTURE_DIAGRAMS.md`](../ARCHITECTURE_DIAGRAMS.md). The principal
entry points are:

- `distillation/build_distill_prompt_cache.py`
- `distillation/validate_style_teacher.py`
- `distillation/cache_teacher_trajectories.py`
- `distillation/train_phased_distill_lora.py`
- `distillation/generate_distilled.py`
- `distillation/generate_evaluation_set.py`
- `distillation/evaluate_distilled.py`

Experiment-specific runners under `distillation/` remain reproducibility
records for the reported Teacher A/B extensions. Evaluation-only utilities
live in `evaluation/`.

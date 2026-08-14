# Efficient PixArt-Sigma LoRA

This project adapts PixArt-Sigma 512 to Chinese ink-wash plant imagery, then distils a 20-step style teacher into 4-step and 2-step joint LoRA students. The GitHub repository contains code, experiment contracts, evaluation summaries, and a deliberately small curated visual set; raw data and model artifacts remain local or in shared storage.

## Primary result: Teacher B 6k -> 4-step / 2-step

The primary experiment extends Teacher B's 4-step student to 6k updates and trains a fresh 7k 2-step student from its selected four-step adapter. All results use held-out prompts, fixed seeds, 512 x 512 output, and guidance scale 1.0 for students.

| Model | Inference steps | Selected checkpoint | CLIPScore | Teacher retention | CMMD to plant data | Median latency | Speed-up | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Teacher B | 20 | style teacher | 0.3651 | 100% | 0.000949 | 2.871 s | 1.00x | Reference |
| Joint LoRA student | 4 | 4,500 | 0.3674 | 100.65% | 0.001087 | 0.480 s | 5.99x | PASS |
| Joint LoRA student | 2 | 7,000 | 0.3535 | 96.84% | 0.001394 | 0.244 s | 11.78x | PASS |

The complete history is in [evaluation/all_4step_2step_training_evaluation_results.csv](evaluation/all_4step_2step_training_evaluation_results.csv) and [evaluation/distillation_extended_results_v2.json](evaluation/distillation_extended_results_v2.json).

## Curated visual examples

Nine existing 512 x 512 images are versioned, with no new images generated for repository preparation:

- Three Teacher B 6k teacher-vs-4-step grids: [evaluation/examples/teacher_b_extend6k_4step](evaluation/examples/teacher_b_extend6k_4step)
- Three Teacher B 6k teacher-vs-2-step grids: [evaluation/examples/teacher_b_extend6k_2step](evaluation/examples/teacher_b_extend6k_2step)
- An official-base / style-teacher same-prompt, same-seed ginkgo comparison set: [evaluation/examples/style_teacher_vs_official_ginkgo](evaluation/examples/style_teacher_vs_official_ginkgo)

## Repository boundary

Tracked: source code under `scripts/`, tests, notebooks, documentation, evaluation prompts/tables, the presentation outline/template source, and the nine curated images above.

Excluded: image datasets, latent archives, prompt embeddings, trajectory caches, checkpoints, LoRA adapters, model weights, generated outputs, and generated PowerPoint/PDF files. Do not bypass this policy with `git add -f`.

## Local assets required for training

Obtain the following from the team's shared storage after cloning. They are intentionally not distributed through GitHub because of size and data-distribution constraints.

| Local path | Purpose |
| --- | --- |
| `data/archives/ink.zip` | 260 image-caption pairs |
| `data/archives/clean_latents_512.zip` | Precomputed 512px clean VAE latents |
| `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` | Local-training prompt embeddings |
| `data/features/distill_t5_plant627_len300_fp16_v1.pt` | Distillation prompt-bank embeddings |
| `data/features/distill_eval_t5_prompts30_len300_fp16_v1.pt` | Cached held-out evaluation embeddings |

See [data/README.md](data/README.md) for asset and validation contracts.

## Setup and entry points

### Option 1: Conda Environment (Recommended)

```powershell
conda env create -f environment.yml
conda activate pixart311
```

### Option 2: Pip Installation

Use Python 3.11. Install a CUDA-enabled PyTorch build appropriate to your machine, then install project dependencies:

```powershell
pip install -r requirements.txt
```

Validate local data assets:

```powershell
py -3.11 scripts/training/train_local_latent_lora.py --validate-assets-only
```

The distillation method, CLI entry points, cache formats, and evaluation gates are documented in [DISTILLATION.md](DISTILLATION.md). Main pipeline scripts are:

- `scripts/distillation/cache_teacher_trajectories.py`
- `scripts/distillation/train_phased_distill_lora.py`
- `scripts/distillation/generate_evaluation_set.py`
- `scripts/distillation/evaluate_distilled.py`

## Layout

```text
data/           local asset contract only (assets are ignored)
evaluation/     tracked metrics, prompts, and curated examples
notebooks/      preprocessing and smoke-test notebooks
scripts/        data preparation, training, distillation, and evaluation code
tests/          contract and regression tests
presentation/   final outline and slide-template source
```

Each run records prompt-cache fingerprints, teacher/adapter hashes, checkpoint, seed, and scheduler configuration. A clone therefore needs the listed external assets to reproduce training, but not to inspect the complete implementation and reported results.

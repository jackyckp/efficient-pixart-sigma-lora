# Efficient PixArt-Sigma LoRA

This project adapts PixArt-Sigma 512 to Chinese ink-wash plant imagery, then distils a 20-step style teacher into 4-step and 2-step joint LoRA students. The repository contains the implementation, experiment contracts, selected adapters, the canonical image-caption archive, evaluation summaries, and curated visual results. Large latent and embedding caches remain local or in shared storage.

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

Tracked: source code under `scripts/`, tests, notebooks, documentation, evaluation prompts/tables, selected LoRA adapters under `models/`, `data/ink.zip`, presentation assets, and the final report source/PDF.

Excluded: latent archives, prompt embeddings, trajectory caches, intermediate checkpoints, generated training/evaluation outputs, and presentation/LaTeX build products. Do not bypass this policy with `git add -f`.

## Getting Started from a Clean GitHub Clone

Due to GitHub size limits, precomputed latents, prompt embeddings, trajectory caches, and intermediate checkpoints are excluded by `.gitignore`. You can choose from **three reproduction paths**:

1. **Track A: Inference & Evaluation (No training required)**
   - Download adapter weights (`py -3.11.2 scripts/inference/download_adapters.py --model teacher_b_primary_2step`).
   - Run 2-step fast inference (`scripts/distillation/generate_distilled.py`). Base model weights auto-download from Hugging Face.
2. **Track B: Training & Distillation with Shared Asset Bundle (Standard Reproduction)**
   - Use the tracked `data/ink.zip`, then obtain `clean_latents_512.zip` and `t5_embeddings_*.pt` from shared storage.
   - Validate assets with `py -3.11.2 scripts/training/train_local_latent_lora.py --validate-assets-only`.
3. **Track C: 100% From-Scratch Cold Start (Zero External Caches Required)**
   - Use the canonical `data/ink.zip` directly to precompute SDXL VAE clean latents (`scripts/data/precompute_clean_latents.py`) and T5 prompt embeddings (`scripts/data/precompute_t5_embeddings.py`), then train from scratch.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for full step-by-step commands and [data/README.md](data/README.md) for data schemas.

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
py -3.11.2 scripts/training/train_local_latent_lora.py --validate-assets-only
```

The distillation method, CLI entry points, cache formats, and evaluation gates are documented in [INSTRUCTIONS.md](INSTRUCTIONS.md) and [ARCHITECTURE_DIAGRAMS.md](ARCHITECTURE_DIAGRAMS.md). Main pipeline scripts are:

- `scripts/distillation/cache_teacher_trajectories.py`
- `scripts/distillation/train_phased_distill_lora.py`
- `scripts/distillation/generate_evaluation_set.py`
- `scripts/distillation/evaluate_distilled.py`

## Repository Structure & Folder Layout

```text
efficient-pixart-sigma-lora/
├── data/                                 # Local asset contracts & datasets
│   ├── README.md                         # ✅ Tracked: Data contracts & schema specifications
│   ├── ink.zip                           # ✅ Tracked: 260 canonical raw image-caption pairs
│   ├── archives/                         # 📦 Generated / Download: Precomputed clean latents
│   │   └── clean_latents_512.zip         #    └─ SDXL VAE clean latents [260, 4, 64, 64]
│   └── features/                         # 📦 Generated / Download: Precomputed prompt caches
│       ├── t5_embeddings_n260_*.pt       #    ├─ Stage 1 Style Teacher prompt embeddings [260, 300, 4096]
│       ├── distill_t5_plant627_*.pt      #    ├─ Stage 3 Distillation prompt cache (via build_distill_prompt_cache.py)
│       └── validation_summary.json       #    └─ Dataset integrity & fingerprint validation record
│
├── models/                               # Pretrained LoRA adapter checkpoints & configs
│   ├── teacher_b_primary_2step/          # ✅ Tracked: Primary 2-step student (adapter_model.safetensors, adapter_config.json)
│   ├── teacher_b_primary_4step/          # ✅ Tracked: Primary 4-step student (adapter_model.safetensors, adapter_config.json)
│   └── best_ink_wash_lora_plant209_step4000/ # ✅ Tracked: 20-step Style Teacher LoRA adapter
│
├── notebooks/                            # Interactive Jupyter & Google Colab notebooks
│   ├── preprocessing/                    # 🛠️ Data preparation & VAE encoding
│   │   └── pixart_clean_latents_colab.ipynb # └─ Colab T4 GPU latent encoding notebook
│   └── evaluation/                       # 📊 Benchmark & report analysis notebooks
│       ├── eval_30prompts_cmmd.ipynb     #    ├─ 30-prompt CMMD & CLIPScore evaluation
│       ├── pixart_data_prep_teacher_eval.ipynb # ├─ Data prep & teacher inspection
│       ├── pixart_matrix_analysis.ipynb  #    ├─ Multi-parameter sweep analysis
│       └── training_10k_report.ipynb     #    └─ 10k style teacher training report
│
├── scripts/                              # Executable CLI tools & training pipelines
│   ├── data/                             # 📥 Latent & T5 precomputation, VLM captioning
│   ├── training/                         # 🏋️ Style teacher training & parameter sweeps
│   ├── distillation/                     # ⚡ Phased distillation, trajectory caching & quality gates
│   ├── evaluation/                       # 📈 CLIPScore, CMMD, and checkpoint grid generators
│   └── inference/                        # 🚀 Single-prompt & batch inference, adapter downloader
│
├── evaluation/                           # Benchmark prompts, evaluation records & curated examples
│   ├── distillation_prompts_v1.json      # ✅ Tracked: 30 held-out evaluation prompts across 4 domains
│   ├── all_4step_2step_training_evaluation_results.csv # ✅ Tracked: Complete multi-step evaluation logs
│   ├── distillation_extended_results_v2.json # ✅ Tracked: Primary benchmark results & metrics
│   └── examples/                         # ✅ Tracked: Curated 512x512 comparison grids
│
├── docs/                                 # Technical documentation, proposals & visual figures
│   ├── style_teacher_evaluation_report.md# Comprehensive style teacher evaluation report
│   ├── Project-Proposal.md               # Original research project proposal
│   └── images/                           # Evaluation plots and visual comparison samples
│
├── outputs/                              # ⚙️ Generated during execution (excluded by .gitignore)
│   ├── style_teacher/                    # ⚙️ Generated: Style teacher checkpoints & training logs
│   ├── distillation/                     # ⚙️ Generated: Trajectory cache shards, student checkpoints & eval metrics
│   └── benchmark_30prompts/              # ⚙️ Generated: 30-prompt benchmark images & CMMD evaluation reports
│
└── tests/                                # Automated test suite (17 test suites, 72 pytest cases)
```

Final presentation source/assets live in `presentation/`. The five-page paper,
its three figures, and the compiled submission PDF live in `report/`; temporary
PowerPoint and LaTeX build files are ignored.

Each run records prompt-cache fingerprints, teacher/adapter hashes, checkpoint, seed, and scheduler configuration. A clone therefore needs the listed external assets to reproduce training, but not to inspect the complete implementation and reported results.

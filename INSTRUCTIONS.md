# End-to-End Instructions: PixArt-Sigma Ink-Wash LoRA Adaptation & Distillation

This guide provides complete, step-by-step instructions to train, distill, evaluate, and deploy the 20-step to 4-step to 2-step **PixArt-Sigma Ink-Wash LoRA** models from scratch.

> [!NOTE]
> All CLI commands default to Python 3.11.2 via `py -3.11.2`. If using Conda, activate the `pixart311` environment first (`conda activate pixart311`).

---

## 📑 Table of Contents

- [0. Prerequisites & Environment Setup](#0-prerequisites--environment-setup)
- 🎨 **[Phase 1: Style Teacher Adaptation & Data Pipeline (Stages 0–2)](#phase-1-style-teacher-adaptation--data-pipeline)**
  - [1. Stage 0: Data Acquisition & Preprocessing (Optional)](#1-stage-0-data-acquisition--preprocessing-optional)
  - [2. Stage 1: Style Teacher Training (20-Step LoRA)](#2-stage-1-style-teacher-training-20-step-lora)
  - [3. Stage 2: Teacher Provenance Validation & Manifest Export](#3-stage-2-teacher-provenance-validation--manifest-export)
  - [4. Stage 3: Distillation Prompt Banking & T5-XXL Caching](#4-stage-3-distillation-prompt-banking--t5-xxl-caching)
  - [5. Stage 4: 20-Step Teacher Trajectory Sharded Caching](#5-stage-4-20-step-teacher-trajectory-sharded-caching)
- ⚡ **[Phase 2: Fast Student LoRA Distillation & Quality Gating (Stages 5–8)](#phase-2-fast-student-lora-distillation--quality-gating)**
  - [6. Stage 5: Distillation Smoke Test & Verification](#6-stage-5-distillation-smoke-test--verification)
  - [7. Stage 6: Student 4-Step Distillation Training](#7-stage-6-student-4-step-distillation-training)
  - [8. Stage 7: 4-Step Quality Evaluation Gate](#8-stage-7-4-step-quality-evaluation-gate)
  - [9. Stage 8: Student 2-Step Distillation Training](#9-stage-8-student-2-step-distillation-training)
  - [10. Stage 9: Final 2-Step Quality Gate & Multi-Model Benchmarking](#10-stage-9-final-2-step-quality-gate--multi-model-benchmarking)
- 🚀 **[Phase 3: Automated Orchestration & Inference Deployment (Stages 10–11)](#phase-3-automated-orchestration--inference-deployment)**
  - [11. Stage 10: Single-Command Automated Orchestration Pipelines](#11-stage-10-single-command-automated-orchestration-pipelines)
  - [12. Stage 11: Fast Single-Image Inference Deployment & Acceptance Records](#12-stage-11-fast-single-image-inference-deployment--acceptance-records)

---

## 0. Prerequisites & Environment Setup

### 0.1 Environment Setup & Dependencies
Install dependencies using Python 3.11.2:

```powershell
py -3.11.2 -m pip install -r requirements.txt
```

Alternatively, if using Conda:
```powershell
conda env create -f environment.yml
conda activate pixart311
```

### 0.2 Local Assets Verification
Verify required raw and feature assets located in [`data/`](data/README.md):
- `data/archives/ink.zip` (260 image-caption pairs)
- `data/archives/clean_latents_512.zip` (Precomputed 512px SDXL VAE clean latents: `image_latents_n260_res512_b9d3c2d1d404.pt`)
- `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` (T5-XXL text prompt embeddings)

Run asset contract validation:
```powershell
py -3.11.2 scripts/training/train_local_latent_lora.py --validate-assets-only
```

---

## Phase 1: Style Teacher Adaptation & Data Pipeline

### 1. Stage 0: Data Acquisition & Preprocessing (Optional)

If starting completely from scratch without existing archives:

#### 1.1 Web Scraping Raw Images
Scrapes ink-wash paintings into category folders under `data/ink/`:
```powershell
py -3.11.2 scripts/data/download_tappu.py
```

#### 1.2 Automated VLM Captioning
Generate descriptive captions with domain trigger words using Florence-2 or JoyCaption:
```powershell
py -3.11.2 scripts/data/auto_caption.py `
  --dir data/ink/plant `
  --model florence-2 `
  --trigger "traditional Chinese ink wash painting style, shuimo hua"
```

---

### 2. Stage 1: Style Teacher Training (20-Step LoRA)

Train the domain-specific 20-step Style Teacher adapter on the 209-image plant subset using the frozen PixArt-Sigma 512 base model.

```powershell
py -3.11.2 scripts/training/train_local_latent_lora.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --plant-only `
  --rank 16 `
  --lora-alpha 16 `
  --max-train-steps 10000 `
  --learning-rate 1e-5 `
  --train-batch-size 1 `
  --gradient-accumulation-steps 1 `
  --mixed-precision fp16 `
  --checkpointing-steps 1000 `
  --seed 42 `
  --output-dir outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05
```

---

### 3. Stage 2: Teacher Provenance Validation & Manifest Export

Validates base model hashes, 574 adapter tensors, 13,765,376 parameters, and exports the teacher manifest:

```powershell
py -3.11.2 scripts/distillation/validate_style_teacher.py `
  --teacher-adapter outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05 `
  --teacher-id plant_n209_r16_step10200 `
  --teacher-guidance-scale 1.0 `
  --output outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json
```

---

### 4. Stage 3: Distillation Prompt Banking & T5-XXL Caching

Generates 627 training prompts (3 variants per plant image: `original`, `subject`, `styled`) and 30 held-out evaluation prompts, caching FP16 T5-XXL embeddings:

```powershell
py -3.11.2 scripts/distillation/build_distill_prompt_cache.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --source-prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --prompt-bank data/distillation/plant_prompt_bank_v1.jsonl `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-cache data/features/distill_t5_plant627_len300_fp16_v1.pt
```

> [!TIP]
> Use `--text-only` on `build_distill_prompt_cache.py` to inspect the generated JSONL prompt bank without loading the heavy T5-XXL text encoder.

---

### 5. Stage 4: 20-Step Teacher Trajectory Sharded Caching

Rolls out the teacher model across all 627 prompts and 2 deterministic seed replicas, saving 21 latent states ($x_0, x_1, \dots, x_{20}$) per trajectory into SHA256-verified safetensors shards:

```powershell
py -3.11.2 scripts/distillation/cache_teacher_trajectories.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --output-dir outputs/distillation/plant_n209_r16_step10200/trajectory_cache_v1 `
  --replicas-per-prompt 2 `
  --shard-size 64
```

---

## Phase 2: Fast Student LoRA Distillation & Quality Gating

### 6. Stage 5: Distillation Smoke Test & Verification

Validates both teachers, caches 4 prompts $\times$ 2 seeds per teacher, and trains 20 optimizer updates for each 4-step student to verify setup integrity:

```powershell
py -3.11.2 scripts/distillation/run_two_teacher_distillation.py `
  --smoke `
  --stop-after 4step `
  --output-root outputs/distillation_smoke
```

Generate a single test image from the smoke-trained checkpoint:
```powershell
py -3.11.2 scripts/distillation/generate_distilled.py `
  --prompt-id plant/220::original `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --adapter outputs/distillation_smoke/plant_n209_r16_step10200/student_4step/lora_adapter `
  --num-inference-steps 4 `
  --guidance-scale 1.0 `
  --seed 42 `
  --output outputs/distillation_smoke/fresh_reload_4step.png
```

---

### 7. Stage 6: Student 4-Step Distillation Training

Distills 20 teacher steps into 4 student jumps: $[(0\to 5), (5\to 10), (10\to 15), (15\to 20)]$ using Pseudo-Huber jump loss (`huber_c=0.001`) and a 20% clean-latent anchor loss:

```powershell
py -3.11.2 scripts/distillation/train_phased_distill_lora.py `
  --trajectory-cache outputs/distillation/plant_n209_r16_step10200/trajectory_cache_v1 `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --latent-bundle data/archives/clean_latents_512.zip `
  --init-adapter outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05 `
  --target-steps 4 `
  --max-train-steps 2000 `
  --learning-rate 5e-6 `
  --checkpoint-every-steps 500 `
  --anchor-probability 0.2 `
  --huber-c 0.001 `
  --output-dir outputs/distillation/plant_n209_r16_step10200/student_4step
```

---

### 8. Stage 7: 4-Step Quality Evaluation Gate

Generate the 120-image evaluation suite (30 held-out prompts $\times$ 4 seeds) and compute quality retention metrics:

#### 8.1 Generate Image Sets (Teacher & 4-Step Student)
```powershell
py -3.11.2 scripts/distillation/generate_evaluation_set.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --student-adapter outputs/distillation/plant_n209_r16_step10200/student_4step/best_adapter `
  --student-steps 4 `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images
```

#### 8.2 Compute Quality Metrics
```powershell
py -3.11.2 scripts/distillation/evaluate_distilled.py `
  --teacher-images outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images/teacher `
  --student-images outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images/student `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_4step/metrics
```

> [!IMPORTANT]
> **4-Step Quality Gate Acceptance Rules**:
> - **CLIPScore Retention**: Mean Student CLIPScore $\ge 90\%$ of Teacher CLIPScore.
> - **CMMD Distance**: Student CMMD $\le 1.5\times$ Teacher CMMD (unbiased squared MMD between normalized CLIP image embeddings with Gaussian RBF kernel).
> - **Inference Speedup**: Median denoising latency speedup $\ge 5\times$.

---

### 9. Stage 8: Student 2-Step Distillation Training

Initialize the 2-step student directly from the best 4-step adapter. In this stage, two jump intervals are learned: $[(0\to 10), (10\to 20)]$, utilizing **50% On-Policy Rollout Matching** on the second jump to eliminate compounding errors.

```powershell
py -3.11.2 scripts/distillation/train_phased_distill_lora.py `
  --trajectory-cache outputs/distillation/plant_n209_r16_step10200/trajectory_cache_v1 `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --latent-bundle data/archives/clean_latents_512.zip `
  --init-adapter outputs/distillation/plant_n209_r16_step10200/student_4step/best_adapter `
  --target-steps 2 `
  --max-train-steps 7000 `
  --learning-rate 2e-6 `
  --checkpoint-every-steps 1000 `
  --anchor-probability 0.2 `
  --on-policy-probability 0.5 `
  --huber-c 0.001 `
  --output-dir outputs/distillation/plant_n209_r16_step10200/student_2step
```

---

### 10. Stage 9: Final 2-Step Quality Gate & Multi-Model Benchmarking

#### 10.1 Generate Evaluation Set for 2-Step Student
```powershell
py -3.11.2 scripts/distillation/generate_evaluation_set.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --student-adapter outputs/distillation/plant_n209_r16_step10200/student_2step/best_adapter `
  --student-steps 2 `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images
```

#### 10.2 Evaluate Final 2-Step Quality & Latency
```powershell
py -3.11.2 scripts/distillation/evaluate_distilled.py `
  --teacher-images outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images/teacher `
  --student-images outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images/student `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_2step/metrics
```

#### 10.3 30-Prompt Benchmark & CMMD Distribution Evaluation
```powershell
py -3.11.2 scripts/inference/eval_30prompts_cmmd.py `
  --guidance-scale 1.0 `
  --seed 42 `
  --output-root outputs/benchmark_30prompts
```

---

## Phase 3: Automated Orchestration & Inference Deployment

### 11. Stage 10: Single-Command Automated Orchestration Pipelines

#### 11.1 Two-Teacher End-to-End Orchestration (Resume-Safe)
Runs validation, caching, 4-step training, quality gating, and 2-step training for both Teacher A and Teacher B:
```powershell
py -3.11.2 scripts/distillation/run_two_teacher_distillation.py `
  --output-root outputs/distillation
```

#### 11.2 Primary Benchmark Experiment (Teacher B 6k $\to$ 2-Step 7k)
Reproduces the primary headline result reported in the repository:
```powershell
py -3.11.2 scripts/distillation/run_teacher_b_extended_6k_then_2step.py
```

> [!NOTE]
> For exploratory work, passing `--skip-quality-gate` allows the runner to proceed directly to 2-step training without waiting for a passing 4-step report.

---

### 12. Stage 11: Fast Single-Image Inference Deployment & Acceptance Records

Generate a 512x512 ink-wash painting in ~0.24 seconds using the 2-step distilled LoRA:

```powershell
py -3.11.2 scripts/distillation/generate_distilled.py `
  --prompt "Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua" `
  --adapter outputs/distillation/plant_n209_r16_step10200/student_2step/lora_adapter `
  --num-inference-steps 2 `
  --guidance-scale 1.0 `
  --seed 42 `
  --output outputs/inference_results/misty_mountains_2step.png
```

#### Acceptance Record Verification
The generator writes an adjacent `.json` metadata sidecar with the image. The acceptance criteria strictly assert:
- `transformer_forward_calls: 2`
- `classifier_free_guidance_branch: false`
- `guidance_scale: 1.0`
- Finite latent values and $512 \times 512$ image output.

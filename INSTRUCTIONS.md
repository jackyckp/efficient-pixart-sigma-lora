# End-to-End Training & Distillation Instructions: PixArt-Sigma Ink-Wash LoRA

This guide provides complete, step-by-step instructions to train, distill, evaluate, and deploy the 20-step to 4-step to 2-step **PixArt-Sigma Ink-Wash LoRA** models from scratch.

---

## 0. Prerequisites & Environment Setup

### 0.1 Environment Creation & Activation
Create the conda environment using the provided [`environment.yml`](environment.yml):

```powershell
conda env create -f environment.yml
conda activate pixart311
```

Alternatively, if updating an existing environment with `pip`:
```powershell
conda activate pixart311
pip install -r requirements.txt
```

### 0.3 Local Assets Verification
Verify required raw and feature assets located in `data/`:
- `data/archives/ink.zip` (260 image-caption pairs)
- `data/archives/clean_latents_512.zip` (Precomputed 512px SDXL VAE clean latents: `image_latents_n260_res512_b9d3c2d1d404.pt`)
- `data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt` (T5-XXL text prompt embeddings)

Run asset contract validation:
```powershell
conda activate pixart311 ; python scripts/training/train_local_latent_lora.py --validate-assets-only
```

---

## 1. Stage 0: Data Acquisition & Preprocessing (Optional / From Scratch)

If starting completely from scratch without existing archives:

### 1.1 Web Scraping Raw Images
Scrapes ink-wash paintings into `data/ink/plant/`, `data/ink/animal/`, etc.:
```powershell
conda activate pixart311 ; python scripts/data/download_tappu.py
```

### 1.2 Automated VLM Captioning
Run Florence-2 or JoyCaption with domain trigger phrases:
```powershell
conda activate pixart311 ; python scripts/data/auto_caption.py --dir data/ink/plant --model florence-2 --trigger "traditional Chinese ink wash painting style, shuimo hua"
```

---

## 2. Stage 1: Style Teacher Training (20-Step LoRA)

Train the domain-specific 20-step Style Teacher adapter on the 209-image plant subset using the frozen PixArt-Sigma 512 base model.

### 2.1 Single Teacher Training
Run rank-16 style teacher training for 10,000 steps:
```powershell
conda activate pixart311 ; python scripts/training/train_local_latent_lora.py `
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

### 2.2 Validate Trained Teacher Provenance & Contract
```powershell
conda activate pixart311 ; python scripts/distillation/validate_style_teacher.py `
  --teacher-adapter outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05 `
  --teacher-id plant_n209_r16_step10200 `
  --teacher-guidance-scale 1.0 `
  --output outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json
```

---

## 3. Stage 2: Distillation Prompt Banking & Teacher Trajectory Caching

### 3.1 Build Distillation Prompt Bank & T5-XXL Cache
Generates 627 training prompts (3 variants per plant image: `original`, `subject`, `styled`) and 30 held-out evaluation prompts:
```powershell
conda activate pixart311 ; python scripts/distillation/build_distill_prompt_cache.py `
  --latent-bundle data/archives/clean_latents_512.zip `
  --source-prompt-cache data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt `
  --prompt-bank data/distillation/plant_prompt_bank_v1.jsonl `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-cache data/features/distill_t5_plant627_len300_fp16_v1.pt
```

### 3.2 Cache 20-Step Teacher Trajectories
Rolls out the teacher model across all prompts and 2 deterministic seed replicas, saving 21 latent states ($x_0, x_1, \dots, x_{20}$) per trajectory into safetensors shards:
```powershell
conda activate pixart311 ; python scripts/distillation/cache_teacher_trajectories.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --output-dir outputs/distillation/plant_n209_r16_step10200/trajectory_cache_v1 `
  --replicas-per-prompt 2 `
  --shard-size 64
```

---

## 4. Stage 3: Student 4-Step Distillation Training

Distills 20 teacher steps into 4 student jumps: $[(0\to 5), (5\to 10), (10\to 15), (15\to 20)]$ using Pseudo-Huber jump loss and a 20% clean-latent anchor loss.

```powershell
conda activate pixart311 ; python scripts/distillation/train_phased_distill_lora.py `
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

## 5. Stage 4: 4-Step Student Quality Evaluation Gate

Generate the 120-image evaluation suite (30 prompts $\times$ 4 seeds) and compute quality retention metrics:

### 5.1 Generate Image Sets (Teacher & 4-Step Student)
```powershell
conda activate pixart311 ; python scripts/distillation/generate_evaluation_set.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --student-adapter outputs/distillation/plant_n209_r16_step10200/student_4step/best_adapter `
  --student-steps 4 `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images
```

### 5.2 Compute Metrics & Validate Quality Gate
```powershell
conda activate pixart311 ; python scripts/distillation/evaluate_distilled.py `
  --teacher-images outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images/teacher `
  --student-images outputs/distillation/plant_n209_r16_step10200/evaluation_4step/images/student `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_4step/metrics
```

**Quality Gate Rules**:
- Student CLIPScore $\ge 90\%$ of Teacher CLIPScore.
- Student CMMD $\le 1.5\times$ Teacher CMMD.
- Median Denoising Latency Speedup $\ge 5\times$.

---

## 6. Stage 5: Student 2-Step Distillation Training

Initialize the 2-step student directly from the best 4-step adapter. In this stage, two jump intervals are learned: $[(0\to 10), (10\to 20)]$, utilizing **50% On-Policy Rollout Matching** on the second jump to eliminate compounding errors.

```powershell
conda activate pixart311 ; python scripts/distillation/train_phased_distill_lora.py `
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

## 7. Stage 6: Final 2-Step Quality Gate & Multi-Model Benchmarking

### 7.1 Generate Evaluation Set for 2-Step Student
```powershell
conda activate pixart311 ; python scripts/distillation/generate_evaluation_set.py `
  --teacher-manifest outputs/distillation/plant_n209_r16_step10200/teacher_manifest.json `
  --student-adapter outputs/distillation/plant_n209_r16_step10200/student_2step/best_adapter `
  --student-steps 2 `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images
```

### 7.2 Evaluate Final 2-Step Quality & Latency
```powershell
conda activate pixart311 ; python scripts/distillation/evaluate_distilled.py `
  --teacher-images outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images/teacher `
  --student-images outputs/distillation/plant_n209_r16_step10200/evaluation_2step/images/student `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir outputs/distillation/plant_n209_r16_step10200/evaluation_2step/metrics
```

### 7.3 30-Prompt Benchmark & CMMD Distribution Evaluation
```powershell
conda activate pixart311 ; python scripts/inference/eval_30prompts_cmmd.py `
  --guidance-scale 1.0 `
  --seed 42 `
  --output-root outputs/benchmark_30prompts
```

---

## 8. Single-Command Automated Pipelines

### 8.1 Two-Teacher End-to-End Orchestration (Resume-Safe)
Runs validation, caching, 4-step training, quality gating, and 2-step training for both Teacher A and Teacher B:
```powershell
conda activate pixart311 ; python scripts/distillation/run_two_teacher_distillation.py `
  --output-root outputs/distillation
```

### 8.2 Primary Benchmark Experiment (Teacher B 6k $\to$ 2-Step 7k)
Reproduces the primary headline result from the repository:
```powershell
conda activate pixart311 ; python scripts/distillation/run_teacher_b_extended_6k_then_2step.py
```

---

## 9. Fast Single-Image Inference

Generate a 512x512 ink-wash painting in ~0.24 seconds using the 2-step distilled LoRA:

```powershell
conda activate pixart311 ; python scripts/distillation/generate_distilled.py `
  --prompt "Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua" `
  --adapter outputs/distillation/plant_n209_r16_step10200/student_2step/lora_adapter `
  --num-inference-steps 2 `
  --guidance-scale 1.0 `
  --seed 42 `
  --output outputs/inference_results/misty_mountains_2step.png
```

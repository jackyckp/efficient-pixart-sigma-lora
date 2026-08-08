# PixArt 20-step to 4-step to 2-step LoRA distillation

This workflow trains two independent rank-16 joint LoRA students. Each student
starts from one 20-step ink-wash style adapter and learns deterministic
trajectory jumps while retaining a standard diffusion anchor loss. The final
2-step adapter runs with `guidance_scale=1.0`, does not execute an unconditional
CFG branch, and performs exactly two PixArt transformer calls.

## Inputs and outputs

Required local inputs:

```text
data/archives/clean_latents_512.zip
data/features/t5_embeddings_n260_len300_fp16_b9d3c2d1d404.pt
outputs/style_teacher/plant_n209_steps10200/r16_lr1e-05
outputs/style_teacher/best_ink_wash_lora_plant209_step4000
```

Generated feature caches remain Git-ignored. Model, cache, checkpoint and
evaluation outputs are written under `outputs/distillation/`.

The two source adapters have the same PixArt-Sigma 512 base, rank/alpha 16,
official target modules, 574 FP32 tensors and 13,765,376 adapter parameters.
The teammate adapter is load-compatible, but its original training manifest is
absent. The validator records this as incomplete provenance and never treats the
README benchmark numbers as independently verified.

## One-time prompt preparation

The prompt builder creates 627 training prompts (three variants for each of the
209 plant captions), 30 held-out evaluation prompts, and the FP16 T5 cache. The
existing original-caption embeddings are reused; T5 is only used for new text.

```powershell
py -3.11 scripts/distillation/build_distill_prompt_cache.py `
  --prompt-bank data/distillation/plant_prompt_bank_v1.jsonl `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-cache data/features/distill_t5_plant627_len300_fp16_v1.pt
```

Use `--text-only` to inspect the exact prompt bank without loading T5.

## Required smoke test

This validates both teachers, caches 4 prompts x 2 seeds per teacher, and trains
20 optimizer updates for each 4-step student. It is deliberately stopped before
long training.

```powershell
py -3.11 scripts/distillation/run_two_teacher_distillation.py `
  --smoke `
  --stop-after 4step `
  --output-root outputs/distillation_smoke
```

The smoke run requires the one-time distillation prompt cache above. A successful
run produces two limited trajectory caches, two adapters, finite loss histories,
checkpoint metadata and resume state.

Generate a smoke image from a cached training prompt:

```powershell
py -3.11 scripts/distillation/generate_distilled.py `
  --prompt-id plant/220::original `
  --prompt-cache data/features/distill_t5_plant627_len300_fp16_v1.pt `
  --adapter outputs/distillation_smoke/plant_n209_r16_step10200/student_4step/lora_adapter `
  --num-inference-steps 4 `
  --guidance-scale 1.0 `
  --seed 42 `
  --output outputs/distillation_smoke/fresh_reload_4step.png
```

## Full 4-step stage

The following command is resume-safe. It validates both teachers, builds or
resumes the two 1,254-trajectory caches, and trains both 2,000-update 4-step
students. Cache shards are SHA256-checked before reuse.

```powershell
py -3.11 scripts/distillation/run_two_teacher_distillation.py `
  --stop-after 4step `
  --output-root outputs/distillation
```

Defaults for this stage are rank/alpha 16, learning rate `5e-6`, checkpoint every
500 optimizer updates, batch size 1, FP16 base weights, FP32 LoRA weights, 80%
pseudo-Huber trajectory updates and 20% clean-latent diffusion anchor updates.

## 4-step quality gate

Run these two commands once for each teacher ID. Replace `$TEACHER` with either
`plant_n209_r16_step10200` or `teammate_plant209_step4000`.

```powershell
$TEACHER = "plant_n209_r16_step10200"

py -3.11 scripts/distillation/generate_evaluation_set.py `
  --teacher-manifest "outputs/distillation/$TEACHER/teacher_manifest.json" `
  --student-adapter "outputs/distillation/$TEACHER/student_4step/best_adapter" `
  --student-steps 4 `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir "outputs/distillation/$TEACHER/evaluation_4step/images"

py -3.11 scripts/distillation/evaluate_distilled.py `
  --teacher-images "outputs/distillation/$TEACHER/evaluation_4step/images/teacher" `
  --student-images "outputs/distillation/$TEACHER/evaluation_4step/images/student" `
  --evaluation-prompts evaluation/distillation_prompts_v1.json `
  --output-dir "outputs/distillation/$TEACHER/evaluation_4step/metrics"
```

The evaluator checks mean CLIP >= 90% of the corresponding teacher, student
CMMD <= 1.5x teacher CMMD, median denoising speedup >= 5x, finite metrics and
exact forward-call metadata. CMMD is the unbiased squared MMD between normalized
CLIP image embeddings with a Gaussian RBF kernel. The report records the CLIP
checkpoint and bandwidth so results are not mixed across configurations.

## Full 2-step stage

After both 4-step evaluation summaries are `PASS`, rerun the orchestrator. It
will reuse completed assets and initialize each 2-step student from its best
4-step adapter.

```powershell
py -3.11 scripts/distillation/run_two_teacher_distillation.py `
  --output-root outputs/distillation
```

The 2-step defaults are learning rate `2e-6`, at most 10,000 optimizer updates
and checkpoint every 1,000 updates. Half of second-phase trajectory examples use
a detached first jump from the current student to reduce rollout distribution
shift. Use `--resume-from <checkpoint>` on the trainer for an interrupted run.

Evaluate the final model by repeating the quality-gate commands with
`student_4step` changed to `student_2step`, `--student-steps 2`, and output paths
changed to `evaluation_2step`.

For exploratory work only, `--skip-quality-gate` allows the orchestrator to
start 2-step training without a passing 4-step report. Such a run must not be
reported as quality-preserving.

## Manual 2-step inference

```powershell
py -3.11 scripts/distillation/generate_distilled.py `
  --prompt "Misty mountain peaks enveloped in soft clouds, ancient pine tree on a cliff, traditional Chinese ink wash painting style, shuimo hua" `
  --adapter outputs/distillation/plant_n209_r16_step10200/student_2step/lora_adapter `
  --num-inference-steps 2 `
  --guidance-scale 1.0 `
  --seed 42 `
  --output outputs/distillation/manual_2step.png
```

The adjacent JSON file is the acceptance record. It must report
`transformer_forward_calls: 2`, `classifier_free_guidance_branch: false`,
`guidance_scale: 1.0`, a finite latent and a 512 x 512 image.

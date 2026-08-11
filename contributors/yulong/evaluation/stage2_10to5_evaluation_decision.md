# Stage 2 Distillation Decision: 10 Steps to 5 Steps

## Decision

Select **Student checkpoint step 8,000** as the canonical 5-step model.

The checkpoint and `final_adapter` are byte-identical (SHA-256: `13413F5322CE4EB340209853CE6F7C5E0BFBAA6B2D756797FCFDD1E78FF85C7E`), so the deployment adapter is:

`D:\AI\pixart_local_generation\models\distilled_students\student_10to5_teacher_step12000_trajectories_r4_train8000_g1_seed43\final_adapter`

## Evaluation design

- 8 prompts: flowers, cat, bird, young woman, elderly man, Toronto skyline, teapot, sneaker
- 3 seeds per prompt: 42, 123, 2026
- 24 matched cases per model
- 6 models: original 20-step reference, canonical 10-step Teacher, and 5-step Student checkpoints at 2k/4k/6k/8k
- 144 generated images in total
- Reference: 20 inference steps, guidance scale 2
- Teacher: 10 inference steps, guidance scale 1
- Student: 5 inference steps, guidance scale 1

## Quantitative result

| Model | Similarity to 10-step Teacher | Similarity to 20-step Reference | Full-prompt CLIP | Content-only CLIP | Median time | Speedup vs 10-step | Speedup vs 20-step |
|---|---:|---:|---:|---:|---:|---:|---:|
| 20-step Reference | 0.9395 | 1.0000 | 0.3505 | 0.3375 | 2.71 s | 0.32x | 1.00x |
| 10-step Teacher | 1.0000 | 0.9395 | 0.3528 | 0.3324 | 0.86 s | 1.00x | 3.14x |
| 5-step Student 2k | 0.9338 | 0.8917 | 0.3532 | 0.3230 | 0.51 s | 1.68x | 5.28x |
| 5-step Student 4k | 0.9313 | 0.8891 | 0.3589 | 0.3226 | 0.51 s | 1.70x | 5.34x |
| 5-step Student 6k | 0.9479 | 0.9006 | 0.3518 | 0.3212 | 0.50 s | 1.74x | 5.45x |
| **5-step Student 8k** | **0.9557** | **0.9083** | **0.3518** | **0.3225** | **0.51 s** | **1.70x** | **5.35x** |

## Interpretation

- The 8k checkpoint has the strongest average similarity to both the current 10-step Teacher and the earlier 20-step reference among all 5-step candidates.
- Full-prompt CLIP alignment is nearly unchanged from the 10-step Teacher (0.3518 versus 0.3528).
- Content-only CLIP is lower than the 10-step Teacher (0.3225 versus 0.3324), so prompt fidelity remains the main guardrail for future testing.
- Targeted visual review of the difficult young-woman, elderly-man, sneaker and cat cases plus representative flower, bird, city and teapot cases shows that 8k is generally the closest 5-step candidate to the 10-step Teacher. Portraits and the sneaker remain the most sensitive categories.
- The 5-step model reduces median generation time from 0.86 s to 0.51 s versus the 10-step Teacher and from 2.71 s to 0.51 s versus the original 20-step reference on this machine.

## Recommendation

Freeze the 8k/final adapter as the canonical 5-step Student. Before attempting 5-to-3 distillation, run one human preference review across all 24 matched cases and record structure failures separately from style differences. Do not use CLIP alone as a quality verdict.

## Evidence

- `metrics_summary.csv`: aggregate quality and speed metrics
- `metrics_detailed.csv`: all 144 image-level records
- `contact_sheets/`: 24 matched six-model comparison sheets
- `chart_dual_reference_similarity.png`: similarity to both distillation references
- `chart_prompt_alignment.png`: text-image alignment guardrails
- `chart_generation_time.png`: measured latency comparison

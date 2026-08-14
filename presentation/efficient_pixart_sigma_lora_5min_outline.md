# Efficient PixArt-Sigma LoRA — 5-Minute / 5-Slide Outline

## Communication goal

By the end, the audience should understand how the project first teaches PixArt-Sigma a transferable Chinese ink-wash style with LoRA, then compresses the learned 20-step generation process into quality-preserving 4-step and 2-step joint LoRA students.

**Total duration:** 5:00
**Slide count:** 5, including cover
**Primary models:** `teacher_b_extend6k_then2step` 4-step and 2-step students
**Narrative:** motivation → system → style generalization → inference acceleration

---

## Slide 1 — Cover

**Speaker time:** 0:00–0:15

### Audience-facing title

**Efficient Ink-Wash Generation with PixArt-Sigma LoRA**

### Subtitle

Domain adaptation and 20→4→2-step diffusion distillation

### Footer

Team member names · Course · Date

### Visual direction

- Minimal cover using one representative output from the primary 2-step model.
- No agenda or technical details on the cover.

### Speaker notes

Our project has two connected goals: adapt PixArt-Sigma to Chinese ink-wash painting with LoRA, then reduce generation from 20 denoising steps to only four or two model calls.

### [Sources]

- `Project-Proposal.md`
- `outputs/distillation_experiments/teacher_b_extend6k_then2step/evaluation_2step/metrics/`

---

## Slide 2 — Intro, Background, Motivation, Data Source, and PixArt

**Speaker time:** 0:15–1:10

### Takeaway title

**Domain adaptation is useful only if generation is both recognizable and affordable**

### Audience-facing content

#### Motivation

- Pretrained text-to-image models can follow prompts but do not consistently reproduce a specialized ink-wash distribution.
- Full-model fine-tuning is expensive; LoRA adapts the model through a small set of trainable low-rank weights.
- Standard 20-step generation preserves quality but requires 20 Transformer calls, motivating few-step distillation.

#### Data source

- **260 validated image–caption pairs** from the Tappu ink-wash corpus.
- Categories: **209 plant, 30 animal, 11 web, 10 other**.
- Primary Style Teacher and distillation experiments use the **209 plant subset**.
- Images are normalized to **512×512** with cached clean VAE latents and T5 prompt embeddings.

#### Why PixArt-Sigma

- A text-conditioned **Diffusion Transformer (DiT)** operating in latent space.
- T5-XXL provides text conditioning; a VAE maps between images and compact latents.
- Transformer attention layers provide a natural target for parameter-efficient LoRA adaptation.

### Visual direction

Use one simple composition:

- Left: a small ink-wash dataset montage or category count.
- Right: `Text → T5 → PixArt-Sigma DiT → VAE → Image`.
- Bottom callout: `Goal: retain ink-wash quality with 4 or 2 Transformer calls`.

### Speaker notes

The complete corpus contains 260 image-caption pairs, but the primary model focuses on the 209 plant paintings because this is the most coherent domain subset. We precompute latents and prompt embeddings once, which makes training reproducible and avoids repeatedly loading the VAE or 19 GB T5 encoder. PixArt-Sigma is a strong base because its denoiser is a Transformer and can be adapted efficiently through attention-layer LoRA weights.

### [Sources]

- `Project-Proposal.md`
- `README.md`
- `data/README.md`
- `data/archives/ink.zip`
- `data/archives/clean_latents_512.zip`

---

## Slide 3 — System Overview, Training Pipeline, and Model Architecture

**Speaker time:** 1:10–2:15

### Takeaway title

**One joint LoRA learns the visual domain first and acceleration second**

### Audience-facing pipeline

1. **Prepare reusable features**
   Clean latents `[N, 4, 64, 64]` + T5 embeddings `[N, 300, 4096]`

2. **Train the 20-step Style Teacher**
   Rank-16 LoRA learns the plant-focused ink-wash distribution

3. **Cache teacher trajectories**
   627 prompt variants × 2 seeds = **1,254 fixed 20-step trajectories**

4. **Train the 4-step joint LoRA**
   Learn four deterministic jumps while retaining the teacher’s style weights

5. **Train the 2-step joint LoRA**
   Initialize from the best extended 4-step adapter and learn two larger jumps

### Model architecture callouts

- **Frozen:** PixArt-Sigma base Transformer, T5 encoder, and VAE.
- **Trainable:** rank-16 attention LoRA, approximately **13.8M parameters**.
- **Loss:** 80% trajectory pseudo-Huber loss + 20% clean-latent diffusion anchor.
- **PixArt contract:** train only the first four epsilon channels; ignore learned-sigma channels in the distillation loss.
- **Deployment:** one adapter, guidance scale 1.0, no unconditional CFG branch.

### Visual direction

Use one horizontal system diagram:

`Images + captions → cached features → 20-step teacher → trajectory cache → 4-step LoRA → 2-step LoRA`

Place the frozen PixArt-Sigma block behind the three LoRA stages to show that only adapter weights change. Avoid a dense layer-by-layer network diagram.

### Speaker notes

The Style Teacher is first trained with the standard diffusion objective. We then run that teacher once to cache complete denoising trajectories. Student training reads the cache without loading the teacher, which fits the available GPU memory. Because each student continues training the existing style adapter, the deployed model needs only one joint LoRA rather than separate style and acceleration adapters.

### [Sources]

- `INSTRUCTIONS.md`
- `ARCHITECTURE_DIAGRAMS.md`
- `scripts/training/train_local_latent_lora.py`
- `scripts/distillation/cache_teacher_trajectories.py`
- `scripts/distillation/train_phased_distill_lora.py`
- `outputs/distillation/teammate_plant209_step4000/trajectory_cache_v1/cache_manifest.json`

---

## Slide 4 — Evaluation & Discussion I: LoRA Style Generalization

**Speaker time:** 2:15–3:30

### Takeaway title

**LoRA transfers ink-wash style to rewritten and unseen prompts, but more training is not always better**

### Evaluation setup

- Same prompt, seed, 20 inference steps, and guidance scale 1.5 for each checkpoint comparison.
- Compare the official PixArt-Sigma base with LoRA ranks 4, 8, 16, and 32 across training checkpoints.
- Two generalization cases:
  - **Palm adaptation:** rewritten from a training subject, but not an exact training caption.
  - **Misty mountain:** new subject composition outside the plant-caption training prompts.

### Audience-facing results

| Prompt case | Official base CLIP | Best LoRA CLIP | Best configuration | Relative gain |
|---|---:|---:|---|---:|
| Palm adaptation | 0.3399 | **0.3561** | Rank 4, step 1,000 | **+4.7%** |
| Misty mountain | 0.3615 | **0.3720** | Rank 8, step 4,000 | **+2.9%** |

### Discussion

- LoRA can transfer the ink-wash domain beyond exact training captions while maintaining prompt alignment.
- The best rank and checkpoint change with the prompt; no single rank dominates every example.
- CLIP often peaks at an early or middle checkpoint and then declines, showing that longer training can reduce generalization.
- This motivates checkpoint selection and held-out evaluation before choosing a Style Teacher.

### Visual direction

- Use two compact prompt examples, each with `Base` and `Best LoRA` images plus the CLIP values.
- Keep the small two-row table as quantitative evidence.
- Do not show the full 40-checkpoint grid in the presentation.

### Speaker notes

These experiments test whether LoRA memorizes training captions or transfers the style. Both the rewritten palm prompt and the unseen misty mountain prompt obtain higher best-case CLIP than the official base model. However, the trajectories are not monotonic: later checkpoints frequently decline. The result is not that one rank is universally best, but that LoRA provides transferable style capacity and requires validation-based checkpoint selection.

### [Sources]

- `outputs/evaluation/style_teacher_clip_scores_palm_adaptation/clip_scores.csv`
- `outputs/evaluation/style_teacher_clip_scores_palm_adaptation/clip_score_metadata.json`
- `outputs/evaluation/style_teacher_clip_scores_misty_mountain/clip_scores.csv`
- `outputs/evaluation/style_teacher_clip_scores_misty_mountain/clip_score_metadata.json`

---

## Slide 5 — Evaluation & Discussion II: Reducing Inference Steps

**Speaker time:** 3:30–5:00

### Takeaway title

**Four steps preserve teacher quality; two steps provide the strongest speed–quality trade-off**

### Evaluation setup

- Primary experiment: `teacher_b_extend6k_then2step`.
- **30 unseen prompts × 4 fixed seeds = 120 images per model**.
- Metrics: CLIP text alignment, CMMD to real ink-wash data, median denoising latency, and exact Transformer call count.

### Primary quantitative results

| Model | Calls | CLIP ↑ | Teacher CLIP retained | CMMD ↓ | Median time | Speedup | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| 20-step Style Teacher B | 20 | 0.3651 | 100% | 0.000949* | 2.871 s | 1.00× | Reference |
| **Primary 4-step** | **4** | **0.3674** | **100.65%** | **0.001087** | **0.480 s** | **5.99×** | **PASS** |
| **Primary 2-step** | **2** | **0.3535** | **96.84%** | **0.001394** | **0.244 s** | **11.78×** | **PASS** |

`* Teacher row reports teacher-to-real CMMD; student rows report student-to-real CMMD.`

### Discussion and conclusion

- **4-step deployment:** choose when visual quality is the priority; it slightly exceeds teacher CLIP while using one fifth as many calls.
- **2-step deployment:** choose when latency is the priority; it retains 96.84% of teacher CLIP with nearly 12× speedup.
- Both primary students pass all predefined quality and execution gates.
- Limitation: the 2-step CMMD is only **2.00% below** its acceptance limit; CLIP-derived metrics do not replace human judgment.
- Next step: expand unseen prompts and run a blinded human preference test before attempting 1-step or timestep-gated dual LoRA.

### Closing line

**A single joint LoRA can carry both ink-wash style and few-step acceleration, reducing PixArt-Sigma from 20 calls to 4 or 2 without adding a second deployment adapter.**

### Visual direction

- Make the results table the main visual.
- Highlight the 4-step and 2-step rows; keep the teacher row neutral.
- End with the decision rule: `Quality priority → 4-step | Speed priority → 2-step`.
- Do not add a separate “Thank you” slide.

### Speaker notes

The primary 4-step model is the strongest measured quality point, while the primary 2-step model is the strongest deployment point. The 2-step result is promising but still close to the CMMD boundary, so the responsible conclusion is a measured quality–speed trade-off rather than a claim of perfect teacher equivalence.

### [Sources]

- `outputs/distillation_experiments/teacher_b_extend6k_then2step/evaluation_4step/metrics/evaluation_summary.json`
- `outputs/distillation_experiments/teacher_b_extend6k_then2step/evaluation_2step/metrics/evaluation_summary.json`
- `outputs/distillation_experiments/teacher_b_extend6k_then2step/evaluation_comparison.json`
- `evaluation/distillation_extended_results_v2.json`

---

## Timing check

| Slide | Content | Time |
|---|---|---:|
| 1 | Cover | 0:15 |
| 2 | Intro + background + motivation + data + PixArt | 0:55 |
| 3 | System overview + training pipeline + architecture | 1:05 |
| 4 | Evaluation & discussion: LoRA style generalization | 1:15 |
| 5 | Evaluation & discussion: inference-step distillation | 1:30 |
| **Total** |  | **5:00** |

# Curated visual examples

The final repository keeps only nine generated images: three representative grids for each primary distilled student, plus three existing style-teacher/official comparison components. All are 512 x 512 PNGs selected from existing evaluation outputs; no images were newly generated for this repository.

| Folder | Contents | Purpose |
| --- | --- | --- |
| `teacher_b_extend6k_4step/` | Three teacher-vs-4-step comparison grids | Primary Teacher B 6k four-step student on held-out prompts and fixed seeds |
| `teacher_b_extend6k_2step/` | Three teacher-vs-2-step comparison grids | Primary Teacher B 6k two-step student on held-out prompts and fixed seeds |
| `style_teacher_vs_official_ginkgo/` | Official base image plus two rank-16 style-teacher images | Existing same-prompt, same-seed ginkgo comparison |

The full 30-prompt x 4-seed image sets, checkpoints, trajectory caches, and model weights remain excluded from Git. They can be regenerated with the tracked distillation/evaluation scripts after obtaining the local assets described in [data/README.md](../../data/README.md).

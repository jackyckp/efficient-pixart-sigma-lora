# Style teacher vs. official base: ginkgo prompt

These are existing 20-step, guidance-1.0 generations using the same unseen ginkgo prompt and seed `2026`:

- `official_base.png`: official PixArt-Sigma base model, without an adapter.
- `teacher_r16_lr1e-05.png`: rank-16 ink-wash style teacher, learning rate `1e-5`.
- `teacher_r16_lr5e-06.png`: rank-16 ink-wash style teacher, learning rate `5e-6`.

They are separate same-condition images rather than a newly composed grid. The source metadata confirms the prompt was not an exact training-caption match.

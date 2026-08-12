# Style-teacher checkpoint grids

This evaluation compares the four teacher LoRA ranks at identical optimizer
checkpoints. It uses one new prompt and a fixed seed, so differences in each
2x2 grid come from the trained adapter checkpoint, not prompt encoding or
sampling noise.

Run:

\`\`\`powershell
& 'C:\Users\miaoj\AppData\Local\Programs\Python\Python311\python.exe' \`
  scripts\evaluation\generate_style_teacher_checkpoint_grids.py \`
  --model-root outputs\style_teacher\all_n260_steps10000 \`
  --output-dir outputs\evaluation\style_teacher_checkpoint_grids
\`\`\`

Defaults:

- Prompt: a new white-crane / lotus-pond ink-wash prompt. The script checks
  saved all-260 training manifests and rejects exact caption matches.
- Sampling: seed 123, PixArt-Sigma 512, 20 inference steps, CFG 1.5.
- Models: rank 4, 8, 16, and 32 at learning rate 1e-5.
- Groups: checkpoints 1,000 through 10,000, producing ten 2x2 grids.

Generated images are deliberately ignored under
\`outputs/evaluation/style_teacher_checkpoint_grids/\`:

\`\`\`text
step_001000_grid.png ... step_010000_grid.png
individual/step_<checkpoint>_rank_<rank>.png
evaluation_metadata.json
\`\`\`

The metadata records prompt auditing, rank/checkpoint paths, checkpoint loss
metadata, settings, and generation time.


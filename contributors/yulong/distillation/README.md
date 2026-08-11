# Progressive LoRA Distillation

This implementation performs offline trajectory distillation in two stages:

```text
20-step style Teacher (guidance 2.0)
              |
              v
10-step intermediate LoRA (guidance 1.0)
              |
              v
5-step Student LoRA (guidance 1.0)
```

The Teacher and Student are never placed on the 8 GiB GPU at the same time. Teacher reverse-sampling trajectories are cached first; the Student is then trained against those targets.

## Selected configurations

| Stage | Target mode | Trajectories/prompt | Target records | Optimizer updates | LR | Selected checkpoint |
|---|---|---:|---:|---:|---:|---:|
| 20 -> 10 | Teacher reverse trajectory | 4 | 10,400 | 16,000 | `1e-6` | 12,000 |
| 10 -> 5 | Teacher reverse trajectory | 4 | 5,200 | 8,000 | `1e-6` | 8,000 |

Both stages use deterministic DDIM sampling with trailing timestep spacing and `eta=0`. Stage 1 bakes the original guidance-2 Teacher behavior into the targets, so the distilled models run at guidance 1 during inference.

## Files

- `cache_teacher_trajectories.py`: creates offline guided Teacher trajectories.
- `train_student_lora.py`: initializes the Student from its Teacher LoRA and optimizes against cached targets.
- `evaluate_student_checkpoints.py`: performs the matched multi-prompt, multi-seed evaluation.
- `compare_teacher_student.py`: creates direct Teacher/Student comparisons.
- `generate_distilled.py`: generates images with distilled adapters.
- `configs/`: exact Stage 1 and Stage 2 run definitions.
- `status/`: original completion records from the local run.

Target caches are deliberately omitted because they are large derived tensors.


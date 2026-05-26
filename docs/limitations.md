# Limitations

This repository is intentionally scoped as a small simulation showcase.

## Current Constraints

- No real robot integration.
- No Isaac Sim implementation yet.
- No prompt embedding inside the student models.
- The student baselines are lightweight CNN + MLP policies.
- Student evaluation is small-scale and should not be treated as a full benchmark sweep.
- The chunk student result is from the available recorded evaluation, not a large multi-seed study.

## What This Means

The reported numbers are useful for illustrating the gap between offline imitation and closed-loop execution, but they are not the final word on policy quality.

## Not Included

- real-world deployment
- uploaded checkpoints
- uploaded datasets
- large-scale hyperparameter sweeps


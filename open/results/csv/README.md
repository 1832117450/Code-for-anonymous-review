# D4RL Hopper CSV Results

This directory contains CSV-only results for the public `hopper-medium-v2` D4RL task.
Checkpoints, logs, generated figures, and private datasets are intentionally omitted.

- `baselines/`: dense and sparse baseline curves on `hopper-medium-v2`.
- `methods/`: TD3+NC dense/sparse curves and Ours sparse curves on `hopper-medium-v2`.

The files are per-seed CSV curves. Summary statistics can be recomputed from
the last 10 evaluation points of these curves.

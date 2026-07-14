# D4RL Hopper-Medium Public Release

This folder contains the public D4RL release of the offline RL experiments.
Only the public `hopper-medium-v2` task is included.

## Contents

- `src/train_ours.py`: main training script for Ours and the TD3+NC lower-only ablation.
- `baselines/`: D4RL baseline implementations: BCQ, TD3+BC, IQL, and CQL.
- `configs/`: reference Hopper configuration files for the reported settings.
- `results/csv/`: CSV-only `hopper-medium-v2` baseline and method outputs. Model checkpoints and logs are not included.
- `plotting/plot_hopper_results.py`: minimal plotting script for the public CSV files.

## Installation

Create a conda environment using `environment.yml`, or install equivalent versions of PyTorch, Gym, MuJoCo, and D4RL.

```bash
conda env create -f environment.yml
conda activate d4rl-release
```

Depending on the local MuJoCo setup, D4RL may require additional system packages and a compatible MuJoCo/Gym version.

Optional Docker usage is provided for users who prefer an isolated Linux
environment:

```bash
docker build -t td3nc-d4rl-hopper .
docker run --gpus all --rm -it -v "$PWD":/workspace td3nc-d4rl-hopper bash
```

The Docker image still relies on D4RL/MuJoCo compatibility. If MuJoCo rendering
is not needed, the training and CSV plotting commands can be run headlessly.

## Data

This release uses only the public D4RL `hopper-medium-v2` dataset. D4RL is
available at <https://github.com/Farama-Foundation/D4RL>. The dataset is
downloaded automatically by D4RL when the training scripts call
`gym.make("hopper-medium-v2")` and `env.get_dataset()`. No private industrial
data, private preprocessing scripts, or private samples are included.

The sparse delayed-reward setting is generated from the public D4RL rewards by
accumulating rewards over a fixed delay window and emitting the accumulated
reward at the delay boundary or at an episode boundary.

## Config Files

The files under `configs/` document the reported hyperparameters. The training
script is CLI-driven, so use these files as a reference when passing command-line
arguments.

- `hopper_medium_sparse_ours.yaml`: Ours, sparse delayed reward.
- `hopper_medium_sparse_td3_nc.yaml`: TD3+NC lower-only ablation, sparse delayed reward.
- `hopper_medium_dense_td3_nc.yaml`: TD3+NC lower-only ablation, dense reward.

## Single-Seed Examples

All commands should be run from this repository root. Baseline scripts use
Weights & Biases by default; set `WANDB_MODE=disabled` if you only want local
execution.

TD3+NC with dense reward:

```bash
python src/train_ours.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --dense-reward \
  --alpha 0.5 \
  --beta 2.0 \
  --k 4 \
  --lower-only \
  --max-timesteps 1000000 \
  --save-dir results/example_td3_nc_dense
```

Ours with sparse delayed reward:

```bash
python src/train_ours.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay-step 50 \
  --alpha 0.5 \
  --beta 2.0 \
  --k 4 \
  --max-timesteps 1000000 \
  --save-dir results/example_hopper_sparse
```

TD3+NC lower-only ablation:

```bash
python src/train_ours.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay-step 50 \
  --alpha 0.5 \
  --beta 2.0 \
  --k 4 \
  --lower-only \
  --max-timesteps 1000000 \
  --save-dir results/example_td3_nc_sparse
```

## Baseline Examples

The baseline scripts also support the same sparse delayed reward protocol via
`--delay_step 50`. The following commands write CSV curves with the same
`step,norm_score` format as the provided files under `results/csv/`.

BCQ:

```bash
python baselines/bcq.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay_step 50 \
  --save_dir results/example_bcq_sparse \
  --max_timesteps 1000000
```

TD3+BC:

```bash
WANDB_MODE=disabled python baselines/td3_bc.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay_step 50 \
  --save_dir results/example_td3_bc_sparse \
  --max_timesteps 1000000
```

IQL:

```bash
WANDB_MODE=disabled python baselines/iql.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay_step 50 \
  --save_dir results/example_iql_sparse \
  --max_timesteps 1000000
```

CQL:

```bash
WANDB_MODE=disabled python baselines/cql.py \
  --env hopper-medium-v2 \
  --seed 0 \
  --delay_step 50 \
  --save_dir results/example_cql_sparse \
  --max_timesteps 1000000
```

For dense-reward baselines, omit `--delay_step 50`.

## Plotting Public Results

```bash
python plotting/plot_hopper_results.py \
  --csv-root results/csv \
  --out-dir figures
```

The script writes learning curves, last-10 summary bars, and a CSV summary to
the output directory. Generated figures are ignored by git.

Learning curves show the cross-seed mean with a half-standard-deviation band,
matching the paper plotting style. Final bar scores are computed by first taking
the last 10 evaluation points within each seed and then reporting the mean and
standard deviation across seeds.

## Minimal Reproduction Loop

1. Create the environment with `environment.yml`.
2. Plot the provided CSV files with `plotting/plot_hopper_results.py`.
3. Re-run one seed using the examples above and compare the generated CSV with
   the files under `results/csv/`.
4. For full reproduction, run seeds `0 1 2 3 4` for each method and compute the
   last-10 evaluation mean.

## Notes

- Sparse reward is constructed by accumulating environment rewards over a fixed delay window and assigning the accumulated reward at the delay boundary or episode end.
- Return-to-go labels are computed only from the offline D4RL dataset rewards and are used for support-conditioned neighbor ranking.
- Final reported scores should be computed from final checkpoints or the last 10 evaluation points, rather than selecting the best checkpoint by test performance.
- The release intentionally omits model checkpoints, local logs, non-Hopper datasets, and auxiliary parameter-sweep outputs.
- This release contains only public D4RL data-processing code and public
  Hopper CSV curves. Private industrial data, private evaluation scripts, and
  paper-only figure assets are not included.

## Naming

- Paper method name: `Ours`.
- Lower-only ablation/baseline name: `TD3+NC`.
- File and directory names use lowercase snake case: `ours` and `td3_nc`.

# SONIC replay-validated LeRobot dataset cleaning

This workflow ensures that the SONIC latent supervision in a LeRobot dataset
comes only from trajectories that RoboCasa can verify as task-successful.

The two synchronized recordings have different roles:

- `datasets/sonic_raw/.../episodes/ep_*/ep_demo.hdf5` contains recorded MuJoCo
  states and is the authority for RoboCasa task success.
- `datasets/robocasa_<Task>_g1_3cam` contains parquet data, three camera videos,
  and the 64-dimensional `action.motion_token` latent used for training.

The exporter may contain discarded, aborted, failed, or incomplete attempts.
Episode indices therefore cannot be matched by order alone.

## Safety policy

The alignment and cleaning process is fail-closed. A LeRobot episode is kept
only when all of the following are true:

1. The raw attempt contains `ep_demo.hdf5`.
2. RoboCasa state replay reports the raw episode as `passed`.
3. Its raw completion timestamp maps uniquely to a LeRobot parquet save time.
4. The source LeRobot episode was not already marked discarded.
5. The parquet and every declared camera video exist.

Missing HDF5, failed replay, missing replay results, unmatched timestamps, or
ambiguous timestamps never enter the cleaned dataset. The source raw and
LeRobot datasets are not modified.

## Step 1: replay raw HDF5 episodes

Replace `<Task>` and the report date as needed:

```bash
cd /home/user/robocasa

MUJOCO_GL=egl \
NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache \
/home/user/miniforge3/envs/robocasa/bin/python \
  robocasa/scripts/evaluate_sonic_raw.py \
  --dataset /home/user/robocasa/datasets/sonic_raw \
  --task <Task> \
  --success-mode any \
  --seed 0 \
  --isolate-episodes \
  --episode-timeout-seconds 300 \
  --report /home/user/robocasa/datasets/<Task>_replay_report_YYYYMMDD.json
```

`--success-mode any` accepts an episode if any recorded checkpoint satisfies
the task's `_check_success()` condition. Use `--success-mode final` when the
terminal recorded state must still satisfy success. Keep the selected mode
consistent for a dataset version.

The replay command is read-only unless `--delete-failed` is explicitly added.
For alignment, keep it read-only and let the cleaned LeRobot output perform the
selection non-destructively.

## Step 2: preview the raw-to-LeRobot alignment

Run without `--apply` first:

```bash
cd /home/user/robocasa

NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache \
/home/user/miniforge3/envs/robocasa/bin/python \
  robocasa/scripts/align_sonic_success_lerobot.py \
  --raw-root /home/user/robocasa/datasets/sonic_raw \
  --lerobot-dataset /home/user/robocasa/datasets/robocasa_<Task>_g1_3cam \
  --replay-report /home/user/robocasa/datasets/<Task>_replay_report_YYYYMMDD.json
```

This prints the source episode indices selected and excluded, then writes:

```text
/home/user/robocasa/datasets/robocasa_<Task>_g1_3cam_raw_replay_alignment.json
```

The default timestamp policy matches each LeRobot parquet save to the nearest
unused preceding raw completion within 15 seconds. A second candidate within
1 second of the best lag is treated as ambiguous and aborts the operation.
These thresholds can be changed with `--max-lag-seconds` and
`--ambiguity-margin-seconds`, but changes should be documented in the resulting
dataset version.

## Step 3: build and verify the cleaned dataset

After reviewing the preview, repeat the command with `--apply`:

```bash
cd /home/user/robocasa

NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache \
/home/user/miniforge3/envs/robocasa/bin/python \
  robocasa/scripts/align_sonic_success_lerobot.py \
  --raw-root /home/user/robocasa/datasets/sonic_raw \
  --lerobot-dataset /home/user/robocasa/datasets/robocasa_<Task>_g1_3cam \
  --replay-report /home/user/robocasa/datasets/<Task>_replay_report_YYYYMMDD.json \
  --apply
```

The output is built in a hidden sibling partial directory and promoted only
after verification:

```text
/home/user/robocasa/datasets/robocasa_<Task>_g1_3cam_replay_cleaned
```

The existing SONIC cleaner performs the following operations:

- excludes every LeRobot episode not backed by a successful raw replay;
- preserves synchronization between parquet and all camera videos;
- removes invalid stale PICO pose frames using stream-mode-aware rules;
- reindexes episodes, frames, and global indices contiguously;
- rebuilds `episodes.jsonl`, episode statistics, and dataset totals;
- validates parquet lengths, video frame counts, resolution, FPS, and metadata;
- removes `discarded_episode_indices` from the verified output.

If the output already exists, inspect it before using
`--replace-existing`. Replacement rotates the old output to a timestamped
backup rather than deleting it.

## Output provenance

Each cleaned dataset contains:

```text
meta/raw_replay_alignment.json
meta/cleanup_report.json
```

The alignment report records the replay report path and SHA-256, task, matching
thresholds, raw episode, source LeRobot index, completion lag, replay status,
selection reason, and kept/excluded indices. The cleanup report records source
to output episode reindexing and frame-cleaning statistics.

Training should use only the `_replay_cleaned` directory, not the original
exporter directory.

## Verified CoolBakedCake result (2026-08-19)

Source:

```text
/home/user/robocasa/datasets/robocasa_CoolBakedCake_g1_3cam
```

Replay report:

```text
/home/user/robocasa/datasets/CoolBakedCake_replay_report_20260819.json
```

Verified output:

```text
/home/user/robocasa/datasets/robocasa_CoolBakedCake_g1_3cam_replay_cleaned
```

Result:

- 12 replay-passed episodes
- 63,230 frames
- 36 videos (three cameras per episode)
- source indices: `5, 6, 7, 8, 9, 11, 14, 15, 17, 18, 19, 24`
- `action.motion_token` shape: 64
- non-finite latent values: 0
- all-zero latent frames: 0

The result above used `--success-mode any`. Two raw episodes reached success at
an intermediate checkpoint but not at the final checkpoint; use a separate
`final`-validated dataset version if terminal-state success is required.

## Validation tests

Run the alignment and existing dataset-cleaner tests with:

```bash
cd /home/user/robocasa
NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache \
  /home/user/miniforge3/envs/robocasa/bin/python -m pytest -q \
  tests/test_align_sonic_success_lerobot.py \
  tests/test_clean_sonic_dataset.py
```

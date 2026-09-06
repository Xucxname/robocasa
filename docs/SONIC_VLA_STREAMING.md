# RoboCasa SONIC VLA Streaming

This path lets the RoboCasa SONIC collector produce the same ZMQ streams that
`gear_sonic/scripts/run_data_exporter.py` expects during real-world VLA
collection. RoboCasa remains the owner of the MuJoCo clock and the DDS-driven
SONIC controller remains unchanged.

## What it publishes

Running `robocasa/scripts/collect_sonic_demos.py --vla-stream` adds four VLA
integration pieces:

- A camera stream on port `5555` using SONIC's `ImageMessageSchema`.
- A subscriber for VR/PICO `manager_state` toggles on port `5556`.
- A keyboard publisher on port `5580` so local collector hotkeys can keep
  `run_data_exporter.py` in sync.
- A state-style metadata publisher on port `5581` that forwards the exact
  instruction sampled for the current RoboCasa episode.

The instruction is captured once after each successful reset. A `ready` state
is repeated on port `5581`; pressing record changes it to a versioned `start`
attempt carrying the same episode ID and instruction. The exporter waits for
that start metadata before entering `RECORDING`, then locks the text before the
episode's first frame.
For object-dependent tasks such as `LoadDishwasher`, a generated instruction
like `Pick up the cup and bowl ...` therefore becomes the LeRobot task label
automatically. `--task-prompt` remains a fallback for an older collector, a
missing/blank task instruction, or non-RoboCasa collection.

Default camera settings match the real VLA path: `robot0_head_camera` is
published as `ego_view` at `640x480`, `30 Hz`, with the MuJoCo image vertically
flipped. Collision geoms are hidden and visual geoms are rendered.

The image renderer runs in a separate process. The 200 Hz collection loop only
copies the latest MuJoCo state into a bounded queue, so image streaming is not in
the control-loop critical path.

During startup, the controller startup band remains enabled, but its orientation
reference is the reset/spawn pelvis pose so fixture-facing spawn yaw is not
pulled back toward the world frame.
Recording remains blocked until that real command arrives, so saved demos still
contain real SONIC gains and q-star targets.

## VR workflow terminals

Run these in separate terminals for RoboCasa VR collection. This mirrors the
manual VR workflow in `dc_ft_docs.md`, except RoboCasa replaces
`gear_sonic/scripts/run_sim_loop.py` as the MuJoCo simulator and camera
publisher. Do not start both sim loops at the same time.

### 1. Start RoboCasa collection with VLA streaming

Use an interactive display for real collection. Do not set `MUJOCO_GL=egl` for
normal VR collection because the collector viewer is interactive.

```bash
cd /home/amaddukuri/Projects/robocasa-dev-sonic-vla
/home/amaddukuri/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python \
  robocasa/scripts/collect_sonic_demos.py \
  --environment Kitchen \
  --layout 1 \
  --robot SonicG1 \
  --out /tmp/sonic_robocasa_demos \
  --vla-stream
```

This process owns the RoboCasa MuJoCo sim, publishes DDS lowstate/odometry to
the SONIC controller, receives DDS lowcmd actions back from the controller, and
publishes simulated ego camera frames for the exporter on port `5555`.

### 2. Start the SONIC controller

This follows the C++ deployment terminal from `dc_ft_docs.md`: source the deploy
environment, use `zmq_manager` input, and pass `sim` as the final mode argument.

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl/gear_sonic_deploy
source scripts/setup_env.sh
./deploy.sh --input-type zmq_manager sim
# Wait until you see "Init done"
```

The `zmq_manager` input subscribes to the PICO manager stream on port `5556`;
the trailing `sim` selects the sim/DDS backend instead of the real robot backend.

### 3. Start the PICO manager

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl
source .venv_teleop/bin/activate
python gear_sonic/scripts/pico_manager_thread_server.py --manager
```

The PICO manager publishes VR pose/planner inputs and `manager_state` recording
toggles on port `5556`. Keep this process running for both teleop control and
episode start/save/discard events.

Each streamed `pose` message remains a complete sliding window of five PICO
frames. SONIC merges that window by `frame_index`; latest-only transport means
the newest complete window, not a one-frame pose. The exporter uses independent
latest-only subscribers for `pose`, `planner`, and `manager_state`, so those
topics cannot replace one another in a shared conflated queue.

`manager_state` repeats a manager session ID, monotonic event counters, and the
last event timestamps. This allows the collector and exporter to discard stale
continuous state after a long save/reset while still observing a one-frame
start/save/discard button edge. These fields affect recording lifecycle only;
they are not part of SONIC's policy observation.

### 4. Start the VLA exporter

```bash
cd /home/amaddukuri/Projects/GR00T-WholeBodyControl
source .venv_data_collection/bin/activate
python gear_sonic/scripts/run_data_exporter.py \
  --task-prompt "<fallback task prompt>" \
  --dataset-name <dataset_name> \
  --root-output-dir /tmp/sonic_vla_exports \
  --no-text-to-speech
```

The exporter consumes the RoboCasa `ego_view` camera stream on port `5555`, the
VR/PICO stream on port `5556`, SONIC state/config streams from the controller
on port `5557`, keyboard sync on port `5580`, and RoboCasa episode instructions
on port `5581`.

Before recording, confirm that the exporter prints both lines below. The second
line is the task text that will be written to each frame in that episode:

```text
[EpisodeInstruction] ready: <session:sequence> attempt=0 -> <sampled instruction>
[EpisodeInstruction] start matched: <session:sequence> attempt=1 -> <sampled instruction>
[EpisodeInstruction] recording task from RoboCasa <session:sequence>: <sampled instruction>
```

Common overrides:

```bash
--vla-camera-name robot0_head_camera
--vla-camera-key ego_view
--vla-camera-width 640
--vla-camera-height 480
--vla-camera-hz 30
--no-vla-camera-flip
--no-vla-keyboard-sync
--vla-instruction-port 5581
--no-vla-instruction-sync
```

Exporter-side overrides are `--instruction-zmq-host`,
`--instruction-zmq-port`, `--instruction-wait-timeout`, and
`--no-sync-robocasa-instruction`. Use the last option with an older RoboCasa
collector that does not publish port `5581`; the existing `--task-prompt` is
then used immediately.

## Episode controls

Local collector hotkeys still work:

- `c`: start recording in RoboCasa and notify the exporter.
- `k`: save the RoboCasa episode and notify the exporter to stop/save.
- `x`: discard the RoboCasa episode and notify the exporter to abort.
- `b`: toggle the startup elastic band after SONIC is balancing.

VR/PICO `manager_state` toggles are also consumed:

- `toggle_data_collection`: start when idle, save when recording.
- `toggle_data_abort`: discard the current episode.

After `k` or `x`, `[sonic-timing]` lines split the transition into finalize,
environment reset, and source/instruction phases. The collection clock is
resynchronized when the new episode is ready, so the 200 Hz loop does not try
to catch up wall-clock deadlines that expired during a hard reset. Camera
publication remains `640x480` at `30 Hz`.

## Replay and validate collected data

The VLA exporter writes a LeRobot dataset containing one parquet file and three
camera videos per episode. Replay the recorded cameras after collection and
write a machine-readable reliability report with:

```bash
cd /home/user/robocasa
conda run -n robocasa python robocasa/scripts/replay_sonic_dataset.py \
  datasets/robocasa_CloseFridgeDrawer_g1_3cam_cleaned \
  --episodes 0 1 2 \
  --output-dir artifacts/dataset_replay/close_fridge_review
```

Each output MP4 synchronizes the head, left-wrist, and right-wrist views and
overlays the task, episode/frame index, timestamp, and the norms of
`action.wbc` and `observation.state`. `replay_report.json` checks:

- episode metadata length against parquet rows and decoded video frames;
- video FPS and resolution against `meta/info.json`;
- contiguous timestamps and frame/episode indices;
- every declared numeric feature's shape and NaN/Inf count;
- stale zero SMPL poses in pose-driven modes; and
- whether an episode was marked in `discarded_episode_indices`.

The report also records each camera's consecutive duplicate-frame ratio as a
diagnostic. It is not a failure criterion: the default 30 Hz camera publisher
feeding a 50 Hz exporter is expected to repeat some encoded frames.

The default selection is the first five non-discarded episodes. Use `--all` to
check the complete dataset, `--shuffle --seed 0` for a reproducible sample, or
`--check-only` to create only the JSON report. Discarded episodes are excluded
unless `--include-discarded` is explicitly supplied.

This is an offline recorded-camera replay. It proves that exporter videos,
parquet data, and their time axes are structurally aligned; task success still
requires watching the replay. The exporter dataset does not contain the full
MuJoCo state, episode model XML, integration warm-start, or SONIC PD gains, so
it cannot reproduce scene physics from `action.wbc` alone.

### Synchronized Rerun inspection

Use Rerun when joint, end-effector, and SONIC token traces need to be inspected
at the same timestamp as the camera. Open episode 0 directly in the native
viewer with:

```bash
cd /home/user/robocasa
conda run -n robocasa python robocasa/scripts/rerun_sonic_dataset.py \
  datasets/robocasa_CloseFridgeDrawer_g1_3cam_cleaned \
  --episode 0 \
  --spawn
```

The saved/viewed recording has four vertically synchronized rows:

1. `Camera Video | Unitree G1`: `observation.images.ego_view`.
2. `SONIC motion_token`: all 64 dimensions from `action.motion_token` are
   stored and displayed by default. Use `--motion-token-dims 0 1 2` when only a
   smaller subset should be visible initially. Token L2 norm and the number of
   changed dimensions per frame are also stored under
   `/sonic/motion_token/summary` but hidden by default.
3. `Right Arm Joint`: all seven right-arm `action.wbc` targets overlaid with
   their `observation.state` measurements.
4. `ee xyz`: right-wrist xyz from `observation.eef_state[7:10]`, expressed in
   the SONIC exporter's Pinocchio forward-kinematics reference frame.

The default non-GUI mode creates a timestamped `.rrd` and a JSON manifest under
`artifacts/rerun`. An explicit output and a shorter inspection window can be
created with:

```bash
conda run -n robocasa python robocasa/scripts/rerun_sonic_dataset.py \
  datasets/robocasa_CloseFridgeDrawer_g1_3cam_cleaned \
  --episode 0 \
  --start-seconds 2.0 \
  --end-seconds 8.0 \
  --save artifacts/rerun/close_fridge_ep0.rrd

conda run -n robocasa rerun artifacts/rerun/close_fridge_ep0.rrd
```

The manifest processes the complete 64D `motion_token`: each dimension gets
min/max/mean/std, nonzero fraction, and transition-change fraction, and the
episode gets all-zero-row, 1/16 quantization-residual, and joint tracking-error
summaries. All plotted streams share the parquet `timestamp` timeline, so the
viewer plays at the recorded 50 Hz rate; the original frame number is retained
as `/metadata/source_frame_index`. This timestamp is `frame_index / fps` at the
exporter tick. Camera capture timestamps were not retained, so it demonstrates
row-level exporter alignment rather than hardware-level synchronization.

Current G1 exports use H.264, which the Rerun 0.22 viewer may fail to decode
with older host FFmpeg builds. `--camera-mode auto` therefore stores selected
camera frames as portable JPEG images. This makes the `.rrd` larger but keeps
the camera visible. Use `--camera-mode asset-video` for a compact embedded MP4
only when the viewer's video decoder is known to support that codec.

For causal simulator replay, retain the collector's `demo.hdf5` in a persistent
`--out` directory and replay its original 200 Hz actions:

```bash
cd /home/user/robocasa
conda run -n robocasa env NUMBA_DISABLE_JIT=1 MUJOCO_GL=egl \
  python -m robocasa.scripts.dataset_scripts.playback_dataset_hdf5 \
  --dataset /path/to/collector/demo.hdf5 \
  --use-actions \
  --n 1 \
  --video_path /tmp/sonic_action_replay.mp4
```

SONIC action replay intentionally rejects an HDF5 file that lacks valid
dataset-level `sonic_gains` or per-episode `states_integration`. Omitting
`--use-actions` performs state visualization, which does not test whether the
recorded actions reproduce the trajectory.

## Real-time check

The renderer was tested on a hard scene with `DivideBuffetTrays`, layout `25`,
style `11`, at `200 Hz` control. Baseline paced collection and VLA-streamed
collection both held `RTF = 1.0` and achieved `200 Hz`; the VLA process
published frames at the requested camera rate without blocking the sim loop.

## Smoke tests

Focused unit tests:

```bash
cd /home/amaddukuri/Projects/robocasa-dev
MUJOCO_GL=egl /home/amaddukuri/Projects/GR00T-WholeBodyControl/.venv_sim/bin/python \
  -m pytest tests/test_sonic_vla_streaming.py -q
```

Exporter smoke test shape:

1. Start mock VR/PICO `manager_state` messages on port `5556`.
2. Start `gear_sonic/scripts/run_data_exporter.py`.
3. Start `collect_sonic_demos.py --vla-stream`.
4. Toggle start/save and verify the exporter writes
   `observation.images.ego_view` videos.

The latest local smoke test wrote two exporter episodes with `98` total frames
and `observation.images.ego_view` videos.

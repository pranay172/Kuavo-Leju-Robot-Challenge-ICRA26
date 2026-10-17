# Reproducing the results

Everything starts from the public LejuRobotics repositories, pinned to the revisions in
[`upstream/pins.env`](../upstream/pins.env), plus the small patches in
[`upstream/patches/`](../upstream/patches):

| Repository | Branch | Commit |
|---|---|---|
| [kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) | `icra` | `b0a4ec484dc31e650cc4ca761f16c2662aef94ff` |
| &nbsp;&nbsp;submodule `third_party/lerobot` ([huggingface/lerobot](https://github.com/huggingface/lerobot)) | | `d9e74a9d374a8f26582ad326c699740a227b483c` |
| [kuavo-ros-opensource](https://github.com/LejuRobotics/kuavo-ros-opensource) | `opensource/kuavo-data-challenge-icra` | `ded65260c33df70bb6020f7a7d086a5a1529367f` |

The `icra` branch of kuavo_data_challenge has moved on since; check out the commit, not the
branch head.

## Patches

| Patch | Why |
|---|---|
| `kuavo_data_challenge/0001-batched-rosbag-conversion.patch` | Rosbag conversion in batches (`kuavo_data/batch_convert.py`: download 100 bags from Hugging Face, convert, delete) with the converter appending to an existing dataset, plus `dataset.finalize()` so LeRobot v3 closes its parquet files. Adds `kuavo_data/consolidate_dataset.py` (merges the batches into one dataset) and `kuavo_data/check_parquet.py` (reports unreadable parquet files). `batch_convert.py` is pinned to the dataset revision below and stops on a failed download, since a missing bag would shift every later episode index. |
| `kuavo-ros-opensource/0001-optional-full-telemetry-recording.patch` | Adds a `record_full_telemetry` launch argument (default `true`, upstream behaviour) to the three simulator launch files. The harness sets it to `false` to keep full rosbag recording off the machine during scoring. |
| `kuavo-ros-opensource/0002-ekf-cpu-core.patch` | Pins the EKF nodelet to CPU core 1 instead of 18. `taskset -c 18` fails on machines with 18 or fewer cores; the simulator readme suggests this change when the robot falls. Pick any core your machine has. |
| `kuavo-ros-opensource/0003-ocs2-assets-path-for-container.patch` | `ocs2_robotic_assets::getPath()` returns a path from Leju's CI machine; point it at the workspace inside the simulator container. |

One more fix is kept out of the default set:
`upstream/optional-patches/kuavo_data_challenge/compute-image-moments-in-float64.patch`.
The converter's `compute_episode_stats` squares uint8 pixels before promoting them, which
overflows and gives RGB standard deviations 9-2000x too small. The datasets behind the results
were converted without this fix, and training replaces their RGB statistics anyway (step 1b),
so leave it out to reproduce the same data. Apply it (`git -C kuavo_data_challenge apply <patch>`) for any new
conversion whose `stats.json` you want to use directly.

`upstream/setup_upstream.sh` clones both repositories at the pinned commits, initialises the
LeRobot submodule and applies the patches:

```bash
export LEJU_ROOT=$HOME/leju      # any empty directory; everything below lives in it
git clone https://github.com/pranay172/Kuavo-Leju-Robot-Challenge-ICRA26.git "$LEJU_ROOT/Kuavo-Leju-Robot-Challenge-ICRA26"
cd "$LEJU_ROOT/Kuavo-Leju-Robot-Challenge-ICRA26"
upstream/setup_upstream.sh
```

## 1. Data

### 1a. Rosbags to LeRobot format

The demonstrations are Leju's public simulation rosbags on Hugging Face,
[LejuRobotics/kuavo_data_challenge_icra](https://huggingface.co/datasets/LejuRobotics/kuavo_data_challenge_icra)
under `sim/TASK<N>-<name>/task<n>_0001.bag` ... `task<n>_1000.bag`: 1,000 bags per task, 283 GB
(Task 1), 234 GB (Task 2) and 884 GB (Task 3) in total. A batch of 100 bags needs 23-88 GB of
temporary disk. The results used revision `17c135ee0e68508aafeae0c6a84348197ff98133`
(7 January 2026); the `sim/` bags are unchanged at later revisions.

Conversion uses the challenge converter (`kuavo_data/CvtRosbag2Lerobot.py`) with its default
config, `configs/data/KuavoRosbag2Lerobot.yaml`: rq2f85 gripper, both arms, 640x480, depth on,
10 Hz, 10 frames dropped at each end. It needs ROS Noetic and the LeRobot environment, which the
policy image provides (build it first, see step 3); the image contains the patched converter.
Mount two host directories for the downloaded bags and the output (not the whole checkout: the
image's Python environment lives inside `/root/kuavo_data_challenge`):

```bash
mkdir -p "$LEJU_ROOT/conversion/lerobot_data" "$LEJU_ROOT/conversion/temp_raw_bags"
docker run --rm -it \
    -v "$LEJU_ROOT/conversion/lerobot_data:/root/kuavo_data_challenge/lerobot_data" \
    -v "$LEJU_ROOT/conversion/temp_raw_bags:/root/kuavo_data_challenge/temp_raw_bags" \
    kuavo-icra-policy:cu128 bash
# inside the container, in /root/kuavo_data_challenge; repeat for 101-200, ..., 901-1000
python kuavo_data/batch_convert.py --task TASK1-ToySorting --start 1 --end 100
# after all ten batches
python kuavo_data/consolidate_dataset.py --repo-id lerobot/TASK1-ToySorting \
    --root /root/kuavo_data_challenge/lerobot_data/TASK1-ToySorting \
    --output-repo-id lerobot/TASK1-ToySorting_Final --size-mb 10000
python kuavo_data/check_parquet.py lerobot_data/TASK1-ToySorting_consolidated
```

The same for `TASK2-ParcelWeighing` and `TASK3-ConveyorBeltSorting`. Run the batches in
increasing order: bags are converted in filename order, so episode *i* is bag *i+1*, and the
train/validation split refers to episode indices. Bags go to `temp_raw_bags/<task>/` and are
deleted after each successful batch. The files are owned by root on the host. The consolidated
dataset, `$LEJU_ROOT/conversion/lerobot_data/<task>_consolidated`, is the task's source dataset:
1,000 episodes with 145,361 (Task 1), 159,620 (Task 2) and 520,334 (Task 3) frames, AV1 video.

Converting a Task 1 bag took about 19 s on a 16-core laptop, so a task needs several hours of
conversion on top of the download (Task 3 bags are about 3x larger). Three sample bags converted
this way gave episodes identical to the ones used for training: equal state, action and
timestamps, and decoded frames with zero pixel difference.

Known limitation of the converter: it trims the common end of the topics but not their common
start, so an episode whose camera starts before the joint sensors pairs its first frames with
later proprioception (seen in 1 of 3 sample bags: the first 6 frames, up to 0.56 s ahead). The
results were trained on data with this behaviour.

### 1b. RGB statistics repair

Place each task's source dataset at `datasets/task<N>_source` (on the training machine:
`/workspace/recovery/datasets/task<N>_source`), then repair the RGB statistics. The command also
writes the train/validation split and the experiment manifest, and builds the hard-linked
dataset views the trainer reads:

```bash
/workspace/venv/bin/python data_prep/repair_rgb_stats.py --source-root /workspace/recovery/datasets/task3_source \
    --output-dir /workspace/recovery/datasets --dataset-prefix task3 --workers 8 --max-threads 16
```

It needs numpy, pyarrow and PyAV (all in the training venv). The split (seed 1729) and the
frame sampling are deterministic, so `task<N>_split.json`, `task<N>_experiment_manifest.json`
and `task<N>_train_stats_*.json` should match the copies in `data_prep/outputs/`.

## 2. Training

The runs used one RTX 3090 (24 GB) on a RunPod PyTorch image with Python 3.12 and torch
2.8.0+cu128 in the system Python; `setup_venv.sh` checks for that torch version and builds a venv
on top of it from `venv_pip_freeze_20261005.txt`. The launchers expect a fixed `/workspace`
layout, which `training/stage_workspace.sh` creates (it also runs `upstream/setup_upstream.sh`
for kuavo_data_challenge only):

```bash
training/stage_workspace.sh
/workspace/setup/setup_venv.sh          # venv on top of the system torch, from the pip freeze
```

Then do the data step above. Tasks 1 and 2 read frames from a JPEG cache, built once per task:

```bash
/workspace/venv/bin/python /workspace/recovery/tools/build_task1_jpeg_cache.py \
    --dataset /workspace/recovery/datasets/task1_rgb_repaired --cache /workspace/frame_cache/task1_jpeg_q95 \
    --workers 21 --seconds 3600 --deadline-unix $(( $(date +%s) + 3600 ))
```

The builder does not resume: it stops with an error at the deadline, so give it enough time
(30 minutes with 21 workers was enough for Task 2). Its `metadata.json` says `"status": "complete"`
when it has finished. The launchers
default to `/workspace/frame_cache/task{1,2}_jpeg_q95` (override with `FRAME_CACHE`). Start a
run with:

```bash
bash /workspace/setup/launch_task1_act_wsd_40000.sh   # or task2 / task3
```

Each launcher prints its run directory. Snapshots land in `<run>/train/step<NNNNNN>/{raw,ema}`
every 5,000 steps. `touch <run>/train/STOP` stops a run cleanly and writes a final snapshot.
Training throughput was 64-70 samples/s on all three tasks, a little over 5 hours for 40k steps.

| Task | Run | Snapshot used |
|---|---|---|
| 1 | stopped by hand at step 36,093 | `step036093/ema` (written at the stop) |
| 2 | 40k steps | `step035000/ema` |
| 3 | stopped by hand at step 49,266 | `step035000/ema` |

Tasks 1 and 3 were stopped after two snapshots in a row failed to improve the DEV score. Task 1
then ran a learning-rate decay phase: the trainer, started again with
`--resume <run>/train --decay-steps 3609`, lowers the learning rate from its current value to
zero on a cosine over that many further steps. Its final snapshot (step 39,702) scored 75.8 on
the DEV seeds against 88.0 for step 36,093, so it is not used and the phase can be skipped.

GPU kernels and video decoding are not bit-deterministic, so retrained weights will differ
slightly. Pick the snapshot by simulator score (step 3) rather than by validation loss.

## 3. Simulator evaluation

Evaluation runs on a Linux machine with an NVIDIA GPU and a desktop X session (MuJoCo renders
the cameras through the X server). It uses two long-running containers, which
`evaluation/eval_seeds.py` drives with `docker exec`:

- **Simulator:** Leju's published image, with the kuavo-ros-opensource checkout mounted at
  `/root/kuavo_ws`.
- **Policy:** built from `docker/policy/Dockerfile`, with the patched kuavo_data_challenge
  checkout in the image and the exact Python package list of the environment behind the scores
  (`docker/policy/requirements.txt`: torch 2.7.1+cu128, LeRobot from the submodule).

Both mount `$LEJU_ROOT/recovery`, the runtime directory holding the harness, the checkpoints
and the outputs, at `/recovery`.

```bash
# Simulator image (from the kuavo-ros-opensource readme)
wget https://kuavo.lejurobot.com/docker_images/kuavo_opensource_mpc_wbc_img_v0.6.1.tar.gz
docker load -i kuavo_opensource_mpc_wbc_img_v0.6.1.tar.gz

# Policy image (about 19 GB; downloads about 4 GB of Python packages)
docker build -t kuavo-icra-policy:cu128 \
    --build-context kdc="$LEJU_ROOT/kuavo_data_challenge" docker/policy

# Harness, X access, containers
evaluation/install_runtime.sh
xhost +SI:localuser:root
docker/start_containers.sh

# Build the simulator workspace (first time only), then let MuJoCo generate its
# CppAD files once: launch, wait until the robot stands, then Ctrl-C.
docker exec -i kuavo-icra-sim bash -s < docker/build_simulator.sh
docker exec -it kuavo-icra-sim bash -c \
    'export ROBOT_VERSION=45; source devel/setup.bash; roslaunch data_challenge_simulator load_kuavo_mujoco_sim1.launch'
```

After a reboot: `xhost +SI:localuser:root` again (otherwise MuJoCo fails with
`could not initialize GLFW`) and `docker start kuavo-icra-sim kuavo-icra-policy`.

### Scoring a model

Copy the chosen snapshot's `ema` directory under the runtime directory, then score it on the
BENCH seeds:

```bash
cp -r <run>/train/step035000/ema "$LEJU_ROOT/recovery/experiments/task3_step035000_ema"
evaluation/run_final_bench.sh 3 "$LEJU_ROOT/recovery/experiments/task3_step035000_ema"
```

As a check of this setup, the Task 2 final model scored 72.9 (30 episodes) with the policy
image built from `docker/policy/Dockerfile`, against 77.3 with the original environment. A later
no-cache rebuild scored 63.1 (29 episodes), with two of its three blocks slowed by the laptop
display being on (see below). Run back to back on seeds 100-109 with the display off, the
original environment scored 67.5 and the rebuilt image 63.9. Over six runs of seeds 100-109,
the 10-episode means of the same model ranged from 44 to 78, so these differences are within
run-to-run noise. These runs are the `bench_task2_nas5_*` rows of `results/evaluations.csv`.

The script runs seeds 100-129 in three 10-episode blocks at the re-plan interval each final
model uses (Task 1: 10, Task 2: 5, Task 3: 10), and prints each block's mean, its latency check
and the pooled mean. It took about 20 minutes for Task 1 or 2 and about 35 minutes for Task 3.

For other seed sets or intervals, call the harness directly:

```bash
cd "$LEJU_ROOT"
python3 recovery/tools/eval_seeds.py --checkpoint <ckpt> --task 2 --seeds 44-58,64-68 \
    --n-action-steps 5 --group-size 10 --time-budget-minutes 40 --label <name> \
    --output recovery/experiments/eval/<name>
```

Seed sets: DEV 44-68, BENCH 100-129, reserve 130-143. Keep each call under 60 minutes (the
harness refuses longer budgets); split larger seed sets into blocks.

### Demo videos

`evaluation/record_demo.sh` runs a model on a few consecutive seeds and records the three
policy cameras at every step. With `FRONT_VIEW=1` it also saves the simulator's MuJoCo viewer,
which follows the robot from the front, at 10 fps: `docker/viewer_dump.cpp` is preloaded into
the simulator in place of the headless shim and copies the 3D view out of each viewer frame
before it would be presented, so no window appears. The script compiles it in the simulator
container on first use. `evaluation/make_demo_gif.py` then turns one episode into an animated
WebP (or GIF): the front view on the left, matched to the camera frames by time, and the head
camera above the two wrist cameras on the right. The clips in `docs/media/` were made this way,
from the final models:

```bash
FRONT_VIEW=1 LEJU_ROOT=$LEJU_ROOT evaluation/record_demo.sh 2 \
    "$LEJU_ROOT/recovery/experiments/task2_step035000_ema" 100 3
# prints the scores and the videos directory; pick a high-scoring episode
RUN=<run directory printed above, without /videos>
docker exec kuavo-icra-policy /root/kuavo_data_challenge/myenv/bin/python \
    "$PWD/evaluation/make_demo_gif.py" "$RUN/videos/episode_0000_seed100" "$PWD/docs/media/task2.webp" \
    --speed 1 --width 640 --front "$RUN/front_view" --title "Task 2 - ParcelWeighing (1x)"
```

Drop `--front` for camera-only clips. The policy container mounts `$LEJU_ROOT` at the same path,
so this works when the repository is under `$LEJU_ROOT`. Files written from the containers are
owned by root. Three episodes took about 2 minutes for Task 1 or 2 and about 4 minutes for Task 3.

### Reading the scores

- The simulator is very stochastic: one seed can score 0 or 100 on different runs. Compare
  means over 20-30 episodes.
- Run `python3 recovery/tools/latency_report.py <eval dir>` on every run. It marks a run
  `DEGRADED` when the 95th-percentile observation age is above 60 ms, a threshold set on Task 3,
  whose clean runs sit near 40 ms; there, degraded runs (77-101 ms p95, multi-second spikes)
  dropped the score from 94.7 to 33. Task 2 runs normally sit at 45-70 ms p95, so for Task 2
  look instead for p95 well above that or for spikes of several seconds. Other heavy load on the
  machine (another simulator, a build, a video call) is the usual cause. On the laptop used here
  (AMD APU with the display on its integrated GPU, RTX 5050), simply having the display on
  doubled policy inference p95 (about 35 to 80-100 ms) and Task 2 then scored mostly 0-40; run
  long evaluations with the screen off.
- An episode that ends in a controller safety exit (`Safety check failed` in the simulator log,
  usually after the arm command jumps between action chunks) is a policy failure, scored at the
  points it reached. The harness marks it invalid and leaves it out of its own `mean_score`;
  the README, `results/evaluations.csv` and `run_final_bench.sh` count it.
- `results/evaluations.csv` lists every evaluation behind the README, with its seeds, scores,
  safety exits and latency.

The scripts in `evaluation/scripts/` are the exact screening and sweep scripts used during
the project. They reference that machine's run names and directories, so read them as a record
of what was run rather than as entry points.

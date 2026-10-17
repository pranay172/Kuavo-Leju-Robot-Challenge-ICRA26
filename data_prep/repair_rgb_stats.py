#!/usr/bin/env python3
"""Repair a task dataset's RGB moments from its videos, leaving the source dataset untouched.

Dependencies: numpy, pyarrow, PyAV. The repair is resumable per video file.
Example:
  python3 data_prep/repair_rgb_stats.py --dataset-prefix task1 \
    --source-root datasets/task1_source --output-dir datasets
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RGB_KEYS = ("observation.images.head_cam_h", "observation.images.wrist_cam_l",
            "observation.images.wrist_cam_r")
QUANTILES = {"q01": .01, "q10": .10, "q50": .50, "q90": .90, "q99": .99}
REPAIR_VERSION = 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def quantile(hist, q):
    cumulative = np.cumsum(hist)
    target = (int(cumulative[-1]) - 1) * q
    low, high = int(np.floor(target)), int(np.ceil(target))
    lo = int(np.searchsorted(cumulative, low + 1))
    hi = int(np.searchsorted(cumulative, high + 1))
    return (lo + (hi - lo) * (target - low)) / 255


def decode_file(job):
    """Decode sequentially once; accumulate float64 spatial moments at chosen frames."""
    path, key, episodes, samples, fps, threads, cache_dir = job
    signature = {"repair_version": REPAIR_VERSION, "path": str(path.resolve()),
                 "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns,
                 "key": key, "episodes": episodes, "samples": samples,
                 "fps": fps, "av_version": av.__version__}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    cache_path = cache_dir / f"{digest}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("signature") == signature:
            return key, cached["stats"], True
    chosen, accum = {}, {}
    for ep in episodes:
        count = min(samples, ep["length"])
        indices = np.round(np.linspace(0, ep["length"] - 1, count)).astype(int)
        first = int(round(ep["from_timestamp"] * fps))
        for frame_index in indices:
            absolute = first + int(frame_index)
            if absolute in chosen:
                raise ValueError(f"Overlapping episode samples at frame {absolute}: {path}")
            chosen[absolute] = ep["episode_index"]
        accum[ep["episode_index"]] = {"sum": np.zeros(3, np.float64),
            "sum_squares": np.zeros(3, np.float64), "hist": np.zeros((3, 256), np.int64),
            "pixels": 0, "frames": 0, "expected_frames": count}
    final = max(chosen)
    consumed = set()
    started = time.monotonic()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = threads
        stream.codec_context.thread_type = "AUTO"
        for ordinal, frame in enumerate(container.decode(stream)):
            actual_index = int(round(float(frame.pts * frame.time_base) * fps))
            if actual_index != ordinal:
                raise ValueError(f"Nonuniform video timeline {path}: ordinal {ordinal}, PTS frame {actual_index}")
            if ordinal > final:
                break
            if ordinal not in chosen:
                continue
            pixels = frame.to_ndarray(format="rgb24")
            height, width = pixels.shape[:2]
            factor = int(max(width, height) / 150) if max(width, height) >= 300 else 1
            pixels = pixels[::factor, ::factor, :].reshape(-1, 3)
            promoted = pixels.astype(np.float64)
            state = accum[chosen[ordinal]]
            state["sum"] += promoted.sum(axis=0)
            state["sum_squares"] += np.square(promoted).sum(axis=0)
            for channel in range(3):
                state["hist"][channel] += np.bincount(pixels[:, channel], minlength=256)
            state["pixels"] += len(pixels)
            state["frames"] += 1
            consumed.add(ordinal)
    if len(consumed) != len(chosen):
        raise ValueError(f"Missing frames in {path}: decoded {len(consumed)}/{len(chosen)} samples")
    result = {}
    for ep, state in accum.items():
        if state["frames"] != state["expected_frames"]:
            raise ValueError(f"Episode {ep}: incomplete sampled frames")
        mean = state["sum"] / state["pixels"] / 255
        variance = (state["sum_squares"] / state["pixels"] / (255 ** 2)) - mean ** 2
        std = np.sqrt(np.maximum(variance, 0))
        minimum = np.array([np.flatnonzero(h)[0] for h in state["hist"]]) / 255
        maximum = np.array([np.flatnonzero(h)[-1] for h in state["hist"]]) / 255
        stats = {"min": minimum, "max": maximum, "mean": mean, "std": std}
        stats.update({name: np.array([quantile(h, q) for h in state["hist"]])
                      for name, q in QUANTILES.items()})
        result[str(ep)] = {name: value.reshape(3, 1, 1).tolist()
                           for name, value in stats.items()}
        result[str(ep)]["count"] = [state["frames"]]
    write_json(cache_path, {"signature": signature, "stats": result,
                           "elapsed_seconds": time.monotonic() - started,
                           "pixel_dtype_for_moments": "float64"})
    return key, result, False


def aggregate_episode_stats(columns, episode_ids, keys):
    """Match LeRobot aggregation, including parallel variance and weighted quantiles."""
    result = {}
    for key in keys:
        prefix = f"stats/{key}/"
        names = [column[len(prefix):] for column in columns if column.startswith(prefix)]
        if not names:
            raise ValueError(f"No episode statistics for {key}")
        stats = {name: np.array([columns[prefix + name][i] for i in episode_ids], np.float64)
                 for name in names}
        means, variances = stats["mean"], np.square(stats["std"])
        counts = stats["count"]
        total_count = counts.sum(axis=0)
        weights = counts
        while weights.ndim < means.ndim:
            weights = np.expand_dims(weights, -1)
        mean = (means * weights).sum(axis=0) / total_count
        variance = ((variances + np.square(means - mean)) * weights).sum(axis=0) / total_count
        merged = {"mean": mean, "std": np.sqrt(np.maximum(variance, 0)),
                  "min": stats["min"].min(axis=0), "max": stats["max"].max(axis=0),
                  "count": total_count.astype(np.int64)}
        for name in QUANTILES:
            if name in stats:
                merged[name] = (stats[name] * weights).sum(axis=0) / total_count
        result[key] = {name: value.tolist() for name, value in merged.items()}
    return result


def link_immutable(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if os.path.samefile(source, target):
            return
        raise ValueError(f"Existing dataset file does not link to source: {target}")
    os.link(source, target)


def build_view(source, destination, info, stats, episode_table):
    destination.mkdir(parents=True, exist_ok=True)
    for path in (source / "data").rglob("*.parquet"):
        link_immutable(path, destination / path.relative_to(source))
    for key in RGB_KEYS:
        for path in (source / "videos" / key).rglob("*.mp4"):
            link_immutable(path, destination / path.relative_to(source))
    link_immutable(source / "meta/tasks.parquet", destination / "meta/tasks.parquet")
    write_json(destination / "meta/info.json", info)
    write_json(destination / "meta/stats.json", stats)
    target = destination / "meta/episodes/chunk-000/file-000.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    pq.write_table(episode_table, tmp, compression="snappy")
    os.replace(tmp, target)


def main(args):
    source, output = args.source_root.resolve(), args.output_dir.resolve()
    prefix = args.dataset_prefix
    task_name = {"task1": "Task1_ToySorting", "task2": "Task2_ParcelWeighing", "task3": "Task3_ConveyorBeltSorting"}[prefix]
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / f"{prefix}_moments_cache"
    cache_dir.mkdir(exist_ok=True)
    info = json.loads((source / "meta/info.json").read_text())
    original_stats = json.loads((source / "meta/stats.json").read_text())
    original_table = pq.read_table(source / "meta/episodes/chunk-000/file-000.parquet")
    original = original_table.to_pydict()
    count = info["total_episodes"]
    if count != 1000 or original["episode_index"] != list(range(count)):
        raise ValueError(f"Expected {task_name}'s 1,000 contiguous episode IDs")
    selected = list(range(min(args.smoke_episodes, count))) if args.smoke_episodes else list(range(count))
    grouped = {}
    for key in RGB_KEYS:
        for ep in selected:
            chunk = original[f"videos/{key}/chunk_index"][ep]
            file = original[f"videos/{key}/file_index"][ep]
            video = source / "videos" / key / f"chunk-{chunk:03d}" / f"file-{file:03d}.mp4"
            grouped.setdefault((video, key), []).append({"episode_index": ep,
                "length": original["length"][ep],
                "from_timestamp": original[f"videos/{key}/from_timestamp"][ep]})
    jobs = [(path, key, episodes, args.samples, info["fps"], args.decoder_threads, cache_dir)
            for (path, key), episodes in grouped.items()]
    repaired = copy.deepcopy(original)
    started = time.monotonic()
    completed, cache_hits = 0, 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(decode_file, job): job for job in jobs}
        for future in as_completed(futures):
            key, stats, cached = future.result()
            completed += 1
            cache_hits += int(cached)
            for ep, ep_stats in stats.items():
                for name, value in ep_stats.items():
                    repaired[f"stats/{key}/{name}"][int(ep)] = value
            print(f"RGB moments {completed}/{len(jobs)} files; {key}; "
                  f"{len(stats)} episodes; cached={cached}; elapsed={time.monotonic()-started:.1f}s", flush=True)
    summary = {"repair_version": REPAIR_VERSION, "source_root": str(source),
        "selected_episode_ids": selected, "samples_per_episode": args.samples,
        "spatial_downsampling": "same stride as LeRobot: floor(max(width,height)/150) when >=300",
        "moments_dtype": "float64", "video_decoder": f"PyAV {av.__version__}",
        "ffmpeg_library_versions": {key: list(value) for key, value in av.library_versions.items()},
        "decoder_threads": args.decoder_threads, "workers": args.workers,
        "files_decoded": len(jobs)-cache_hits, "cache_hits": cache_hits,
        "elapsed_seconds": time.monotonic()-started,
        "rgb_stats": aggregate_episode_stats(repaired, selected, RGB_KEYS)}
    for key in RGB_KEYS:
        std = np.array(summary["rgb_stats"][key]["std"])
        if not np.all(np.isfinite(std)) or np.min(std) <= .01:
            raise ValueError(f"Unexpected corrected RGB standard deviation: {key}: {std}")
    if args.smoke_episodes:
        write_json(output / f"{prefix}_rgb_smoke_receipt.json", summary)
        print("Smoke check passed; no full dataset metadata views written.", flush=True)
        return
    # Dataset-level scalar and depth statistics are preserved exactly; only RGB moments change.
    corrected_stats = copy.deepcopy(original_stats)
    corrected_stats.update(summary["rgb_stats"])
    view_info = copy.deepcopy(info)
    view_info["features"] = {k: v for k, v in view_info["features"].items() if "depth" not in k}
    permutation = np.random.default_rng(args.split_seed).permutation(count).tolist()
    validation = sorted(permutation[:100]); train = sorted(permutation[100:]); overfit = sorted(permutation[100:120])
    # LeRobot v3 metadata splits convention is a string; noncontiguous split IDs stay in manifest.
    view_info["splits"] = {"train": "0:1000"}
    corrected_table = pa.Table.from_pydict(repaired, schema=original_table.schema)
    build_view(source, output / f"{prefix}_rgb_original", view_info, original_stats, original_table)
    build_view(source, output / f"{prefix}_rgb_repaired", view_info, corrected_stats, corrected_table)
    keys = list(original_stats)
    variants = {"original": original, "repaired": repaired}
    for variant, columns in variants.items():
        write_json(output / f"{prefix}_train_stats_{variant}.json", aggregate_episode_stats(columns, train, keys))
        write_json(output / f"{prefix}_overfit_stats_{variant}.json", aggregate_episode_stats(columns, overfit, keys))
    split = {"seed": args.split_seed, "train_episodes": train, "validation_episodes": validation,
             "small_subset_episodes": overfit}
    write_json(output / f"{prefix}_split.json", split)
    manifest = {"version": 1, "task": task_name, "original_root": f"{prefix}_rgb_original",
        "corrected_root": f"{prefix}_rgb_repaired", "source_root": f"{prefix}_source", "split": f"{prefix}_split.json",
        "train_episodes": train, "validation_episodes": validation, "small_subset_episodes": overfit,
        "original_train_stats": f"{prefix}_train_stats_original.json", "repaired_train_stats": f"{prefix}_train_stats_repaired.json",
        "original_overfit_stats": f"{prefix}_overfit_stats_original.json", "repaired_overfit_stats": f"{prefix}_overfit_stats_repaired.json",
        "original_stats_episodes": train, "repaired_stats_episodes": train, "overfit_stats_episodes": overfit,
        "rgb_keys": list(RGB_KEYS), "rgb_only_feature_view": True,
        "source_scalar_and_video_files_shared_by_hardlink": True,
        "source_metadata_independent_of_views": True}
    write_json(output / f"{prefix}_experiment_manifest.json", manifest)
    scalar_unchanged = all(corrected_stats[key] == original_stats[key] for key in keys if key not in RGB_KEYS)
    if not scalar_unchanged:
        raise AssertionError("Non-RGB statistics changed")
    source_inode = (source / "meta/episodes/chunk-000/file-000.parquet").stat().st_ino
    fixed_inode = (output / f"{prefix}_rgb_repaired/meta/episodes/chunk-000/file-000.parquet").stat().st_ino
    if source_inode == fixed_inode:
        raise AssertionError("Repaired episode metadata must not hardlink source metadata")
    summary.update({"train_episode_ids": train, "validation_episode_ids": validation,
        "overfit_episode_ids": overfit, "train_frame_count": int(sum(original["length"][i] for i in train)),
        "validation_frame_count": int(sum(original["length"][i] for i in validation)),
        "overfit_frame_count": int(sum(original["length"][i] for i in overfit)),
        "train_rgb_sample_count_per_camera": int(sum(min(args.samples, original["length"][i]) for i in train)),
        "non_rgb_global_stats_preserved": scalar_unchanged, "manifest": f"{prefix}_experiment_manifest.json"})
    write_json(output / f"{prefix}_rgb_repair_receipt.json", summary)
    print(f"Full {task_name} RGB repair, fixed split, and train-only statistics completed.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-prefix", choices=("task1", "task2", "task3"), default="task1")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--decoder-threads", type=int, default=2)
    parser.add_argument("--split-seed", type=int, default=1729)
    parser.add_argument("--smoke-episodes", type=int, default=0)
    parser.add_argument("--max-threads", type=int, default=4, help="Total decoder thread budget (4 on the laptop; larger on a many-core pod)")
    args = parser.parse_args()
    if (args.samples < 2 or not 1 <= args.workers <= args.max_threads or not 1 <= args.decoder_threads <= 4
            or args.workers * args.decoder_threads > args.max_threads or args.smoke_episodes < 0):
        parser.error("Need >=2 samples, nonnegative smoke episodes, and at most --max-threads total decoder threads")
    main(args)

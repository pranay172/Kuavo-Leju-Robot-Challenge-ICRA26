#!/usr/bin/env python3
"""Fast ACT weights-only experiments: grouped AdamW, EMA and spawned data workers.

Existing trainers/checkpoints are untouched. --init-checkpoint loads weights only;
optimizer/schedule always start fresh. Every snapshot contains raw/ema policy +
processors. --benchmark is a bounded throughput probe, not a saved training run.
"""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import math
import os
import random
import signal
import subprocess
import threading
import time
import types
from pathlib import Path

from train_act_experiment import (
    WORKSPACE, RGB_KEYS, build_augmentation, configure_policy, import_runtime,
    load_manifest, prepare_data, read_json, save_checkpoint, seed_everything,
    validate, write_json,
)

_STOP = False


def seed_worker(worker_id):
    """DataLoader already seeds torch; seed Python/NumPy and bound decoder threads."""
    import numpy as np
    import torch
    value = torch.initial_seed() % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.set_num_threads(1)


def numpy_collate(items):
    """Avoid torch shared-memory IPC (the policy Docker has only 64 MiB shm).

    Frames remain float32 with the original decoder values. NumPy is serialized
    through multiprocessing pipes; conversion/pinning happens in the parent.
    """
    import numpy as np
    import torch
    first = items[0]
    if isinstance(first, dict):
        return {key: numpy_collate([item[key] for item in items]) for key in first}
    if isinstance(first, torch.Tensor):
        return np.stack([item.detach().cpu().numpy() for item in items])
    if isinstance(first, np.ndarray):
        return np.stack(items)
    if isinstance(first, (bool, int, float, np.number)):
        return np.asarray(items)
    if isinstance(first, str):
        return list(items)
    if isinstance(first, (tuple, list)):
        return [numpy_collate(list(values)) for values in zip(*items)]
    raise TypeError(f"Unsupported data field type: {type(first)}")


def tensor_batch(raw, torch):
    import numpy as np
    if isinstance(raw, dict):
        return {key: tensor_batch(value, torch) for key, value in raw.items()}
    if isinstance(raw, np.ndarray) and raw.dtype.kind not in "OUS":
        return torch.from_numpy(raw)
    if isinstance(raw, (list, tuple)):
        return [tensor_batch(value, torch) for value in raw]
    return raw


def cached_query_videos(reader, query_timestamps, ep_idx):
    """Replace only RGB decoding; Arrow indexing/action chunks/padding are untouched."""
    import torch
    from torchvision.io import read_file, decode_jpeg, ImageReadMode
    episode = reader.meta.episodes[ep_idx]
    result = {}
    for key, timestamps in query_timestamps.items():
        relative = str(reader.meta.get_video_file_path(ep_idx, key))
        descriptor = reader._frame_cache_index[relative]
        fps = float(descriptor["fps"])
        start = float(descriptor.get("start_time", 0.0))
        offset = float(episode[f"videos/{key}/from_timestamp"])
        frames = []
        for timestamp in timestamps:
            shifted = offset + float(timestamp)
            index = round((shifted - start) * fps)
            if not 0 <= index < descriptor["frame_count"]:
                raise IndexError(f"Cache timestamp {shifted} outside {relative}: frame {index}")
            if abs(start + index / fps - shifted) > reader.tolerance_s + 1e-7:
                raise ValueError(f"Cache timestamp mismatch in {relative}: {shifted}")
            path = reader._frame_cache_root / descriptor["directory"] / descriptor["pattern"].format(frame_index=index)
            image = decode_jpeg(read_file(str(path)), mode=ImageReadMode.RGB, device="cpu")
            # Video decoder returns CHW float32 in [0,1]; retain exactly that contract.
            frames.append(image.to(torch.float32).div_(255.0))
        result[key] = torch.stack(frames).squeeze(0)
    return result


def attach_frame_cache(dataset, cache):
    cache = Path(cache).resolve()
    metadata = read_json(cache / "metadata.json")
    if metadata.get("status") != "complete":
        raise ValueError(f"Frame cache is not complete: {cache}")
    dataset._frame_cache_root = cache
    dataset._frame_cache_index = metadata["video_files"]
    dataset._query_videos = types.MethodType(cached_query_videos, dataset)
    return dataset


def channels_last_backbone_input(module, arguments):
    import torch
    if arguments and isinstance(arguments[0], torch.Tensor) and arguments[0].ndim == 4:
        return (arguments[0].contiguous(memory_format=torch.channels_last), *arguments[1:])
    return arguments


def configure_execution(args, torch):
    # seed_everything sets deterministic=True; apply speed options AFTER each seed reset.
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    torch.backends.cudnn.deterministic = not args.cudnn_benchmark


class LazyVideoDataset:
    """Spawn only a small reader specification, not the large HF/Arrow dataset.

    Each child opens the existing local dataset once. No parent decoder state or
    large Arrow pickle crosses the spawn pipe, and global indexing is preserved.
    """
    def __init__(self, dataset):
        self.spec = {"repo_id": dataset.repo_id, "root": str(dataset.root),
                     "delta_timestamps": dataset.delta_timestamps,
                     "video_backend": dataset.video_backend}
        self.frame_cache = getattr(dataset, "_frame_cache_root", None)
        self.length = len(dataset)
        self.reader = None

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.reader is None:
            import lerobot_patches.custom_patches
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
            self.reader = LeRobotDataset(**self.spec, episodes=None,
                                        image_transforms=None, download_videos=False)
            if self.frame_cache is not None:
                attach_frame_cache(self.reader, self.frame_cache)
        return self.reader[index]


class TensorLoader:
    def __init__(self, loader, torch, deadline=None):
        self.loader, self.torch = loader, torch
        self.deadline = deadline

    def __iter__(self):
        for batch in self.loader:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                break
            yield tensor_batch(batch, self.torch)

    def __len__(self):
        return len(self.loader)


def make_loader(runtime, dataset, indices, microbatch, workers, seed, shuffle,
                prefetch=2, timeout=120, sampler=None):
    torch = runtime["torch"]
    from torch.utils.data import DataLoader, Subset
    generator = torch.Generator().manual_seed(seed)
    settings = dict(batch_size=microbatch, shuffle=False if sampler is not None else shuffle, num_workers=workers,
                    generator=generator, pin_memory=True,
                    worker_init_fn=seed_worker)
    if sampler is not None: settings["sampler"] = sampler
    if workers:
        settings.update(persistent_workers=True, prefetch_factor=prefetch,
                        multiprocessing_context="spawn", timeout=timeout)
        if getattr(dataset, "_worker_transport", "tensor") == "numpy":
            settings["collate_fn"] = numpy_collate
    reader = LazyVideoDataset(dataset) if workers else dataset
    return TensorLoader(DataLoader(Subset(reader, indices), **settings), torch)


def close_loader(loader):
    """Best-effort owned-worker cleanup; reader/iteration failures still propagate."""
    detail = {"status": "none", "shutdown_error": None, "forced_worker_pids": [],
              "worker_exitcodes": {}}
    if loader is None:
        return detail
    iterator = getattr(loader.loader, "_iterator", None)
    workers = list(getattr(iterator, "_workers", ())) if iterator is not None else []
    detail["status"] = "clean"
    if iterator is not None:
        try:
            # PyTorch unregisters its SIGCHLD worker tracking in this method's finally.
            iterator._shutdown_workers()
        except Exception as error:
            detail.update(status="cleanup_warning",
                          shutdown_error=f"{type(error).__name__}: {error}")
        finally:
            # Only reap this DataLoader's children; never signal another job/process.
            reap_deadline = time.monotonic() + 10
            for worker in workers:
                try:
                    worker.join(timeout=min(.2, max(0, reap_deadline-time.monotonic())))
                    if worker.is_alive():
                        detail["forced_worker_pids"].append(worker.pid)
                        detail["status"] = "cleanup_warning"
                        worker.terminate()
                        worker.join(timeout=min(.3, max(0, reap_deadline-time.monotonic())))
                    if worker.is_alive():
                        worker.kill()
                        worker.join(timeout=min(.3, max(0, reap_deadline-time.monotonic())))
                    detail["worker_exitcodes"][str(worker.pid)] = worker.exitcode
                except Exception as error:
                    detail.setdefault("reap_errors", []).append(f"{worker.pid}: {type(error).__name__}: {error}")
                    detail["status"] = "cleanup_warning"
            loader.loader._iterator = None
    gc.collect()
    return detail


def build_optimizer(policy, config):
    groups = policy.get_optim_params()
    expected = {id(parameter) for parameter in policy.parameters() if parameter.requires_grad}
    actual = [id(parameter) for group in groups for parameter in group["params"]]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Optimizer groups must cover every trainable parameter exactly once")
    if len(groups) != 2 or any(not group["params"] for group in groups):
        raise ValueError("Expected nonempty ACT main/backbone optimizer groups")
    groups[0].update(name="main", lr=config.optimizer_lr)
    groups[1].update(name="backbone", lr=config.optimizer_lr_backbone)
    optimizer = config.get_optimizer_preset().build(groups)
    return optimizer, [{"name": group["name"], "lr": group["lr"],
                        "parameter_count": sum(p.numel() for p in group["params"])}
                       for group in optimizer.param_groups]


class WeightEMA:
    def __init__(self, policy, decay):
        self.decay = decay
        self.policy = copy.deepcopy(policy).eval()
        for parameter in self.policy.parameters():
            parameter.requires_grad_(False)
        self.updates = 0

    def update(self, raw, torch):
        with torch.no_grad():
            effective_decay = min(self.decay, (1.0+self.updates)/(10.0+self.updates))
            source = raw.state_dict()
            for name, target in self.policy.state_dict().items():
                current = source[name].detach()
                if target.is_floating_point() or target.is_complex():
                    target.lerp_(current, 1.0 - effective_decay)
                else:
                    target.copy_(current)
        self.updates += 1


def process_batch(runtime, raw, preprocessor, device, augmentation):
    torch = runtime["torch"]
    batch = {}
    for key, value in raw.items():
        if isinstance(value, torch.Tensor):
            # Workers use NumPy IPC; explicit parent pinning preserves async H2D.
            if device.startswith("cuda") and not value.is_pinned():
                value = value.pin_memory()
            value = value.to(device, non_blocking=True)
        batch[key] = value
    if augmentation is not None:
        for key in RGB_KEYS:
            batch[key] = augmentation(batch[key])
    return preprocessor(batch)


def merge_batches(batches, torch):
    if len(batches) == 1:
        return batches[0]
    first = batches[0]
    if isinstance(first, dict):
        return {key: merge_batches([batch[key] for batch in batches], torch) for key in first}
    if isinstance(first, torch.Tensor):
        return torch.cat(batches, dim=0)
    if isinstance(first, list):
        return list(itertools.chain.from_iterable(batches))
    raise TypeError(f"Unsupported merged batch field: {type(first)}")


def slice_batch(batch, start, end, torch):
    if isinstance(batch, dict):
        return {key: slice_batch(value, start, end, torch) for key, value in batch.items()}
    if isinstance(batch, torch.Tensor):
        return batch[start:end]
    if isinstance(batch, list):
        return batch[start:end]
    raise TypeError(f"Unsupported sliced batch field: {type(batch)}")


def pooled_group(pool, microbatch, accumulation, torch):
    result = []
    for _ in range(accumulation):
        if microbatch >= 8 and not pool["pending"]:
            chunks = list(itertools.islice(pool["iterator"], microbatch // 8))
            if not chunks:
                break
            result.append(merge_batches(chunks, torch))
            continue
        chunks = []
        for _ in range(microbatch // 4):
            if not pool["pending"]:
                try:
                    native = next(pool["iterator"])
                except StopIteration:
                    break
                count = len(native["action"])
                pool["pending"].extend(slice_batch(native, i, min(i+4, count), torch)
                                       for i in range(0, count, 4))
            chunks.append(pool["pending"].pop(0))
        if not chunks:
            break
        result.append(merge_batches(chunks, torch))
    return result


def optimize(runtime, policy, raw_group, preprocessor, device, augmentation,
             optimizer, scaler, amp):
    torch = runtime["torch"]
    samples = sum(len(raw["action"]) for raw in raw_group)
    optimizer.zero_grad(set_to_none=True)
    metrics = {"loss": 0.0, "l1_loss": 0.0, "kld_loss": 0.0}
    for raw in raw_group:
        weight = len(raw["action"]) / samples
        batch = process_batch(runtime, raw, preprocessor, device, augmentation)
        with torch.autocast(device_type=device.split(":")[0], dtype=runtime.get("autocast_dtype", torch.float16), enabled=amp):
            loss, terms = policy.forward(batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite ACT loss")
        scaler.scale(loss * weight).backward()
        metrics["loss"] += float(loss.detach()) * weight
        for name in ("l1_loss", "kld_loss"):
            metrics[name] += float(terms.get(name, 0.0)) * weight
    previous_scale = scaler.get_scale()
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), policy.config.get_optimizer_preset().grad_clip_norm,
        error_if_nonfinite=not scaler.is_enabled())
    # GradScaler detects nonfinite gradients in unscale_ BEFORE clipping.
    # Its step still skips/backoffs on fp16 overflow; bf16/FP32 abort above.
    # Rare finite-gradient norm overflow must also skip (not a momentum-only update).
    finite_norm_overflow = (scaler.is_enabled()
        and not bool(torch.isfinite(gradient_norm).item())
        and all(bool(torch.isfinite(parameter.grad).all().item())
                for parameter in policy.parameters() if parameter.grad is not None))
    if finite_norm_overflow:
        scaler.update(new_scale=previous_scale * scaler.get_backoff_factor())
        successful = False
    else:
        scaler.step(optimizer)
        scaler.update()
        successful = scaler.get_scale() >= previous_scale
    optimizer.zero_grad(set_to_none=True)
    return metrics, samples, successful


def new_scaler(torch, amp, autocast_dtype="float16"):
    return torch.amp.GradScaler("cuda", enabled=amp and autocast_dtype == "float16")


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        if key in ("MemAvailable", "SwapFree"):
            values[key] = int(value.split()[0]) * 1024
    return values


def loader_memory(loader):
    iterator = getattr(loader.loader, "_iterator", None)
    workers = getattr(iterator, "_workers", ())
    rss, pss = [], []
    for worker in workers:
        try:
            status = Path(f"/proc/{worker.pid}/status").read_text()
            rollup = Path(f"/proc/{worker.pid}/smaps_rollup").read_text()
            for rows, key, result in ((status, "VmRSS:", rss), (rollup, "Pss:", pss)):
                value = next(line for line in rows.splitlines() if line.startswith(key))
                result.append(int(value.split()[1]) * 1024)
        except (OSError, StopIteration):
            pass
    return {"worker_rss_sum_bytes": sum(rss), "worker_pss_sum_bytes": sum(pss),
            "worker_processes_measured": len(rss)}


class UtilizationSampler:
    def __init__(self):
        self.samples = []
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._collect, daemon=True)

    def _collect(self):
        while not self.event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2, check=False)
                if result.returncode == 0:
                    util, memory = result.stdout.splitlines()[0].split(",")
                    self.samples.append((time.monotonic(), float(util), float(memory)))
            except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
                pass
            self.event.wait(0.75)

    def start(self):
        self.thread.start()

    def stop(self):
        self.event.set()
        self.thread.join(timeout=3)

    def window(self, start, end):
        rows = [row for row in self.samples if start <= row[0] <= end]
        return {"gpu_util_mean_percent": sum(row[1] for row in rows) / len(rows) if rows else None,
                "gpu_util_samples": len(rows),
                "nvidia_smi_peak_memory_mib": max((row[2] for row in rows), default=None)}


def export_snapshot(output, step, raw, ema, preprocessor, postprocessor, metrics=None):
    destination = output / f"step{step:06d}"
    if destination.exists():
        if (destination/"raw/model.safetensors").exists() and (destination/"ema/model.safetensors").exists():
            return str(destination)
        destination=output/f"step{step:06d}_{time.time_ns()}"
    common = {"optimizer_step": step, "ema_decay": ema.decay,
              "ema_updates": ema.updates, "validation": metrics,
              "initialization": getattr(raw, "_initialization_description", "fresh optimizer/schedule")}
    save_checkpoint(destination / "raw", raw, preprocessor, postprocessor,
                    {**common, "weights": "raw"})
    save_checkpoint(destination / "ema", ema.policy, preprocessor, postprocessor,
                    {**common, "weights": "ema"})
    return str(destination)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--manifest", type=Path, default=WORKSPACE / "recovery/datasets/task1_experiment_manifest.json")
    result.add_argument("--source", type=Path, default=WORKSPACE / "kuavo_data_challenge")
    result.add_argument("--policy-config", type=Path, default=WORKSPACE / "recovery/experiments/act_task1_reference/config.json")
    result.add_argument("--variant", choices=("original", "repaired"), default="repaired")
    result.add_argument("--mode", choices=("overfit", "ab"), default="ab")
    result.add_argument("--output", type=Path, help="Required for fresh run; full resume recovers and enforces saved run output")
    result.add_argument("--init-checkpoint", type=Path)
    result.add_argument("--resume", "--resume-state", dest="resume", type=Path)
    result.add_argument("--resume-every", type=int, default=2500)
    result.add_argument("--log-every", type=int, default=25)
    result.add_argument("--lr-schedule", choices=("constant","cosine"), default="constant")
    result.add_argument("--decay-steps", type=int, help="New decay phase from saved current LR over N additional updates")
    result.add_argument("--seed", type=int, default=2025)
    result.add_argument("--max-steps", type=int, default=40000)
    result.add_argument("--schedule-steps", type=int)
    result.add_argument("--warmup-steps", type=int, default=500)
    result.add_argument("--snapshot-every", type=int, default=5000)
    result.add_argument("--validation-every", type=int, default=1000)
    result.add_argument("--validation-frames-per-episode", type=int, default=5)
    result.add_argument("--max-validation-batches", type=int)
    result.add_argument("--microbatch", type=int, default=16)
    result.add_argument("--accumulation", type=int, default=2)
    result.add_argument("--workers", type=int, default=8)
    result.add_argument("--prefetch", type=int, default=2)
    result.add_argument("--loader-timeout", type=int, default=120)
    result.add_argument("--device", default="cuda")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--frame-cache", type=Path, help="Complete JPEG cache metadata.json root; only replaces RGB video decoding")
    result.add_argument("--worker-transport", choices=("tensor", "numpy"), default="tensor", help="tensor needs ample /dev/shm; numpy retains laptop-compatible pipe transport")
    result.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--autocast-dtype", choices=("float16", "bfloat16"), default="float16")
    result.add_argument("--ema-decay", type=float, default=0.9995)
    result.add_argument("--lr", type=float, default=3e-5)
    result.add_argument("--lr-backbone", type=float, default=1e-5)
    result.add_argument("--weight-decay", type=float)
    result.add_argument("--augmentation-config", type=Path, default=WORKSPACE / "recovery/experiments/act_task1_full900_augmentation.json")
    result.add_argument("--benchmark", action="store_true")
    result.add_argument("--benchmark-seconds", type=float, default=180)
    result.add_argument("--deadline-unix", type=float, help="External hard stage deadline (Unix UTC seconds)")
    result.add_argument("--max-wall-seconds", type=float, default=21600,
                        help="Wall-clock training cap in seconds")
    return result


def initialize(args):
    runtime = import_runtime(args.source)
    torch, np = runtime["torch"], runtime["np"]
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    seed_everything(torch, np, args.seed)
    configure_execution(args, torch)
    runtime["autocast_dtype"] = getattr(torch, args.autocast_dtype)
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    manifest = load_manifest(args.manifest)
    root = manifest["original_root" if args.variant == "original" else "corrected_root"]
    metadata = runtime["LeRobotDatasetMetadata"](f"lerobot/{manifest['task']}", root=root)
    config = configure_policy(runtime, args.policy_config, metadata, args.device, args.amp)
    for flag, field in (("lr", "optimizer_lr"), ("lr_backbone", "optimizer_lr_backbone"),
                        ("weight_decay", "optimizer_weight_decay")):
        if getattr(args, flag) is not None:
            setattr(config, field, getattr(args, flag))
    dataset, train_indices, validation_indices, stats, stats_key = prepare_data(
        runtime, manifest, args.variant, args.mode, config, args.validation_frames_per_episode)
    dataset._worker_transport = args.worker_transport
    if args.frame_cache:
        attach_frame_cache(dataset, args.frame_cache)
    policy = runtime["CustomACTPolicyWrapper"](config)
    if args.init_checkpoint:
        from safetensors.torch import load_file
        location = args.init_checkpoint.resolve()
        weights = location / "model.safetensors" if location.is_dir() else location
        policy.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    policy._initialization_description = ("checkpoint weights only; fresh optimizer/schedule" if args.init_checkpoint else "ImageNet backbone; fresh ACT/VAE and optimizer/schedule")
    policy.to(args.device)
    if args.channels_last:
        policy.model.backbone.to(memory_format=torch.channels_last)
        policy.model.backbone.register_forward_pre_hook(channels_last_backbone_input)
    preprocessor, postprocessor = runtime["make_pre_post_processors"](config, dataset_stats=stats)
    augmentation = build_augmentation(runtime, read_json(args.augmentation_config)) if args.mode == "ab" else None
    return runtime, config, dataset, train_indices, validation_indices, policy, preprocessor, postprocessor, augmentation


def benchmark(args, initialized, started):
    runtime, config, dataset, train_indices, _, policy, preprocessor, _, augmentation = initialized
    torch = runtime["torch"]
    if not args.device.startswith("cuda"):
        raise ValueError("Throughput benchmark requires CUDA")
    output = args.output.resolve()
    deadline = started + args.benchmark_seconds
    if args.deadline_unix:
        deadline = min(deadline, time.monotonic() + max(0, args.deadline_unix - time.time()))
    initial_weights = {name: value.detach().cpu().clone() for name, value in policy.state_dict().items()}
    sampler = UtilizationSampler()
    sampler.start()
    cases = [(workers, batch) for workers in (0, 4, 8) for batch in (4, 8)]
    records = []
    pool = {}
    cleanup_records = []
    try:
        for case_index, (workers, microbatch) in enumerate(cases):
            if _STOP or time.monotonic() >= deadline - 8:
                records.append({"workers": workers, "microbatch": microbatch, "status": "skipped_budget"})
                continue
            if pool and pool["workers"] != workers:
                cleanup_started = time.monotonic()
                pool["iterator"] = None
                pool["pending"] = []
                close_loader(pool["loader"])
                cleanup_records.append({"workers": pool["workers"], "seconds": time.monotonic() - cleanup_started})
                pool.clear()
            remaining_cases = len(cases) - case_index
            case_budget = min(28.0, max(8.0, (deadline - time.monotonic() - 5) / remaining_cases))
            case_deadline = min(deadline - 10, time.monotonic() + case_budget)
            record, oom = benchmark_case(args, initialized, initial_weights, workers, microbatch,
                                         case_deadline, sampler, pool)
            records.append(record)
            print(json.dumps({"benchmark_case": record}), flush=True)
            write_json(output / "summary.json", {"kind": "throughput_benchmark", "status": "running", "cases": records})
        # Batch16 would only be useful if twice the measured batch8 activation
        # memory plus model/optimizer/EMA buffers has comfortable headroom.
        batch8 = [row for row in records if row.get("microbatch") == 8 and row.get("status") == "valid"]
        capacity = torch.cuda.get_device_properties(args.device).total_memory
        best8 = max(batch8, key=lambda row: row["samples_per_second"], default=None)
        estimated16 = 2 * best8["peak_allocated_bytes"] if best8 else capacity * 2
        if best8 and estimated16 < capacity * 0.88 and time.monotonic() < deadline - 15:
            if pool and pool["workers"] != best8["workers"]:
                pool["iterator"] = None
                pool["pending"] = []
                close_loader(pool["loader"])
                pool.clear()
            row, _ = benchmark_case(args, initialized, initial_weights, best8["workers"], 16,
                                    deadline - 10, sampler, pool)
            row["screened_workers16"] = "best batch8 loader; larger batch feasibility"
            records.append(row)
        else:
            records.append({"microbatch": 16, "status": "skipped_memory_or_budget",
                            "estimated_peak_bytes_conservative": estimated16,
                            "gpu_capacity_bytes": capacity})
    finally:
        if pool:
            cleanup_started = time.monotonic()
            pool["iterator"] = None
            pool["pending"] = []
            close_loader(pool["loader"])
            cleanup_records.append({"workers": pool["workers"], "seconds": time.monotonic() - cleanup_started})
            pool.clear()
        sampler.stop()
    valid = [row for row in records if row.get("status") == "valid"]
    winner = max(valid, key=lambda row: row["samples_per_second"], default=None)
    summary = {"kind": "throughput_benchmark", "status": "complete", "cases": records,
               "elapsed_seconds": time.monotonic() - started,
               "budget_seconds": args.benchmark_seconds, "effective_batch": 16,
               "optimizer": {"lr": config.optimizer_lr, "lr_backbone": config.optimizer_lr_backbone,
                             "weight_decay": config.optimizer_weight_decay},
               "ema_decay": args.ema_decay, "init_checkpoint": str(args.init_checkpoint),
               "recommended": {key: winner[key] for key in ("workers", "microbatch", "accumulation", "samples_per_second")} if winner else None,
               "measurement": "steady optimizer steps after one warmup; H2D, decoding, augmentation and EMA included; no validation/snapshot",
               "worker_transport": "spawn + lazy local readers + NumPy IPC avoids Docker64MiB shared-memory limit",
               "worker_pool_cleanup": cleanup_records,
               "benchmark_loader_batch": 8,
               "benchmark_batch_composition": "native8 loader; slice into4 views for GPUmicrobatch4, concatenate two8 batches for16; reuse worker pools",
               "total_peak_memory_includes": "raw model, EMA model, optimizer and AMP activations"}
    write_json(output / "summary.json", summary)
    lines = ["# ACT v2 throughput benchmark", "", f"Elapsed {summary['elapsed_seconds']:.1f}s, effective batch16; no saved training weights.", "",
             "| Workers | Microbatch | Status | Updates/s | Samples/s | Peak GiB | GPU util % |", "|---:|---:|---|---:|---:|---:|---:|"]
    for row in records:
        def number(key, divisor=1):
            value = row.get(key)
            return f"{value/divisor:.2f}" if value is not None else "—"
        lines.append(f"| {row.get('workers', '—')} | {row['microbatch']} | {row['status']} | {number('updates_per_second')} | {number('samples_per_second')} | {number('peak_allocated_bytes', 2**30)} | {number('gpu_util_mean_percent')} |")
    lines.extend(["", f"Recommendation: {summary['recommended']}", "",
                  "Short probe; worker startup and decoder variability are reported per case. Batch16 memory screening is conservative."])
    (output / "NOTE.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"benchmark_summary": summary}), flush=True)


def benchmark_case(args, initialized, initial_weights, workers, microbatch, deadline, sampler, pool):
    runtime, config, dataset, train_indices, _, policy, preprocessor, _, augmentation = initialized
    torch = runtime["torch"]
    record = {"workers": workers, "microbatch": microbatch, "accumulation": 16 // microbatch,
              "effective_batch": 16, "status": "invalid", "updates": 0,
              "host_memory_before": host_memory()}
    loader = optimizer = ema = raw_group = None
    start = time.monotonic()
    measure_start = measure_end = None
    updates = samples = skipped = 0
    try:
        policy.load_state_dict(initial_weights, strict=True)
        policy.train()
        optimizer, groups = build_optimizer(policy, config)
        record["parameter_groups"] = groups
        ema = WeightEMA(policy, args.ema_decay)
        scaler = new_scaler(torch, config.use_amp, args.autocast_dtype)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(args.device)
        record["worker_pool_reused"] = bool(pool)
        if not pool:
            loader = make_loader(runtime, dataset, train_indices, 8, workers,
                                 args.seed, True, args.prefetch, max(args.loader_timeout, 30))
            pool.update(workers=workers, loader=loader, iterator=iter(loader), pending=[])
        else:
            loader = pool["loader"]
        warmup = True
        while not _STOP and time.monotonic() < deadline:
            raw_group = pooled_group(pool, microbatch, record["accumulation"], torch)
            if not raw_group:
                break
            _, count, success = optimize(runtime, policy, raw_group, preprocessor, args.device,
                                         augmentation, optimizer, scaler, config.use_amp)
            if success:
                ema.update(policy, torch)
            else:
                skipped += 1
            torch.cuda.synchronize(args.device)
            raw_group = None
            now = time.monotonic()
            if warmup:
                measure_start = now
                record["warmup_and_loader_startup_seconds"] = now - start
                record.update(loader_memory(loader))
                record["host_memory_after_warmup"] = host_memory()
                warmup = False
            elif success:
                updates += 1
                samples += count
                measure_end = now
        elapsed = (measure_end - measure_start) if measure_end is not None else 0
        record.update(status="valid" if updates >= 2 and elapsed > 0 else "insufficient_budget",
                      updates=updates, samples=samples, measurement_seconds=elapsed,
                      updates_per_second=updates/elapsed if elapsed else None,
                      samples_per_second=samples/elapsed if elapsed else None,
                      skipped_amp_steps=skipped,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device),
                      total_case_seconds=time.monotonic() - start)
        if elapsed:
            record.update(sampler.window(measure_start, measure_end))
    except torch.OutOfMemoryError as error:
        record.update(status="oom", error=str(error).splitlines()[0],
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device))
    except Exception as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
    finally:
        raw_group = optimizer = ema = None
        policy.zero_grad(set_to_none=True)
        record["loader_cleanup_seconds"] = 0.0
        record["loader_cleanup_note"] = "pool retained for next microbatch; final/group cleanup reported separately"
        record["host_memory_after_cleanup"] = host_memory()
        record["total_case_including_cleanup_seconds"] = time.monotonic() - start
        gc.collect()
        torch.cuda.empty_cache()
    return record, record["status"] == "oom"


def train(args, initialized, started):
    from act_resume_state import (EpochOffsetSampler, capture_rng, restore_rng,
        make_scheduler, save_resume)
    runtime, config, dataset, train_indices, validation_indices, policy, preprocessor, postprocessor, augmentation = initialized
    torch, np = runtime["torch"], runtime["np"]
    output = args.output.resolve()
    resumed = getattr(args, "_resume_payload", None)
    optimizer, groups = build_optimizer(policy, config)
    scaler = new_scaler(torch, config.use_amp, args.autocast_dtype)
    ema = WeightEMA(policy, args.ema_decay)
    step = samples = skipped = epoch = batches_in_epoch = consecutive_skipped = 0
    previous_total = previous_phase = 0.0
    phase = "main"
    history = {}
    if resumed:
        policy.load_state_dict(resumed["raw"], strict=True)
        ema.policy.load_state_dict(resumed["ema"], strict=True)
        optimizer.load_state_dict(resumed["optimizer"])
        scaler.load_state_dict(resumed["scaler"])
        step = int(resumed["optimizer_step"])
        samples = int(resumed["consumed_samples"])
        skipped = int(resumed.get("skipped_amp_updates", 0))
        epoch = int(resumed.get("epoch", 0))
        batches_in_epoch = int(resumed.get("batches_in_epoch", 0))
        ema.updates = int(resumed["ema_updates"])
        previous_total = float(resumed["total_training_seconds"])
        previous_phase = float(resumed.get("phase_training_seconds", previous_total))
        phase = resumed.get("phase", "main")
        history = copy.deepcopy(resumed.get("summary", {}))
        spec = copy.deepcopy(resumed["schedule_spec"])
        saved_lrs = [group["lr"] for group in optimizer.param_groups]
        scheduler = make_scheduler(optimizer, spec)
        scheduler.load_state_dict(resumed["scheduler"])
        # LambdaLR construction initializes LR; preserve saved optimizer LRs exactly.
        for group, value in zip(optimizer.param_groups, saved_lrs): group["lr"] = value
        raw_equal = all(torch.equal(value.detach().cpu(), resumed["raw"][name])
                        for name, value in policy.state_dict().items())
        ema_equal = all(torch.equal(value.detach().cpu(), resumed["ema"][name])
                        for name, value in ema.policy.state_dict().items())
        if not raw_equal or not ema_equal: raise RuntimeError("Resume weight equality failed")
        print(json.dumps({"resume_restored": {
            "raw_weights_equal": raw_equal, "ema_weights_equal": ema_equal,
            "optimizer_step": step, "step": step, "learning_rates": saved_lrs,
            "scaler_scale": scaler.get_scale(), "ema_updates": ema.updates,
            "consumed_samples": samples, "elapsed_previous": previous_total,
            "epoch": epoch, "batches_in_epoch": batches_in_epoch,
            "phase": phase, "validation_history": len(history.get("validation", [])),
            "snapshot_history": len(history.get("snapshots", []))}}), flush=True)
        if args.decay_steps is not None:
            if phase == "decay":
                raise ValueError("Existing decay resumes without --decay-steps; do not reset its clock")
            phase = "decay"
            previous_phase = 0.0
            args.max_steps = step + args.decay_steps
            spec = {"kind": "decay", "decay_steps": args.decay_steps, "start_step": step}
            for group, value in zip(optimizer.param_groups, saved_lrs):
                group["lr"] = value
                group["initial_lr"] = value
            scheduler = make_scheduler(optimizer, spec)
        else:
            if phase == "decay":
                args.max_steps = spec["start_step"] + spec["decay_steps"]
        rng_warnings = restore_rng(torch, np, resumed["rng"])
        if rng_warnings: print(json.dumps({"resume_rng_warnings": rng_warnings}), flush=True)
    else:
        spec = {"kind": args.lr_schedule, "warmup_steps": args.warmup_steps,
                "total_steps": args.schedule_steps or args.max_steps}
        scheduler = make_scheduler(optimizer, spec)
    train_loader = validation_loader = None
    summary = history
    summary.update(kind="act_v2_training", status="running", max_steps=args.max_steps,
        train_frames=len(train_indices), validation_frames=len(validation_indices),
        init_checkpoint=str(args.init_checkpoint) if args.init_checkpoint else None,
        weights_only_initialization=bool(args.init_checkpoint),
        initialization="full resume" if resumed else "ImageNet backbone; fresh ACT/VAE",
        frame_cache=str(args.frame_cache) if args.frame_cache else None,
        autocast_dtype=args.autocast_dtype, cudnn_benchmark=args.cudnn_benchmark,
        channels_last_backbone=args.channels_last, workers=args.workers,
        microbatch=args.microbatch, accumulation=args.accumulation,
        effective_batch=args.microbatch*args.accumulation, amp=config.use_amp,
        ema_decay=args.ema_decay, ema_warmup="min(target, (1+n)/(10+n)), n prior EMA updates",
        parameter_groups=groups, schedule_spec=spec, phase=phase, phase_kind=phase,
        phase_wall_budget=args.max_wall_seconds, weight_decay=config.optimizer_weight_decay,
        snapshot_every=args.snapshot_every, resume_every=args.resume_every,
        validation_uses="EMA; zero VAE prior; no temporal ensemble",
        source=str(args.source.resolve()), manifest=str(args.manifest.resolve()),
        worker_transport=f"spawn + {args.worker_transport} IPC" if args.workers else "in-process")
    summary.setdefault("snapshots", [])
    summary.setdefault("validation", [])
    last_snapshot = summary["snapshots"][-1] if summary["snapshots"] else None
    last_snapshot_step = (int(Path(last_snapshot).name.split("_")[0][4:]) if last_snapshot else -1)
    latest_metrics = None
    invocation_initial_samples = samples
    invocation_initial_time = time.monotonic()
    sampler = EpochOffsetSampler(len(train_indices), args.seed, epoch,
                                 batches_in_epoch*args.microbatch)
    def update_summary():
        active = time.monotonic() - started
        summary.update(optimizer_step=step, step=step, consumed_samples=samples,
            consumed_frames=samples, ema_updates=ema.updates, epoch=epoch,
            batches_in_epoch=batches_in_epoch, skipped_amp_updates=skipped,
            elapsed_seconds=previous_total+active, total_training_seconds=previous_total+active,
            cumulative_training_seconds=previous_total+active,
            phase_training_seconds=previous_phase+active,
            invocation_elapsed_seconds=active, learning_rates=scheduler.get_last_lr(),
            scaler_scale=scaler.get_scale(),
            samples_per_second=(samples-invocation_initial_samples)/max(.001,time.monotonic()-invocation_initial_time))
    def snapshot():
        nonlocal last_snapshot, last_snapshot_step
        last_snapshot = export_snapshot(output, step, policy, ema, preprocessor,
                                        postprocessor, latest_metrics)
        last_snapshot_step = step
        if last_snapshot not in summary["snapshots"]: summary["snapshots"].append(last_snapshot)
        print(json.dumps({"snapshot_written": {"optimizer_step": step, "path": last_snapshot,
              "raw": str(Path(last_snapshot)/"raw"), "ema": str(Path(last_snapshot)/"ema")}}), flush=True)
    def resume_state(reason):
        update_summary()
        arguments = {key:value for key,value in vars(args).items() if not key.startswith("_")}
        payload = {key:copy.deepcopy(summary.get(key)) for key in (
            "optimizer_step","phase","phase_kind","phase_training_seconds","total_training_seconds",
            "elapsed_seconds","ema_updates","consumed_samples","learning_rates","scaler_scale","epoch",
            "batches_in_epoch","schedule_spec","phase_wall_budget","skipped_amp_updates")}
        payload.update(raw=policy.state_dict(), ema=ema.policy.state_dict(),
            optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
            scaler=scaler.state_dict(), rng=capture_rng(torch,np),
            arguments=arguments, summary=copy.deepcopy(summary))
        path = save_resume(output,step,payload,torch,keep=2)
        summary["last_resume"] = path
        print(json.dumps({"resume_written": {"optimizer_step": step, "path": path, "reason": reason}}), flush=True)
        return path
    def stop_requested():
        return _STOP or (output/"STOP").exists()
    def budget_exhausted():
        active = time.monotonic()-started
        return (previous_phase+active >= args.max_wall_seconds-20
            or (args.deadline_unix is not None and time.time() >= args.deadline_unix-20))
    update_summary()
    write_json(output/"summary.json", summary)
    try:
        train_loader = make_loader(runtime,dataset,train_indices,args.microbatch,args.workers,
            args.seed,False,args.prefetch,args.loader_timeout,sampler=sampler)
        validation_loader = make_loader(runtime,dataset,validation_indices,args.microbatch,0,
            args.seed+1,False)
        iterator = iter(train_loader)
        policy.train()
        while step < args.max_steps and not budget_exhausted() and not stop_requested():
            raw_group = list(itertools.islice(iterator,args.accumulation))
            if not raw_group:
                epoch += 1
                batches_in_epoch = 0
                sampler.epoch, sampler.offset = epoch, 0
                iterator = iter(train_loader)
                continue
            batches_in_epoch += len(raw_group)
            metrics,count,successful = optimize(runtime,policy,raw_group,preprocessor,
                args.device,augmentation,optimizer,scaler,config.use_amp)
            samples += count
            if not successful:
                skipped += 1
                consecutive_skipped += 1
                if consecutive_skipped > 50: raise RuntimeError("More than 50 consecutive AMP skipped updates")
                if stop_requested() or budget_exhausted(): break
                continue
            consecutive_skipped = 0
            ema.update(policy,torch)
            scheduler.step()
            step += 1
            summary["train"] = metrics
            update_summary()
            if step == 1 or step % args.log_every == 0:
                write_json(output/"summary.json",summary)
                print(json.dumps({"optimizer_step":step,"step":step,"elapsed_seconds":summary["elapsed_seconds"],
                    "samples_per_second":summary["samples_per_second"],"loss":metrics["loss"],
                    "l1_loss":metrics["l1_loss"],"kld_loss":metrics["kld_loss"],
                    "learning_rates":summary["learning_rates"],"main_lr":summary["learning_rates"][0],
                    "backbone_lr":summary["learning_rates"][1],"skipped_amp_updates":skipped,
                    "ema_updates":ema.updates,"consumed_samples":samples,"scaler_scale":scaler.get_scale()}),flush=True)
            # STOP means finish this optimizer update, then flush state/snapshots below.
            if stop_requested() or budget_exhausted(): break
            should_validate = step % args.validation_every == 0 or step == args.max_steps
            if should_validate:
                remaining = args.max_wall_seconds-summary["phase_training_seconds"]
                if remaining > 90 and (args.deadline_unix is None or args.deadline_unix-time.time() > 90):
                    validation_loader.deadline = time.monotonic()+remaining-20
                    if args.deadline_unix: validation_loader.deadline=min(validation_loader.deadline,
                        time.monotonic()+args.deadline_unix-time.time()-20)
                    latest_metrics = validate(runtime,ema.policy,preprocessor,postprocessor,
                        validation_loader,args.device,args.max_validation_batches,posterior_loss=False)
                    latest_metrics["complete_planned_validation"] = (
                        latest_metrics["observations"] == len(validation_indices) and args.max_validation_batches is None)
                    record = {"optimizer_step":step,**latest_metrics}
                    summary["validation"].append(record)
                    directory=output/"validation";directory.mkdir(exist_ok=True)
                    path=directory/f"step{step:06d}.json"
                    if path.exists(): path=directory/f"step{step:06d}_{time.time_ns()}.json"
                    write_json(path,record)
                    print(json.dumps({"validation_written":{"optimizer_step":step,"path":str(path),
                        "metrics":latest_metrics}}),flush=True)
                    policy.train()
            if step % args.snapshot_every == 0 or step == args.max_steps: snapshot()
            if step % args.resume_every == 0: resume_state("periodic")
            update_summary();write_json(output/"summary.json",summary)
        reason = ("stop" if stop_requested() else "complete" if step>=args.max_steps else "budget")
        if last_snapshot_step != step: snapshot()
        summary.update(status="complete" if step>=args.max_steps else "stopped_at_budget_or_signal",
                       stop_reason=reason,last_snapshot=last_snapshot)
        resume_state(reason)
        update_summary();write_json(output/"summary.json",summary)
        print(json.dumps({"training_finished":{"optimizer_step":step,"reason":reason,
            "phase":phase,"learning_rates":summary["learning_rates"],"last_resume":summary["last_resume"],
            "elapsed_seconds":summary["elapsed_seconds"]}}),flush=True)
    except Exception as error:
        update_summary()
        summary.update(status="failed",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        summary["loader_cleanup"]={"train":close_loader(train_loader),
                                   "validation":close_loader(validation_loader)}
        update_summary()
        write_json(output/"summary.json",summary)
        (output/"NOTE.md").write_text(f"ACT run: {summary['status']}; step {step}; phase {phase}; "
            f"total active seconds {summary['total_training_seconds']:.1f}. "
            f"Latest complete resume: {summary.get('last_resume')}. All raw/EMA weight snapshots retained.\n")


def main():
    global _STOP
    import contextlib, io, sys, traceback
    from act_resume_state import load_resume, restore_arguments
    started=time.monotonic()
    argument_parser=parser()
    args=argument_parser.parse_args()
    provided=set()
    for token in sys.argv[1:]:
        option=token.split("=",1)[0]
        if option in argument_parser._option_string_actions:
            provided.add(argument_parser._option_string_actions[option].dest)
    def stop(signum,frame):
        global _STOP
        _STOP=True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    if args.resume:
        if args.init_checkpoint: raise ValueError("--resume and --init-checkpoint are exclusive")
        import torch
        args._resume_payload=load_resume(args.resume,torch)
        restore_arguments(args,provided,args._resume_payload)
        saved_output=Path(args._resume_payload["arguments"]["output"]).resolve()
        if args.output is None: args.output=saved_output
        elif args.output.resolve()!=saved_output:
            raise ValueError("Full resume must reuse its saved output run")
    elif args.decay_steps is not None:
        raise ValueError("--decay-steps requires full --resume")
    if args.output is None: raise ValueError("--output is required for a fresh run")
    if min(args.max_steps,args.microbatch,args.accumulation,args.snapshot_every,
        args.resume_every,args.validation_every,args.prefetch,args.validation_frames_per_episode,args.log_every)<1:
        raise ValueError("Step/batch/snapshot/resume/validation/log settings must be positive")
    if args.decay_steps is not None and args.decay_steps<1: raise ValueError("Decay steps must be positive")
    if args.workers<0 or not 0<=args.ema_decay<1: raise ValueError("Invalid workers or EMA decay")
    if args.schedule_steps is not None and args.schedule_steps<args.max_steps and args.lr_schedule!="constant":
        raise ValueError("Cosine schedule cannot be shorter than max-steps")
    output=args.output.resolve()
    if output.exists():
        if not args.resume or not (output/"summary.json").exists():
            raise ValueError("Existing output allowed only for full resume of a training run")
        prior=read_json(output/"summary.json")
        if prior.get("kind")!="act_v2_training": raise ValueError("Output is not an ACT training run")
    else: output.mkdir(parents=True)
    # Legacy policy constructors print human text; keep our trainer stdout JSON lines.
    captured=io.StringIO()
    with contextlib.redirect_stdout(captured): initialized=initialize(args)
    for message in captured.getvalue().splitlines():
        if message: print(json.dumps({"initialization_message":message}),flush=True)
    if args.benchmark: benchmark(args,initialized,started)
    else: train(args,initialized,started)

if __name__=="__main__":
    import traceback
    try: main()
    except Exception as error:
        print(json.dumps({"fatal_error":f"{type(error).__name__}: {error}",
                          "traceback":traceback.format_exc()}),flush=True)
        raise SystemExit(1)

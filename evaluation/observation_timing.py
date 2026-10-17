"""Track timestamps of actual buffered observations without copying image data."""
from collections import Counter, deque
import functools
import math
import statistics
import threading
import weakref


def finite_number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def timing_at_use(snapshot, now):
    """Recompute source ages when the policy consumes a captured observation."""
    if not snapshot:
        return {"association": "unavailable", "used_timestamps": {}, "ages_sec": {}}
    result = {**snapshot}
    result["used_timestamps"] = dict(snapshot["used_timestamps"])
    result["use_ros_time"] = finite_number(now)
    result["ages_sec"] = {
        key: result["use_ros_time"] - timestamp
        for key, timestamp in result["used_timestamps"].items()
        if timestamp is not None and result["use_ros_time"] is not None
    }
    timestamps = [v for v in result["used_timestamps"].values() if v is not None]
    result["cross_topic_skew_sec"] = max(timestamps) - min(timestamps) if timestamps else None
    return result


def numeric_summary(values):
    values = sorted(x for x in values if x is not None and math.isfinite(x))
    if not values:
        return {"count": 0}
    position = (len(values) - 1) * .95
    lower, upper = math.floor(position), math.ceil(position)
    p95 = values[lower] + (values[upper] - values[lower]) * (position - lower)
    return {"count": len(values), "min": values[0], "median": statistics.median(values),
            "p95": p95, "max": values[-1], "negative_count": sum(x < 0 for x in values)}


def summarize_timing(samples):
    keys = sorted({key for sample in samples for key in sample.get("used_timestamps", {})})
    return {
        "sample_count": len(samples),
        "fully_associated_count": sum(sample.get("association") == "exact_object_identity_at_append" for sample in samples),
        "source_ages_sec": {key: numeric_summary([sample.get("ages_sec", {}).get(key) for sample in samples]) for key in keys},
        "cross_topic_skew_sec": numeric_summary([sample.get("cross_topic_skew_sec") for sample in samples]),
        "method": "Source timestamps match actual data objects returned by get_latest_obs; ages use ROS time at policy consumption. Missing associations are reported, never replaced by latest-buffer timestamps.",
    }


class ObservationTimingMonitor:
    """Install reversible in-memory hooks on the existing ObsBuffer class.

    The append hook stores references to the original objects and their timestamps.
    Matching by object identity remains correct when another callback appends a
    newer message between get_latest_obs() returning and the timestamp lookup.
    Cache misses are explicit. Original return values and exceptions are preserved.
    """
    def __init__(self, buffer_class, ros_now):
        self.buffer_class = buffer_class
        self.ros_now = ros_now
        self.history = weakref.WeakKeyDictionary()
        self.latest = weakref.WeakKeyDictionary()
        self.lock = threading.RLock()
        self.original_append = buffer_class._append_data
        self.original_latest = buffer_class.get_latest_obs
        self.errors = Counter()
        monitor = self

        @functools.wraps(self.original_append)
        def append(instance, key, data, timestamp):
            try:
                monitor.record(instance, key, data, timestamp)
            except Exception as error:
                monitor.errors[type(error).__name__] += 1
            return monitor.original_append(instance, key, data, timestamp)

        @functools.wraps(self.original_latest)
        def latest(instance, *args, **kwargs):
            observation = monitor.original_latest(instance, *args, **kwargs)
            try:
                monitor.capture(instance, observation)
            except Exception as error:
                monitor.errors[type(error).__name__] += 1
            return observation

        self.append_hook, self.latest_hook = append, latest
        buffer_class._append_data = append
        buffer_class.get_latest_obs = latest

    def record(self, instance, key, data, timestamp):
        with self.lock:
            histories = self.history.setdefault(instance, {})
            if key not in histories:
                buffer = instance.obs_buffer_data[key]["data"]
                histories[key] = deque(maxlen=(buffer.maxlen or 512) + 1)
            histories[key].append((data, finite_number(timestamp)))

    def capture(self, instance, observation):
        with self.lock:
            histories = self.history.get(instance, {})
            timestamps, unmatched = {}, []
            for key, actual in observation.items():
                found = False
                for recorded, timestamp in reversed(histories.get(key, ())):
                    if recorded is actual:
                        timestamps[key], found = timestamp, True
                        break
                if not found or timestamps[key] is None:
                    timestamps[key] = None
                    unmatched.append(key)
            self.latest[instance] = {
                "association": "exact_object_identity_at_append" if not unmatched else "missing_object_timestamp",
                "used_timestamps": timestamps, "unmatched_keys": unmatched,
                "capture_ros_time": finite_number(self.ros_now()),
            }

    def snapshot(self, instance, now):
        with self.lock:
            snapshot = self.latest.get(instance)
            return timing_at_use(snapshot, now)

    def close(self):
        # Preserve an external replacement instead of overwriting it on cleanup.
        if self.buffer_class._append_data is self.append_hook:
            self.buffer_class._append_data = self.original_append
        if self.buffer_class.get_latest_obs is self.latest_hook:
            self.buffer_class.get_latest_obs = self.original_latest

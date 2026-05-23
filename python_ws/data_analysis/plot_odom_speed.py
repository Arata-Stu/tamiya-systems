#!/usr/bin/env python3
"""Plot speed timeseries from nav_msgs/Odometry messages in rosbag files."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from rosbags.highlevel import AnyReader


CSV_FIELDS = [
    "source_bag",
    "sample_index",
    "time_from_start_sec",
    "odom_header_stamp_ns",
    "odom_recorded_stamp_ns",
    "vx_mps",
    "vy_mps",
    "vz_mps",
    "speed_mps",
    "speed_3d_mps",
    "yaw_rate_radps",
]


@dataclass(frozen=True)
class OdomSpeedSample:
    source_bag: str
    sample_index: int
    time_from_start_sec: float
    header_stamp_ns: int
    recorded_stamp_ns: int
    vx_mps: float
    vy_mps: float
    vz_mps: float
    speed_mps: float
    speed_3d_mps: float
    yaw_rate_radps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bag",
        required=True,
        nargs="+",
        action="append",
        metavar="PATH",
        help="Path(s) to rosbag2 directories or metadata.yaml files. Can be repeated.",
    )
    parser.add_argument("--odom-topic", default="/visual_slam/tracking/odometry")
    parser.add_argument("--csv", default="", help="Output CSV path. Default: /tmp/<bag>_odom_speed.csv")
    parser.add_argument("--plot", default="", help="Output PNG path. Default: /tmp/<bag>_odom_speed.png")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Output summary JSON path. Default: next to CSV.",
    )
    parser.add_argument(
        "--max-samples-per-bag",
        type=int,
        default=0,
        help="Optional cap for quick inspection. 0 means no cap.",
    )
    parser.add_argument(
        "--use-recorded-time",
        action="store_true",
        help="Use rosbag recorded timestamps for x-axis instead of odometry header stamps.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip PNG generation.")
    return parser.parse_args()


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


def normalize_bag_paths(raw_bags: list[list[str]]) -> list[Path]:
    bag_paths: list[Path] = []
    seen: set[Path] = set()
    for group in raw_bags:
        for path_str in group:
            bag_path = normalize_bag_path(path_str)
            if bag_path not in seen:
                bag_paths.append(bag_path)
                seen.add(bag_path)
    return bag_paths


def default_stem_for_bags(bag_paths: list[Path]) -> str:
    if len(bag_paths) == 1:
        return bag_paths[0].name if bag_paths[0].is_dir() else bag_paths[0].stem
    return f"combined_{len(bag_paths)}bags"


def _build_default_ros2_typestore():
    try:
        from rosbags.typesys import Stores, get_typestore
    except Exception:
        return None

    preferred_store_names = (
        "ROS2_JAZZY",
        "ROS2_IRON",
        "ROS2_HUMBLE",
        "ROS2_GALACTIC",
        "ROS2_FOXY",
    )
    for store_name in preferred_store_names:
        if hasattr(Stores, store_name):
            return get_typestore(getattr(Stores, store_name))

    for store in Stores:
        if store.name.startswith("ROS2_"):
            return get_typestore(store)

    return None


def _open_reader(bag_path: Path) -> AnyReader:
    reader_kwargs: dict[str, Any] = {}
    default_typestore = _build_default_ros2_typestore()

    try:
        supports_default_typestore = "default_typestore" in inspect.signature(AnyReader).parameters
    except (TypeError, ValueError):
        supports_default_typestore = False

    if supports_default_typestore and default_typestore is not None:
        reader_kwargs["default_typestore"] = default_typestore

    return AnyReader([bag_path], **reader_kwargs)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def load_odom_speed_samples(
    bag_path: Path,
    topic: str,
    use_recorded_time: bool,
    max_samples: int,
) -> list[OdomSpeedSample]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    with _open_reader(bag_path) as reader:
        connections = [conn for conn in reader.connections if conn.topic == topic]
        if not connections:
            raise RuntimeError(f"Topic not found in bag: topic={topic} path={bag_path}")

        rows: list[OdomSpeedSample] = []
        first_time_ns: int | None = None
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            header_stamp_ns = stamp_to_ns(msg.header.stamp)
            recorded_stamp_ns = int(timestamp)
            axis_stamp_ns = recorded_stamp_ns if use_recorded_time else header_stamp_ns
            if first_time_ns is None:
                first_time_ns = axis_stamp_ns
            twist = msg.twist.twist
            vx = float(twist.linear.x)
            vy = float(twist.linear.y)
            vz = float(twist.linear.z)
            speed_2d = math.hypot(vx, vy)
            speed_3d = math.sqrt(vx * vx + vy * vy + vz * vz)
            rows.append(
                OdomSpeedSample(
                    source_bag=str(bag_path),
                    sample_index=len(rows),
                    time_from_start_sec=(axis_stamp_ns - first_time_ns) / 1.0e9,
                    header_stamp_ns=header_stamp_ns,
                    recorded_stamp_ns=recorded_stamp_ns,
                    vx_mps=vx,
                    vy_mps=vy,
                    vz_mps=vz,
                    speed_mps=speed_2d,
                    speed_3d_mps=speed_3d,
                    yaw_rate_radps=float(twist.angular.z),
                )
            )
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return math.nan
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = min(max(q, 0.0), 1.0)
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def summarize(samples: list[OdomSpeedSample]) -> dict[str, Any]:
    if not samples:
        raise RuntimeError("No odometry samples loaded.")
    speeds = sorted(sample.speed_mps for sample in samples)
    yaw_rates = sorted(abs(sample.yaw_rate_radps) for sample in samples)
    duration = max(sample.time_from_start_sec for sample in samples)
    return {
        "samples": len(samples),
        "duration_sec": duration,
        "mean_rate_hz": (len(samples) - 1) / duration if duration > 0.0 and len(samples) > 1 else 0.0,
        "speed_mps": {
            "min": speeds[0],
            "mean": mean(speeds),
            "median": median(speeds),
            "p90": percentile(speeds, 0.90),
            "p95": percentile(speeds, 0.95),
            "p99": percentile(speeds, 0.99),
            "max": speeds[-1],
        },
        "abs_yaw_rate_radps": {
            "median": median(yaw_rates),
            "p95": percentile(yaw_rates, 0.95),
            "max": yaw_rates[-1],
        },
    }


def summarize_by_bag(samples: list[OdomSpeedSample]) -> dict[str, dict[str, Any]]:
    by_bag: dict[str, list[OdomSpeedSample]] = {}
    for sample in samples:
        by_bag.setdefault(sample.source_bag, []).append(sample)
    return {source_bag: summarize(rows) for source_bag, rows in by_bag.items()}


def summarize_combined(samples: list[OdomSpeedSample]) -> dict[str, Any]:
    by_bag_summary = summarize_by_bag(samples)
    combined = summarize(samples)
    combined["duration_sec"] = sum(item["duration_sec"] for item in by_bag_summary.values())
    combined["mean_rate_hz"] = (
        (combined["samples"] - len(by_bag_summary)) / combined["duration_sec"]
        if combined["duration_sec"] > 0.0 and combined["samples"] > len(by_bag_summary)
        else 0.0
    )
    return combined


def write_csv(path: Path, samples: list[OdomSpeedSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for sample in samples:
            writer.writerow(
                {
                    "source_bag": sample.source_bag,
                    "sample_index": sample.sample_index,
                    "time_from_start_sec": f"{sample.time_from_start_sec:.9f}",
                    "odom_header_stamp_ns": sample.header_stamp_ns,
                    "odom_recorded_stamp_ns": sample.recorded_stamp_ns,
                    "vx_mps": f"{sample.vx_mps:.9g}",
                    "vy_mps": f"{sample.vy_mps:.9g}",
                    "vz_mps": f"{sample.vz_mps:.9g}",
                    "speed_mps": f"{sample.speed_mps:.9g}",
                    "speed_3d_mps": f"{sample.speed_3d_mps:.9g}",
                    "yaw_rate_radps": f"{sample.yaw_rate_radps:.9g}",
                }
            )


def write_summary_json(path: Path, samples: list[OdomSpeedSample]) -> dict[str, Any]:
    by_bag = summarize_by_bag(samples)
    summary = {
        "combined": summarize_combined(samples),
        "by_bag": by_bag,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def plot_speed(path: Path, samples: list[OdomSpeedSample], summary: dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plot output: {exc}") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    by_bag: dict[str, list[OdomSpeedSample]] = {}
    for sample in samples:
        by_bag.setdefault(sample.source_bag, []).append(sample)

    fig, (ax_speed, ax_hist) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    for source_bag, bag_samples in by_bag.items():
        label = Path(source_bag).name
        ax_speed.plot(
            [sample.time_from_start_sec for sample in bag_samples],
            [sample.speed_mps for sample in bag_samples],
            linewidth=1.0,
            alpha=0.85,
            label=label,
        )

    combined = summary["combined"]["speed_mps"]
    ax_speed.axhline(combined["p95"], color="tab:orange", linestyle="--", linewidth=1.0, label="p95")
    ax_speed.axhline(combined["max"], color="tab:red", linestyle=":", linewidth=1.0, label="max")
    ax_speed.set_title(
        "VSLAM odometry speed "
        f"(max={combined['max']:.3f} m/s, p95={combined['p95']:.3f} m/s, "
        f"median={combined['median']:.3f} m/s)"
    )
    ax_speed.set_xlabel("time from start [s]")
    ax_speed.set_ylabel("planar speed [m/s]")
    ax_speed.grid(True, alpha=0.3)
    ax_speed.legend(loc="best", fontsize="small")

    speeds = [sample.speed_mps for sample in samples]
    ax_hist.hist(speeds, bins=40, color="tab:blue", alpha=0.8)
    ax_hist.set_xlabel("planar speed [m/s]")
    ax_hist.set_ylabel("samples")
    ax_hist.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def print_summary(summary: dict[str, Any], csv_path: Path, plot_path: Path | None, summary_path: Path) -> None:
    speed = summary["combined"]["speed_mps"]
    combined = summary["combined"]
    print(f"[INFO] samples={combined['samples']} duration={combined['duration_sec']:.3f}s rate={combined['mean_rate_hz']:.2f}Hz")
    print(
        "[INFO] speed_mps "
        f"median={speed['median']:.3f} p90={speed['p90']:.3f} "
        f"p95={speed['p95']:.3f} p99={speed['p99']:.3f} max={speed['max']:.3f}"
    )
    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] summary_json={summary_path}")
    if plot_path is not None:
        print(f"[INFO] plot={plot_path}")


def main() -> int:
    args = parse_args()
    bag_paths = normalize_bag_paths(args.bag)
    if not bag_paths:
        raise RuntimeError("At least one --bag path is required.")

    stem = default_stem_for_bags(bag_paths)
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else Path("/tmp") / f"{stem}_odom_speed.csv"
    plot_path = None if args.no_plot else (
        Path(args.plot).expanduser().resolve() if args.plot else Path("/tmp") / f"{stem}_odom_speed.png"
    )
    summary_path = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else csv_path.with_name(csv_path.stem + "_summary.json")
    )

    samples: list[OdomSpeedSample] = []
    for bag_path in bag_paths:
        bag_samples = load_odom_speed_samples(
            bag_path,
            args.odom_topic,
            args.use_recorded_time,
            max(0, int(args.max_samples_per_bag)),
        )
        print(f"[INFO] bag={bag_path} odom_samples={len(bag_samples)}")
        samples.extend(bag_samples)

    if not samples:
        raise RuntimeError(f"No odometry samples found on {args.odom_topic}.")

    write_csv(csv_path, samples)
    summary = write_summary_json(summary_path, samples)
    if plot_path is not None:
        plot_speed(plot_path, samples, summary)
    print_summary(summary, csv_path, plot_path, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

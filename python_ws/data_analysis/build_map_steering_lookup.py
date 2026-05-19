#!/usr/bin/env python3
"""Build a MAP-style steering lookup table from rosbag odometry and drive commands."""

from __future__ import annotations

import argparse
import csv
import inspect
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


RAW_CSV_FIELDS = [
    "odom_recorded_stamp_ns",
    "odom_header_stamp_ns",
    "cmd_recorded_stamp_ns",
    "cmd_header_stamp_ns",
    "speed_mps",
    "yaw_rate_radps",
    "lateral_accel_mps2",
    "steering_angle_rad",
    "command_speed_mps",
]


@dataclass(frozen=True)
class OdomSample:
    recorded_stamp_ns: int
    header_stamp_ns: int
    speed_mps: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class DriveSample:
    recorded_stamp_ns: int
    header_stamp_ns: int
    steering_angle_rad: float
    speed_mps: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a MAP-compatible steering lookup table from rosbag odometry "
            "and Ackermann drive commands."
        )
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to rosbag2 directory or metadata.yaml",
    )
    parser.add_argument(
        "--odom-topic",
        default="/visual_slam/tracking/odometry",
        help="Odometry topic used for speed and yaw-rate estimation.",
    )
    parser.add_argument(
        "--cmd-topic",
        default="/jetracer/cmd_drive",
        help="Ackermann command topic. Default is the final command actually sent to the vehicle.",
    )
    parser.add_argument(
        "--lookup-csv",
        default=None,
        help="Output CSV for the lookup table. Defaults to /tmp/<bag>_map_lookup_table.csv",
    )
    parser.add_argument(
        "--counts-csv",
        default=None,
        help="Optional CSV for per-bin sample counts. Defaults next to --lookup-csv",
    )
    parser.add_argument(
        "--raw-csv",
        default=None,
        help="Optional CSV for matched raw samples. Defaults next to --lookup-csv",
    )
    parser.add_argument(
        "--command-delay-sec",
        type=float,
        default=0.0,
        help="Apply this response delay when pairing odometry with prior commands.",
    )
    parser.add_argument(
        "--min-speed",
        type=float,
        default=1.0,
        help="Discard odometry samples below this speed.",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=None,
        help="Upper speed bound for matched samples and lookup columns. Default: infer from data.",
    )
    parser.add_argument(
        "--speed-bin-size",
        type=float,
        default=0.25,
        help="Speed bin size [m/s] used for lookup columns.",
    )
    parser.add_argument(
        "--min-abs-steer",
        type=float,
        default=0.01,
        help="Discard samples whose absolute steering angle is below this threshold.",
    )
    parser.add_argument(
        "--max-abs-steer",
        type=float,
        default=None,
        help="Upper steering bound [rad] for matched samples and lookup rows. Default: infer from data.",
    )
    parser.add_argument(
        "--steer-bin-size",
        type=float,
        default=0.01,
        help="Steering bin size [rad] used for lookup rows.",
    )
    parser.add_argument(
        "--max-abs-yaw-rate",
        type=float,
        default=8.0,
        help="Discard samples whose absolute yaw rate exceeds this threshold.",
    )
    parser.add_argument(
        "--max-abs-lateral-accel",
        type=float,
        default=20.0,
        help="Discard samples whose absolute lateral acceleration exceeds this threshold.",
    )
    parser.add_argument(
        "--min-samples-per-bin",
        type=int,
        default=5,
        help="Minimum raw samples required before a bin is treated as directly observed.",
    )
    return parser.parse_args()


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


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
    reader_kwargs = {}
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


def read_connections(reader: AnyReader, topic: str):
    connections = [conn for conn in reader.connections if conn.topic == topic]
    if not connections:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    return connections


def load_odom_samples(bag_path: Path, topic: str) -> list[OdomSample]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    samples: list[OdomSample] = []
    with _open_reader(bag_path) as reader:
        connections = read_connections(reader, topic)
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            twist = msg.twist.twist
            speed_mps = math.hypot(float(twist.linear.x), float(twist.linear.y))
            samples.append(
                OdomSample(
                    recorded_stamp_ns=int(timestamp),
                    header_stamp_ns=stamp_to_ns(msg.header.stamp),
                    speed_mps=speed_mps,
                    yaw_rate_radps=float(twist.angular.z),
                )
            )
    return samples


def load_drive_samples(bag_path: Path, topic: str) -> list[DriveSample]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    samples: list[DriveSample] = []
    with _open_reader(bag_path) as reader:
        connections = read_connections(reader, topic)
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            drive = getattr(msg, "drive", msg)
            header = getattr(msg, "header", None)
            header_stamp_ns = stamp_to_ns(header.stamp) if header is not None else int(timestamp)
            samples.append(
                DriveSample(
                    recorded_stamp_ns=int(timestamp),
                    header_stamp_ns=header_stamp_ns,
                    steering_angle_rad=float(drive.steering_angle),
                    speed_mps=float(getattr(drive, "speed", 0.0)),
                )
            )
    return samples


def default_output_path(bag_path: Path, suffix: str) -> Path:
    stem = bag_path.name if bag_path.is_dir() else bag_path.stem
    return Path("/tmp") / f"{stem}{suffix}"


def find_latest_drive_index(drive_times_ns: list[int], target_ns: int) -> int:
    return bisect_right(drive_times_ns, target_ns) - 1


def build_raw_rows(
    odom_samples: list[OdomSample],
    drive_samples: list[DriveSample],
    command_delay_sec: float,
    min_speed: float,
    max_speed: float | None,
    min_abs_steer: float,
    max_abs_steer: float | None,
    max_abs_yaw_rate: float,
    max_abs_lateral_accel: float,
) -> list[dict[str, float | int]]:
    if not odom_samples:
        raise RuntimeError("No odometry samples found.")
    if not drive_samples:
        raise RuntimeError("No drive command samples found.")

    drive_times_ns = [sample.recorded_stamp_ns for sample in drive_samples]
    delay_ns = int(round(command_delay_sec * 1_000_000_000.0))
    rows: list[dict[str, float | int]] = []

    for odom in odom_samples:
        if not math.isfinite(odom.speed_mps) or odom.speed_mps < min_speed:
            continue
        if max_speed is not None and odom.speed_mps > max_speed:
            continue
        if not math.isfinite(odom.yaw_rate_radps) or abs(odom.yaw_rate_radps) > max_abs_yaw_rate:
            continue

        target_cmd_time_ns = odom.recorded_stamp_ns - delay_ns
        drive_index = find_latest_drive_index(drive_times_ns, target_cmd_time_ns)
        if drive_index < 0:
            continue

        drive = drive_samples[drive_index]
        abs_steer = abs(drive.steering_angle_rad)
        if not math.isfinite(abs_steer) or abs_steer < min_abs_steer:
            continue
        if max_abs_steer is not None and abs_steer > max_abs_steer:
            continue

        lateral_accel_mps2 = odom.speed_mps * odom.yaw_rate_radps
        if not math.isfinite(lateral_accel_mps2) or abs(lateral_accel_mps2) > max_abs_lateral_accel:
            continue

        rows.append(
            {
                "odom_recorded_stamp_ns": odom.recorded_stamp_ns,
                "odom_header_stamp_ns": odom.header_stamp_ns,
                "cmd_recorded_stamp_ns": drive.recorded_stamp_ns,
                "cmd_header_stamp_ns": drive.header_stamp_ns,
                "speed_mps": odom.speed_mps,
                "yaw_rate_radps": odom.yaw_rate_radps,
                "lateral_accel_mps2": lateral_accel_mps2,
                "steering_angle_rad": drive.steering_angle_rad,
                "command_speed_mps": drive.speed_mps,
            }
        )

    if not rows:
        raise RuntimeError(
            "No matched samples remained after filtering. Try lowering --min-speed, "
            "widening steering limits, or reducing --command-delay-sec."
        )

    return rows


def infer_max_value(rows: list[dict[str, float | int]], key: str, quantile: float) -> float:
    values = np.asarray([abs(float(row[key])) for row in rows], dtype=float)
    if values.size == 0:
        raise RuntimeError(f"No values available to infer {key}.")
    return float(np.quantile(values, quantile))


def build_centers(start: float, stop: float, step: float) -> np.ndarray:
    if step <= 0.0:
        raise ValueError("Bin step must be positive.")
    if stop < start:
        stop = start
    count = int(math.floor((stop - start) / step + 0.5))
    centers = start + np.arange(count + 1, dtype=float) * step
    if centers[-1] < stop - 1.0e-9:
        centers = np.append(centers, stop)
    return centers


def nearest_center_index(value: float, centers: np.ndarray) -> int:
    return int(np.argmin(np.abs(centers - value)))


def interpolate_column(column: np.ndarray, steer_bins: np.ndarray) -> np.ndarray:
    result = column.copy()
    result[0] = 0.0
    valid = np.flatnonzero(np.isfinite(result))
    if valid.size == 0:
        return result
    if valid[0] != 0:
        valid = np.insert(valid, 0, 0)
        result[0] = 0.0
    interpolated = np.interp(
        steer_bins,
        steer_bins[valid],
        result[valid],
        left=0.0,
        right=float(result[valid[-1]]),
    )
    return np.maximum.accumulate(np.maximum(interpolated, 0.0))


def interpolate_rows(matrix: np.ndarray, speed_bins: np.ndarray) -> np.ndarray:
    result = matrix.copy()
    for row_idx in range(result.shape[0]):
        row = result[row_idx, :]
        valid = np.flatnonzero(np.isfinite(row))
        if valid.size == 0:
            continue
        result[row_idx, :] = np.interp(
            speed_bins,
            speed_bins[valid],
            row[valid],
            left=float(row[valid[0]]),
            right=float(row[valid[-1]]),
        )
    result[0, :] = 0.0
    return result


def build_lookup(
    rows: list[dict[str, float | int]],
    speed_bins: np.ndarray,
    steer_bins: np.ndarray,
    min_samples_per_bin: int,
) -> tuple[np.ndarray, np.ndarray]:
    values: list[list[list[float]]] = [
        [[] for _ in range(len(speed_bins))]
        for _ in range(len(steer_bins))
    ]

    for row in rows:
        speed_idx = nearest_center_index(float(row["speed_mps"]), speed_bins)
        steer_idx = nearest_center_index(abs(float(row["steering_angle_rad"])), steer_bins)
        values[steer_idx][speed_idx].append(abs(float(row["lateral_accel_mps2"])))

    lookup = np.full((len(steer_bins), len(speed_bins)), np.nan, dtype=float)
    counts = np.zeros((len(steer_bins), len(speed_bins)), dtype=int)

    for steer_idx in range(len(steer_bins)):
        for speed_idx in range(len(speed_bins)):
            samples = values[steer_idx][speed_idx]
            counts[steer_idx, speed_idx] = len(samples)
            if len(samples) >= min_samples_per_bin:
                lookup[steer_idx, speed_idx] = float(np.median(samples))

    lookup[0, :] = 0.0
    counts[0, :] = np.maximum(counts[0, :], 1)

    for speed_idx in range(lookup.shape[1]):
        lookup[:, speed_idx] = interpolate_column(lookup[:, speed_idx], steer_bins)

    lookup = interpolate_rows(lookup, speed_bins)
    lookup[0, :] = 0.0
    lookup = np.maximum.accumulate(lookup, axis=0)
    return lookup, counts


def write_raw_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_lookup_csv(path: Path, speed_bins: np.ndarray, steer_bins: np.ndarray, lookup: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.zeros((len(steer_bins) + 1, len(speed_bins) + 1), dtype=float)
    matrix[0, 1:] = speed_bins
    matrix[1:, 0] = steer_bins
    matrix[1:, 1:] = lookup
    np.savetxt(path, matrix, delimiter=",", fmt="%.18e")


def write_counts_csv(path: Path, speed_bins: np.ndarray, steer_bins: np.ndarray, counts: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["steer_rad"] + [f"{value:.6f}" for value in speed_bins])
        for steer_idx, steer in enumerate(steer_bins):
            writer.writerow([f"{steer:.6f}"] + counts[steer_idx, :].astype(int).tolist())


def main() -> int:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)

    lookup_csv = Path(args.lookup_csv).expanduser().resolve() if args.lookup_csv else default_output_path(bag_path, "_map_lookup_table.csv")
    counts_csv = Path(args.counts_csv).expanduser().resolve() if args.counts_csv else lookup_csv.with_name(lookup_csv.stem + "_counts.csv")
    raw_csv = Path(args.raw_csv).expanduser().resolve() if args.raw_csv else lookup_csv.with_name(lookup_csv.stem + "_raw.csv")

    odom_samples = load_odom_samples(bag_path, args.odom_topic)
    drive_samples = load_drive_samples(bag_path, args.cmd_topic)
    rows = build_raw_rows(
        odom_samples=odom_samples,
        drive_samples=drive_samples,
        command_delay_sec=args.command_delay_sec,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        min_abs_steer=args.min_abs_steer,
        max_abs_steer=args.max_abs_steer,
        max_abs_yaw_rate=args.max_abs_yaw_rate,
        max_abs_lateral_accel=args.max_abs_lateral_accel,
    )

    max_speed = args.max_speed if args.max_speed is not None else infer_max_value(rows, "speed_mps", 0.995)
    max_abs_steer = args.max_abs_steer if args.max_abs_steer is not None else infer_max_value(rows, "steering_angle_rad", 0.995)

    speed_bins = build_centers(args.min_speed, max_speed, args.speed_bin_size)
    steer_bins = build_centers(0.0, max_abs_steer, args.steer_bin_size)
    lookup, counts = build_lookup(rows, speed_bins, steer_bins, args.min_samples_per_bin)

    write_raw_csv(raw_csv, rows)
    write_lookup_csv(lookup_csv, speed_bins, steer_bins, lookup)
    write_counts_csv(counts_csv, speed_bins, steer_bins, counts)

    observed_cells = int(np.count_nonzero(counts >= args.min_samples_per_bin))
    total_cells = int(counts.size)
    print(f"[INFO] bag={bag_path}")
    print(f"[INFO] odom_topic={args.odom_topic} cmd_topic={args.cmd_topic}")
    print(f"[INFO] matched_samples={len(rows)} speed_bins={len(speed_bins)} steer_bins={len(steer_bins)}")
    print(f"[INFO] observed_cells={observed_cells}/{total_cells} min_samples_per_bin={args.min_samples_per_bin}")
    print(f"[INFO] lookup_csv={lookup_csv}")
    print(f"[INFO] counts_csv={counts_csv}")
    print(f"[INFO] raw_csv={raw_csv}")
    print("[INFO] lookup table stores positive steering magnitudes and positive lateral accelerations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Fit speed-controller feedforward parameters from rosbag odometry and throttle commands."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader


RAW_CSV_FIELDS = [
    "source_bag",
    "odom_recorded_stamp_ns",
    "odom_header_stamp_ns",
    "cmd_recorded_stamp_ns",
    "cmd_header_stamp_ns",
    "speed_mps",
    "speed_abs_mps",
    "accel_mps2",
    "yaw_rate_radps",
    "steering_angle_rad",
    "command_throttle",
    "command_abs_throttle",
]


@dataclass(frozen=True)
class OdomSample:
    recorded_stamp_ns: int
    header_stamp_ns: int
    speed_mps: float
    accel_mps2: float
    yaw_rate_radps: float


@dataclass(frozen=True)
class DriveSample:
    recorded_stamp_ns: int
    header_stamp_ns: int
    steering_angle_rad: float
    command_throttle: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate speed_controller feedforward from rosbag odometry and final "
            "JetRacer throttle commands. Use open-loop identification bags first."
        )
    )
    parser.add_argument(
        "--bag",
        required=True,
        nargs="+",
        action="append",
        metavar="PATH",
        help="Path(s) to rosbag2 directories or metadata.yaml files. Can be repeated.",
    )
    parser.add_argument(
        "--odom-topic",
        default="/visual_slam/tracking/odometry",
        help="Odometry topic used as measured speed.",
    )
    parser.add_argument(
        "--cmd-topic",
        default="/jetracer/cmd_drive",
        help=(
            "Ackermann command topic whose drive.speed is throttle-like. "
            "For initial open-loop fitting this is usually /jetracer/cmd_drive."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=["forward", "reverse", "both"],
        default="forward",
        help="Which command direction to fit. The model is fitted on absolute values.",
    )
    parser.add_argument(
        "--command-delay-sec",
        type=float,
        default=0.20,
        help="Pair odometry at time t with the latest command at t - delay.",
    )
    parser.add_argument("--min-speed", type=float, default=0.20)
    parser.add_argument("--max-speed", type=float, default=None)
    parser.add_argument("--min-abs-command", type=float, default=0.03)
    parser.add_argument(
        "--max-abs-command",
        type=float,
        default=0.75,
        help="Discard commands above this magnitude to avoid saturated or unsafe samples.",
    )
    parser.add_argument(
        "--max-abs-steer",
        type=float,
        default=0.20,
        help="Keep mostly-straight samples for speed fitting. Use a larger value to include turns.",
    )
    parser.add_argument("--max-abs-yaw-rate", type=float, default=3.0)
    parser.add_argument(
        "--max-abs-accel",
        type=float,
        default=2.5,
        help="Filter rapidly accelerating/decelerating samples for a steadier feedforward fit.",
    )
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--force-zero-offset", action="store_true")
    parser.add_argument("--allow-negative-offset", action="store_true")
    parser.add_argument(
        "--param-yaml",
        default=None,
        help="Output ROS parameter YAML. Default: /tmp/<bag>_speed_controller_feedforward.param.yaml",
    )
    parser.add_argument(
        "--raw-csv",
        default=None,
        help="Output matched raw samples CSV. Default: next to --param-yaml.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help="Output summary JSON. Default: next to --param-yaml.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG plot. Default: next to --param-yaml.",
    )
    parser.add_argument("--no-plot", action="store_true")
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


def read_connections(reader: AnyReader, topic: str):
    connections = [conn for conn in reader.connections if conn.topic == topic]
    if not connections:
        raise RuntimeError(f"Topic not found in bag: {topic}")
    return connections


def signed_planar_speed(vx: float, vy: float) -> float:
    speed = math.hypot(vx, vy)
    if abs(vx) > 1.0e-4:
        return math.copysign(speed, vx)
    return speed


def load_odom_samples(bag_path: Path, topic: str) -> list[OdomSample]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    base_rows: list[tuple[int, int, float, float]] = []
    with _open_reader(bag_path) as reader:
        connections = read_connections(reader, topic)
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            twist = msg.twist.twist
            speed_mps = signed_planar_speed(float(twist.linear.x), float(twist.linear.y))
            base_rows.append(
                (
                    int(timestamp),
                    stamp_to_ns(msg.header.stamp),
                    speed_mps,
                    float(twist.angular.z),
                )
            )

    base_rows.sort(key=lambda row: row[0])
    samples: list[OdomSample] = []
    previous_stamp_ns: int | None = None
    previous_speed_mps: float | None = None
    for recorded_stamp_ns, header_stamp_ns, speed_mps, yaw_rate_radps in base_rows:
        accel_mps2 = math.nan
        if previous_stamp_ns is not None and previous_speed_mps is not None:
            dt = (recorded_stamp_ns - previous_stamp_ns) / 1.0e9
            if dt > 1.0e-4:
                accel_mps2 = (speed_mps - previous_speed_mps) / dt
        samples.append(
            OdomSample(
                recorded_stamp_ns=recorded_stamp_ns,
                header_stamp_ns=header_stamp_ns,
                speed_mps=speed_mps,
                accel_mps2=accel_mps2,
                yaw_rate_radps=yaw_rate_radps,
            )
        )
        previous_stamp_ns = recorded_stamp_ns
        previous_speed_mps = speed_mps
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
                    command_throttle=float(getattr(drive, "speed", 0.0)),
                )
            )
    samples.sort(key=lambda sample: sample.recorded_stamp_ns)
    return samples


def find_latest_drive_index(drive_times_ns: list[int], target_ns: int) -> int:
    return bisect_right(drive_times_ns, target_ns) - 1


def direction_ok(command: float, direction: str, min_abs_command: float) -> bool:
    if direction == "forward":
        return command >= min_abs_command
    if direction == "reverse":
        return command <= -min_abs_command
    return abs(command) >= min_abs_command


def build_raw_rows(
    odom_samples: list[OdomSample],
    drive_samples: list[DriveSample],
    args: argparse.Namespace,
    source_bag: str,
    require_rows: bool,
) -> list[dict[str, float | int | str]]:
    if not odom_samples:
        raise RuntimeError("No odometry samples found.")
    if not drive_samples:
        raise RuntimeError("No drive command samples found.")

    drive_times_ns = [sample.recorded_stamp_ns for sample in drive_samples]
    delay_ns = int(round(args.command_delay_sec * 1_000_000_000.0))
    rows: list[dict[str, float | int | str]] = []

    for odom in odom_samples:
        speed_abs = abs(odom.speed_mps)
        if not math.isfinite(speed_abs) or speed_abs < args.min_speed:
            continue
        if args.max_speed is not None and speed_abs > args.max_speed:
            continue
        if math.isfinite(odom.accel_mps2) and abs(odom.accel_mps2) > args.max_abs_accel:
            continue
        if not math.isfinite(odom.yaw_rate_radps) or abs(odom.yaw_rate_radps) > args.max_abs_yaw_rate:
            continue

        drive_index = find_latest_drive_index(drive_times_ns, odom.recorded_stamp_ns - delay_ns)
        if drive_index < 0:
            continue

        drive = drive_samples[drive_index]
        command_abs = abs(drive.command_throttle)
        if not direction_ok(drive.command_throttle, args.direction, args.min_abs_command):
            continue
        if command_abs < args.min_abs_command:
            continue
        if args.max_abs_command is not None and command_abs > args.max_abs_command:
            continue
        if args.max_abs_steer is not None and abs(drive.steering_angle_rad) > args.max_abs_steer:
            continue

        rows.append(
            {
                "source_bag": source_bag,
                "odom_recorded_stamp_ns": odom.recorded_stamp_ns,
                "odom_header_stamp_ns": odom.header_stamp_ns,
                "cmd_recorded_stamp_ns": drive.recorded_stamp_ns,
                "cmd_header_stamp_ns": drive.header_stamp_ns,
                "speed_mps": odom.speed_mps,
                "speed_abs_mps": speed_abs,
                "accel_mps2": odom.accel_mps2,
                "yaw_rate_radps": odom.yaw_rate_radps,
                "steering_angle_rad": drive.steering_angle_rad,
                "command_throttle": drive.command_throttle,
                "command_abs_throttle": command_abs,
            }
        )

    if require_rows and not rows:
        raise RuntimeError(
            "No matched samples remained after filtering. Try lowering --min-speed, "
            "raising --max-abs-steer, raising --max-abs-accel, or changing --command-delay-sec."
        )
    return rows


def fit_feedforward(
    rows: list[dict[str, float | int | str]],
    force_zero_offset: bool,
    allow_negative_offset: bool,
) -> dict[str, float]:
    x = np.asarray([float(row["speed_abs_mps"]) for row in rows], dtype=float)
    y = np.asarray([float(row["command_abs_throttle"]) for row in rows], dtype=float)
    if x.size < 2:
        raise RuntimeError("At least two samples are required for feedforward fitting.")

    if force_zero_offset:
        denom = float(np.dot(x, x))
        if denom <= 1.0e-12:
            raise RuntimeError("Cannot fit zero-offset model because speed samples are too small.")
        gain = float(np.dot(x, y) / denom)
        offset = 0.0
    else:
        matrix = np.vstack([x, np.ones_like(x)]).T
        gain, offset = np.linalg.lstsq(matrix, y, rcond=None)[0]
        gain = float(gain)
        offset = float(offset)
        if offset < 0.0 and not allow_negative_offset:
            offset = 0.0
            gain = float(np.dot(x, y) / max(float(np.dot(x, x)), 1.0e-12))

    predicted = gain * x + offset
    residual = y - predicted
    ss_res = float(np.dot(residual, residual))
    centered = y - float(np.mean(y))
    ss_tot = float(np.dot(centered, centered))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-12 else 1.0
    rmse = math.sqrt(ss_res / max(1, x.size))
    return {
        "feedforward_gain": gain,
        "feedforward_offset": offset,
        "r2": r2,
        "rmse_throttle": rmse,
        "samples": int(x.size),
        "speed_min_mps": float(np.min(x)),
        "speed_max_mps": float(np.max(x)),
        "command_min": float(np.min(y)),
        "command_max": float(np.max(y)),
    }


def write_raw_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_param_yaml(path: Path, fit: dict[str, float], direction: str) -> None:
    gain = fit["feedforward_gain"]
    offset = fit["feedforward_offset"]
    if direction == "reverse":
        body = (
            "/**:\n"
            "  ros__parameters:\n"
            f"    reverse_feedforward_gain: {gain:.9g}\n"
            f"    reverse_feedforward_offset: {offset:.9g}\n"
        )
    elif direction == "forward":
        body = (
            "/**:\n"
            "  ros__parameters:\n"
            f"    feedforward_gain: {gain:.9g}\n"
            f"    feedforward_offset: {offset:.9g}\n"
            f"    reverse_feedforward_gain: {gain:.9g}\n"
            f"    reverse_feedforward_offset: {offset:.9g}\n"
        )
    else:
        body = (
            "/**:\n"
            "  ros__parameters:\n"
            f"    feedforward_gain: {gain:.9g}\n"
            f"    feedforward_offset: {offset:.9g}\n"
            f"    reverse_feedforward_gain: {gain:.9g}\n"
            f"    reverse_feedforward_offset: {offset:.9g}\n"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def write_summary_json(
    path: Path,
    args: argparse.Namespace,
    fit: dict[str, float],
    bag_summaries: list[dict[str, Any]],
    outputs: dict[str, str],
) -> None:
    summary = {
        "fit": fit,
        "filters": {
            "direction": args.direction,
            "command_delay_sec": args.command_delay_sec,
            "min_speed": args.min_speed,
            "max_speed": args.max_speed,
            "min_abs_command": args.min_abs_command,
            "max_abs_command": args.max_abs_command,
            "max_abs_steer": args.max_abs_steer,
            "max_abs_yaw_rate": args.max_abs_yaw_rate,
            "max_abs_accel": args.max_abs_accel,
            "force_zero_offset": args.force_zero_offset,
        },
        "topics": {
            "odom": args.odom_topic,
            "cmd": args.cmd_topic,
        },
        "bags": bag_summaries,
        "outputs": outputs,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plot_fit(path: Path, rows: list[dict[str, float | int | str]], fit: dict[str, float]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for plot output: {exc}") from exc

    speeds = np.asarray([float(row["speed_abs_mps"]) for row in rows], dtype=float)
    commands = np.asarray([float(row["command_abs_throttle"]) for row in rows], dtype=float)
    x_line = np.linspace(0.0, max(float(np.max(speeds)), fit["speed_max_mps"]), 100)
    y_line = fit["feedforward_gain"] * x_line + fit["feedforward_offset"]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(speeds, commands, s=8, alpha=0.35, label="matched samples")
    ax.plot(x_line, y_line, color="tab:red", linewidth=2.0, label="fit")
    ax.set_xlabel("measured speed abs [m/s]")
    ax.set_ylabel("command abs throttle")
    ax.set_title(
        f"speed feedforward fit: throttle = {fit['feedforward_gain']:.4f} * speed "
        f"+ {fit['feedforward_offset']:.4f} (R2={fit['r2']:.3f})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    bag_paths = normalize_bag_paths(args.bag)
    if not bag_paths:
        raise RuntimeError("At least one --bag path is required.")

    stem = default_stem_for_bags(bag_paths)
    param_yaml = (
        Path(args.param_yaml).expanduser().resolve()
        if args.param_yaml
        else Path("/tmp") / f"{stem}_speed_controller_feedforward.param.yaml"
    )
    raw_csv = (
        Path(args.raw_csv).expanduser().resolve()
        if args.raw_csv
        else param_yaml.with_name(param_yaml.stem + "_raw.csv")
    )
    summary_json = (
        Path(args.summary_json).expanduser().resolve()
        if args.summary_json
        else param_yaml.with_name(param_yaml.stem + "_summary.json")
    )
    plot_path = None if args.no_plot else (
        Path(args.plot).expanduser().resolve()
        if args.plot
        else param_yaml.with_name(param_yaml.stem + ".png")
    )

    rows: list[dict[str, float | int | str]] = []
    bag_summaries: list[dict[str, Any]] = []
    for bag_path in bag_paths:
        try:
            odom_samples = load_odom_samples(bag_path, args.odom_topic)
            drive_samples = load_drive_samples(bag_path, args.cmd_topic)
            bag_rows = build_raw_rows(
                odom_samples=odom_samples,
                drive_samples=drive_samples,
                args=args,
                source_bag=str(bag_path),
                require_rows=len(bag_paths) == 1,
            )
        except Exception as exc:
            if len(bag_paths) == 1:
                raise
            print(f"[WARN] Skip bag={bag_path}: {exc}")
            continue

        rows.extend(bag_rows)
        bag_summaries.append(
            {
                "path": str(bag_path),
                "odom_samples": len(odom_samples),
                "cmd_samples": len(drive_samples),
                "matched_samples": len(bag_rows),
            }
        )

    if len(rows) < args.min_samples:
        raise RuntimeError(
            f"Only {len(rows)} matched samples remained, fewer than --min-samples={args.min_samples}."
        )

    fit = fit_feedforward(rows, args.force_zero_offset, args.allow_negative_offset)
    outputs = {
        "param_yaml": str(param_yaml),
        "raw_csv": str(raw_csv),
        "summary_json": str(summary_json),
    }
    if plot_path is not None:
        outputs["plot"] = str(plot_path)

    write_raw_csv(raw_csv, rows)
    write_param_yaml(param_yaml, fit, args.direction)
    write_summary_json(summary_json, args, fit, bag_summaries, outputs)
    if plot_path is not None:
        plot_fit(plot_path, rows, fit)

    print(f"[INFO] bags={len(bag_summaries)}/{len(bag_paths)}")
    for item in bag_summaries:
        print(
            f"[INFO] bag={item['path']} odom={item['odom_samples']} "
            f"cmd={item['cmd_samples']} matched={item['matched_samples']}"
        )
    print(
        "[INFO] fit throttle = "
        f"{fit['feedforward_gain']:.6f} * speed_mps + {fit['feedforward_offset']:.6f} "
        f"(samples={fit['samples']}, R2={fit['r2']:.4f}, rmse={fit['rmse_throttle']:.4f})"
    )
    print(f"[INFO] param_yaml={param_yaml}")
    print(f"[INFO] raw_csv={raw_csv}")
    print(f"[INFO] summary_json={summary_json}")
    if plot_path is not None:
        print(f"[INFO] plot={plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

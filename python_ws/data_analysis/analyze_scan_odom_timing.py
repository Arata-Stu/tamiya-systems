#!/usr/bin/env python3
"""rosbag内の /scan と /odometry の header.stamp 差を集計する。"""

from __future__ import annotations

import argparse
import csv
import inspect
import math
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from rosbags.highlevel import AnyReader


CSV_FIELDS = [
    "scan_index",
    "scan_header_stamp_ns",
    "scan_recorded_stamp_ns",
    "prev_odom_header_stamp_ns",
    "prev_odom_recorded_stamp_ns",
    "next_odom_header_stamp_ns",
    "next_odom_recorded_stamp_ns",
    "nearest_odom_header_stamp_ns",
    "nearest_odom_recorded_stamp_ns",
    "prev_delta_ms",
    "next_delta_ms",
    "nearest_signed_delta_ms",
    "nearest_abs_delta_ms",
    "odom_bracket_gap_ms",
]


@dataclass(frozen=True)
class TopicStamp:
    header_stamp_ns: int
    recorded_stamp_ns: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze timestamp alignment between LaserScan and Odometry messages "
            "inside a rosbag."
        )
    )
    parser.add_argument(
        "--bag",
        required=True,
        help="Path to rosbag2 directory or metadata.yaml",
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--odom-topic", default="/visual_slam/tracking/odometry")
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional CSV output path. Defaults to /tmp/<bag>_scan_odom_timing.csv",
    )
    parser.add_argument(
        "--max-scans",
        type=int,
        default=None,
        help="Optional cap on number of scan messages to analyze.",
    )
    parser.add_argument(
        "--plot",
        default=None,
        help="Optional PNG path for nearest-delta timeseries plot.",
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
        supports_default_typestore = (
            "default_typestore" in inspect.signature(AnyReader).parameters
        )
    except (TypeError, ValueError):
        supports_default_typestore = False

    if supports_default_typestore and default_typestore is not None:
        reader_kwargs["default_typestore"] = default_typestore

    return AnyReader([bag_path], **reader_kwargs)


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def load_topic_stamps(
    bag_path: Path,
    topic: str,
    msgtype: str,
    max_count: int | None = None,
) -> list[TopicStamp]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    try:
        with _open_reader(bag_path) as reader:
            connections = [
                c for c in reader.connections if c.topic == topic and c.msgtype == msgtype
            ]
            if not connections:
                raise RuntimeError(
                    f"Topic not found: topic={topic} msgtype={msgtype} path={bag_path}"
                )

            stamps: list[TopicStamp] = []
            for _idx, (conn, timestamp, raw) in enumerate(
                reader.messages(connections=connections)
            ):
                msg = reader.deserialize(raw, conn.msgtype)
                stamps.append(
                    TopicStamp(
                        header_stamp_ns=stamp_to_ns(msg.header.stamp),
                        recorded_stamp_ns=int(timestamp),
                    )
                )
                if max_count is not None and len(stamps) >= max_count:
                    break

            return stamps
    except Exception as exc:
        msg = str(exc)
        if "default_typestore" in msg and "no type definitions" in msg.lower():
            raise RuntimeError(
                "Bag contains no type definitions and could not load a default ROS2 "
                "typestore. Please update rosbags and ensure rosbags.typesys "
                "Stores/get_typestore are available."
            ) from exc
        raise


def ns_to_ms(ns: int | None) -> float | None:
    if ns is None:
        return None
    return float(ns) / 1_000_000.0


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


def format_ms(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{value:.3f}"


def compute_rows(
    scan_stamps: list[TopicStamp],
    odom_stamps: list[TopicStamp],
) -> list[dict[str, int | float | None]]:
    if not scan_stamps:
        raise RuntimeError("No scan messages found.")
    if not odom_stamps:
        raise RuntimeError("No odometry messages found.")

    odom_header_ns = [item.header_stamp_ns for item in odom_stamps]
    rows: list[dict[str, int | float | None]] = []

    for scan_index, scan in enumerate(scan_stamps):
        insert_idx = bisect_left(odom_header_ns, scan.header_stamp_ns)

        prev_odom = odom_stamps[insert_idx - 1] if insert_idx > 0 else None
        next_odom = odom_stamps[insert_idx] if insert_idx < len(odom_stamps) else None

        prev_delta_ns = (
            scan.header_stamp_ns - prev_odom.header_stamp_ns if prev_odom else None
        )
        next_delta_ns = (
            next_odom.header_stamp_ns - scan.header_stamp_ns if next_odom else None
        )

        nearest_odom = None
        nearest_signed_delta_ns = None
        if prev_odom and next_odom:
            if abs(prev_delta_ns) <= abs(next_delta_ns):
                nearest_odom = prev_odom
                nearest_signed_delta_ns = (
                    prev_odom.header_stamp_ns - scan.header_stamp_ns
                )
            else:
                nearest_odom = next_odom
                nearest_signed_delta_ns = (
                    next_odom.header_stamp_ns - scan.header_stamp_ns
                )
        elif prev_odom:
            nearest_odom = prev_odom
            nearest_signed_delta_ns = prev_odom.header_stamp_ns - scan.header_stamp_ns
        elif next_odom:
            nearest_odom = next_odom
            nearest_signed_delta_ns = next_odom.header_stamp_ns - scan.header_stamp_ns

        row = {
            "scan_index": scan_index,
            "scan_header_stamp_ns": scan.header_stamp_ns,
            "scan_recorded_stamp_ns": scan.recorded_stamp_ns,
            "prev_odom_header_stamp_ns": prev_odom.header_stamp_ns if prev_odom else None,
            "prev_odom_recorded_stamp_ns": (
                prev_odom.recorded_stamp_ns if prev_odom else None
            ),
            "next_odom_header_stamp_ns": next_odom.header_stamp_ns if next_odom else None,
            "next_odom_recorded_stamp_ns": (
                next_odom.recorded_stamp_ns if next_odom else None
            ),
            "nearest_odom_header_stamp_ns": (
                nearest_odom.header_stamp_ns if nearest_odom else None
            ),
            "nearest_odom_recorded_stamp_ns": (
                nearest_odom.recorded_stamp_ns if nearest_odom else None
            ),
            "prev_delta_ms": ns_to_ms(prev_delta_ns),
            "next_delta_ms": ns_to_ms(next_delta_ns),
            "nearest_signed_delta_ms": ns_to_ms(nearest_signed_delta_ns),
            "nearest_abs_delta_ms": (
                abs(ns_to_ms(nearest_signed_delta_ns))
                if nearest_signed_delta_ns is not None
                else None
            ),
            "odom_bracket_gap_ms": (
                ns_to_ms(next_odom.header_stamp_ns - prev_odom.header_stamp_ns)
                if prev_odom and next_odom
                else None
            ),
        }
        rows.append(row)

    return rows


def estimate_rate_hz(stamps: Iterable[TopicStamp]) -> float:
    header_values = [item.header_stamp_ns for item in stamps]
    if len(header_values) < 2:
        return math.nan
    diffs_sec = [
        (curr - prev) / 1_000_000_000.0
        for prev, curr in zip(header_values, header_values[1:])
        if curr > prev
    ]
    if not diffs_sec:
        return math.nan
    return 1.0 / mean(diffs_sec)


def write_csv(rows: list[dict[str, int | float | None]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(
    bag_path: Path,
    scan_topic: str,
    odom_topic: str,
    scan_stamps: list[TopicStamp],
    odom_stamps: list[TopicStamp],
    rows: list[dict[str, int | float | None]],
    csv_path: Path,
    plot_path: Path | None,
) -> None:
    nearest_abs = sorted(
        row["nearest_abs_delta_ms"]
        for row in rows
        if row["nearest_abs_delta_ms"] is not None
    )
    nearest_signed = [
        row["nearest_signed_delta_ms"]
        for row in rows
        if row["nearest_signed_delta_ms"] is not None
    ]
    bracket_gaps = sorted(
        row["odom_bracket_gap_ms"] for row in rows if row["odom_bracket_gap_ms"] is not None
    )
    no_prev = sum(1 for row in rows if row["prev_odom_header_stamp_ns"] is None)
    no_next = sum(1 for row in rows if row["next_odom_header_stamp_ns"] is None)

    print(f"[INFO] bag={bag_path}")
    print(f"[INFO] scan_topic={scan_topic} count={len(scan_stamps)} est_rate_hz={estimate_rate_hz(scan_stamps):.3f}")
    print(f"[INFO] odom_topic={odom_topic} count={len(odom_stamps)} est_rate_hz={estimate_rate_hz(odom_stamps):.3f}")
    print(f"[INFO] csv={csv_path}")
    if plot_path is not None:
        print(f"[INFO] plot={plot_path}")
    print(
        "[INFO] nearest_abs_delta_ms: "
        f"min={format_ms(nearest_abs[0] if nearest_abs else math.nan)} "
        f"median={format_ms(median(nearest_abs) if nearest_abs else math.nan)} "
        f"p90={format_ms(percentile(nearest_abs, 0.90))} "
        f"p99={format_ms(percentile(nearest_abs, 0.99))} "
        f"max={format_ms(nearest_abs[-1] if nearest_abs else math.nan)}"
    )
    print(
        "[INFO] nearest_signed_delta_ms: "
        f"mean={format_ms(mean(nearest_signed) if nearest_signed else math.nan)} "
        f"median={format_ms(median(nearest_signed) if nearest_signed else math.nan)}"
    )
    print(
        "[INFO] odom_bracket_gap_ms: "
        f"median={format_ms(median(bracket_gaps) if bracket_gaps else math.nan)} "
        f"p90={format_ms(percentile(bracket_gaps, 0.90))} "
        f"max={format_ms(bracket_gaps[-1] if bracket_gaps else math.nan)}"
    )
    print(f"[INFO] edge_cases: no_prev_odom={no_prev} no_next_odom={no_next}")


def maybe_plot(rows: list[dict[str, int | float | None]], plot_path: Path | None) -> None:
    if plot_path is None:
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            f"matplotlib is required for --plot but could not be imported: {exc}"
        ) from exc

    x = [row["scan_index"] for row in rows]
    y = [row["nearest_signed_delta_ms"] for row in rows]

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, y, linewidth=1.0)
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("scan index")
    ax.set_ylabel("nearest odom - scan [ms]")
    ax.set_title("Scan/Odometry Header Timestamp Alignment")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)
    csv_path = (
        Path(args.csv).expanduser().resolve()
        if args.csv
        else (Path("/tmp") / f"{bag_path.name}_scan_odom_timing.csv").resolve()
    )
    plot_path = Path(args.plot).expanduser().resolve() if args.plot else None

    scan_stamps = load_topic_stamps(
        bag_path=bag_path,
        topic=args.scan_topic,
        msgtype="sensor_msgs/msg/LaserScan",
        max_count=args.max_scans,
    )
    odom_stamps = load_topic_stamps(
        bag_path=bag_path,
        topic=args.odom_topic,
        msgtype="nav_msgs/msg/Odometry",
    )
    rows = compute_rows(scan_stamps=scan_stamps, odom_stamps=odom_stamps)
    write_csv(rows, csv_path)
    maybe_plot(rows, plot_path)
    print_summary(
        bag_path=bag_path,
        scan_topic=args.scan_topic,
        odom_topic=args.odom_topic,
        scan_stamps=scan_stamps,
        odom_stamps=odom_stamps,
        rows=rows,
        csv_path=csv_path,
        plot_path=plot_path,
    )


if __name__ == "__main__":
    main()

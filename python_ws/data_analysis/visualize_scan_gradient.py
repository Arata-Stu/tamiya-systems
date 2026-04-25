#!/usr/bin/env python3
"""rosbagのLaserScan(/scan)をインデックス色グラデーションで可視化する。"""

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read LaserScan from rosbag and visualize points with index-based color gradient."
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory, metadata.yaml, or rosbag1 .bag")
    parser.add_argument("--topic", default="/scan", help="LaserScan topic name")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to visualize (0-origin). Use -1 to visualize the last frame.",
    )
    parser.add_argument("--cmap", default="turbo", help="Matplotlib colormap name")
    parser.add_argument("--marker_size", type=float, default=8.0, help="Scatter marker size")
    parser.add_argument(
        "--max_range",
        type=float,
        default=None,
        help="Fixed axis range in meters. If omitted, uses scan.range_max.",
    )
    parser.add_argument("--output", default=None, help="Optional output image path (e.g. scan.png)")
    parser.add_argument("--no_show", action="store_true", help="Do not open interactive window")
    return parser.parse_args()


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


def load_target_scan(bag_path: Path, topic: str, frame_index: int):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    target_msg = None
    target_ts = None
    seen = 0

    with AnyReader([bag_path]) as reader:
        connections = [
            c for c in reader.connections if c.topic == topic and c.msgtype == "sensor_msgs/msg/LaserScan"
        ]
        if not connections:
            raise RuntimeError(f"LaserScan topic not found: topic={topic} path={bag_path}")

        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            if frame_index >= 0:
                if seen == frame_index:
                    target_msg = msg
                    target_ts = timestamp
                    break
            else:
                target_msg = msg
                target_ts = timestamp
            seen += 1

    if target_msg is None:
        if frame_index >= 0:
            raise RuntimeError(
                f"Requested frame index {frame_index} is out of range. total_frames={seen}"
            )
        raise RuntimeError("No LaserScan messages found in bag.")

    selected_frame = frame_index if frame_index >= 0 else max(0, seen - 1)
    return target_msg, target_ts, selected_frame


def scan_to_xy(msg):
    ranges = np.asarray(msg.ranges, dtype=np.float32)
    beam_indices = np.arange(ranges.shape[0], dtype=np.int32)
    angles = msg.angle_min + beam_indices.astype(np.float32) * msg.angle_increment

    valid = np.isfinite(ranges)
    valid &= ranges >= float(msg.range_min)
    valid &= ranges <= float(msg.range_max)

    if not np.any(valid):
        raise RuntimeError("All scan points are invalid after range filtering.")

    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    valid_indices = beam_indices[valid]

    x = valid_ranges * np.cos(valid_angles)
    y = valid_ranges * np.sin(valid_angles)
    return x, y, valid_indices, ranges.shape[0]


def visualize_and_save(
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    frame_id: int,
    timestamp_ns: int,
    total_beams: int,
    bag_path: Path,
    topic: str,
    cmap: str,
    marker_size: float,
    max_range: float | None,
    output_path: str | None,
    no_show: bool,
):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(x, y, c=indices, cmap=cmap, s=marker_size, linewidths=0)

    ax.scatter([0.0], [0.0], c="red", s=24, label="LiDAR")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"{bag_path.name} | {topic} | frame={frame_id} | valid={len(indices)}/{total_beams} | ts={timestamp_ns}"
    )

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Beam index")
    ax.legend(loc="upper right")

    if max_range is not None:
        axis_lim = float(max_range)
    else:
        axis_lim = float(max(np.max(np.abs(x)), np.max(np.abs(y))))
        axis_lim = max(axis_lim, 1.0)

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved figure: {out}")

    if not no_show:
        plt.show()


def main() -> None:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)
    msg, timestamp_ns, selected_frame = load_target_scan(bag_path, args.topic, args.frame)
    x, y, indices, total_beams = scan_to_xy(msg)

    visualize_and_save(
        x=x,
        y=y,
        indices=indices,
        frame_id=selected_frame,
        timestamp_ns=timestamp_ns,
        total_beams=total_beams,
        bag_path=bag_path,
        topic=args.topic,
        cmap=args.cmap,
        marker_size=args.marker_size,
        max_range=args.max_range,
        output_path=args.output,
        no_show=args.no_show,
    )


if __name__ == "__main__":
    main()

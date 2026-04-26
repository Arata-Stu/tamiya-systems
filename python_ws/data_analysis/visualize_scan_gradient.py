#!/usr/bin/env python3
"""rosbagのLaserScan(/scan)をインデックス色グラデーションで可視化する。"""

import argparse
import inspect
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
    parser.add_argument(
        "--video_output",
        default=None,
        help="Optional output video path (.mp4 or .gif). If set, runs timeseries visualization mode.",
    )
    parser.add_argument("--video_fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--video_start", type=int, default=0, help="Start frame index for video mode")
    parser.add_argument("--video_end", type=int, default=None, help="End frame index for video mode (inclusive)")
    parser.add_argument("--video_step", type=int, default=1, help="Frame stride for video mode")
    parser.add_argument("--no_show", action="store_true", help="Do not open interactive window")
    return parser.parse_args()


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


def _build_default_ros2_typestore():
    """Create a sensible default typestore for ROS2 bags without embedded definitions."""
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
    """Open AnyReader with default typestore when supported/available."""
    reader_kwargs = {}
    default_typestore = _build_default_ros2_typestore()

    try:
        supports_default_typestore = "default_typestore" in inspect.signature(AnyReader).parameters
    except (TypeError, ValueError):
        supports_default_typestore = False

    if supports_default_typestore and default_typestore is not None:
        reader_kwargs["default_typestore"] = default_typestore

    return AnyReader([bag_path], **reader_kwargs)


def _iter_scan_messages(bag_path: Path, topic: str):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    try:
        with _open_reader(bag_path) as reader:
            connections = [
                c for c in reader.connections if c.topic == topic and c.msgtype == "sensor_msgs/msg/LaserScan"
            ]
            if not connections:
                raise RuntimeError(f"LaserScan topic not found: topic={topic} path={bag_path}")

            for frame_id, (conn, timestamp, raw) in enumerate(reader.messages(connections=connections)):
                yield frame_id, timestamp, reader.deserialize(raw, conn.msgtype)
    except Exception as exc:
        msg = str(exc)
        if "default_typestore" in msg and "no type definitions" in msg.lower():
            raise RuntimeError(
                "Bag contains no type definitions and could not load a default ROS2 typestore. "
                "Please update rosbags and ensure rosbags.typesys Stores/get_typestore are available."
            ) from exc
        raise


def load_target_scan(bag_path: Path, topic: str, frame_index: int):
    target_msg = None
    target_ts = None
    selected_frame = None
    total_frames = 0

    for seen, timestamp, msg in _iter_scan_messages(bag_path, topic):
        total_frames = seen + 1
        if frame_index >= 0:
            if seen == frame_index:
                target_msg = msg
                target_ts = timestamp
                selected_frame = seen
                break
        else:
            target_msg = msg
            target_ts = timestamp
            selected_frame = seen

    if target_msg is None:
        if frame_index >= 0:
            raise RuntimeError(
                f"Requested frame index {frame_index} is out of range. total_frames={total_frames}"
            )
        raise RuntimeError("No LaserScan messages found in bag.")

    return target_msg, target_ts, selected_frame


def collect_video_frames(
    bag_path: Path,
    topic: str,
    start_frame: int,
    end_frame: int | None,
    frame_step: int,
):
    if start_frame < 0:
        raise ValueError("--video_start must be >= 0.")
    if end_frame is not None and end_frame < start_frame:
        raise ValueError("--video_end must be >= --video_start.")
    if frame_step <= 0:
        raise ValueError("--video_step must be >= 1.")

    frames = []
    skipped_invalid = 0

    for frame_id, timestamp, msg in _iter_scan_messages(bag_path, topic):
        if frame_id < start_frame:
            continue
        if end_frame is not None and frame_id > end_frame:
            break
        if (frame_id - start_frame) % frame_step != 0:
            continue

        try:
            x, y, indices, total_beams = scan_to_xy(msg)
        except RuntimeError:
            skipped_invalid += 1
            continue

        frames.append((frame_id, timestamp, x, y, indices, total_beams))

    if not frames:
        raise RuntimeError("No valid frames found for video generation with current frame selection.")

    return frames, skipped_invalid


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


def visualize_video_and_save(
    frames,
    bag_path: Path,
    topic: str,
    cmap: str,
    marker_size: float,
    max_range: float | None,
    video_output_path: str,
    video_fps: float,
    no_show: bool,
):
    try:
        import matplotlib.pyplot as plt
        from matplotlib import animation
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install with: pip install matplotlib"
        ) from exc

    out = Path(video_output_path).expanduser().resolve()
    suffix = out.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError("--video_output must end with .mp4 or .gif")
    if video_fps <= 0:
        raise ValueError("--video_fps must be > 0.")

    if max_range is not None:
        axis_lim = float(max_range)
    else:
        axis_lim = 1.0
        for _, _, x, y, _, _ in frames:
            axis_lim = max(axis_lim, float(np.max(np.abs(x))), float(np.max(np.abs(y))))

    max_total_beams = max(frame[5] for frame in frames)
    max_beam_index = max(1, max_total_beams - 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(
        [],
        [],
        c=[],
        cmap=cmap,
        s=marker_size,
        linewidths=0,
        vmin=0,
        vmax=max_beam_index,
    )

    ax.scatter([0.0], [0.0], c="red", s=24, label="LiDAR")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    title = ax.set_title("")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Beam index")
    ax.legend(loc="upper right")

    def update(plot_index: int):
        frame_id, timestamp_ns, x, y, indices, total_beams = frames[plot_index]
        points = np.column_stack((x, y))
        scatter.set_offsets(points)
        scatter.set_array(indices.astype(np.float32))
        scatter.set_clim(0, max(1, total_beams - 1))
        title.set_text(
            f"{bag_path.name} | {topic} | frame={frame_id} | valid={len(indices)}/{total_beams} | ts={timestamp_ns}"
        )
        return scatter, title

    ani = animation.FuncAnimation(
        fig=fig,
        func=update,
        frames=len(frames),
        interval=1000.0 / float(video_fps),
        blit=False,
        repeat=False,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        if suffix == ".mp4":
            ani.save(out, writer="ffmpeg", fps=video_fps, dpi=150)
        else:
            ani.save(out, writer="pillow", fps=video_fps, dpi=150)
    except Exception as exc:
        if suffix == ".mp4":
            raise RuntimeError(
                "Failed to save MP4. Please install ffmpeg, or output GIF by using --video_output xxx.gif."
            ) from exc
        raise RuntimeError("Failed to save GIF. Please ensure pillow is installed.") from exc

    print(f"[INFO] Saved video: {out}")
    if not no_show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)
    if args.video_output:
        frames, skipped_invalid = collect_video_frames(
            bag_path=bag_path,
            topic=args.topic,
            start_frame=args.video_start,
            end_frame=args.video_end,
            frame_step=args.video_step,
        )
        if skipped_invalid:
            print(f"[WARN] Skipped {skipped_invalid} invalid scan frame(s).")

        visualize_video_and_save(
            frames=frames,
            bag_path=bag_path,
            topic=args.topic,
            cmap=args.cmap,
            marker_size=args.marker_size,
            max_range=args.max_range,
            video_output_path=args.video_output,
            video_fps=args.video_fps,
            no_show=args.no_show,
        )
        return

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

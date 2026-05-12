#!/usr/bin/env python3
"""rosbag画像にcrop範囲を重ねて動画/GIF化する。"""

from __future__ import annotations

import argparse
import inspect
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    AnyReader = object

try:
    import cv2
except ImportError:
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay camera crop region on rosbag image frames and export MP4/GIF for visual review."
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory, metadata.yaml, or rosbag1 .bag")
    parser.add_argument("--topic", required=True, help="Image topic name")
    parser.add_argument("--output", required=True, help="Output path (.mp4 or .gif)")
    parser.add_argument("--video_fps", type=float, default=10.0, help="Output FPS")
    parser.add_argument("--video_start", type=int, default=0, help="Start frame index")
    parser.add_argument("--video_end", type=int, default=None, help="End frame index (inclusive)")
    parser.add_argument("--video_step", type=int, default=1, help="Frame stride")
    parser.add_argument("--top_ratio", type=float, default=0.0, help="Crop ratio from top edge [0,1)")
    parser.add_argument("--bottom_ratio", type=float, default=0.0, help="Crop ratio from bottom edge [0,1)")
    parser.add_argument("--left_ratio", type=float, default=0.0, help="Crop ratio from left edge [0,1)")
    parser.add_argument("--right_ratio", type=float, default=0.0, help="Crop ratio from right edge [0,1)")
    parser.add_argument(
        "--shade_alpha",
        type=float,
        default=0.35,
        help="Overlay alpha for cropped-out area. 0=none, 1=solid.",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=2,
        help="Crop boundary line thickness in pixels",
    )
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional output width. Height is scaled automatically.",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Optional label shown on frames. Defaults to crop ratios.",
    )
    return parser.parse_args()


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    return np


def require_rosbags():
    if AnyReader is object:
        raise RuntimeError("rosbags is required. Install with: pip install rosbags")
    return AnyReader


def require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")
    return cv2


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

    preferred_store_names = ("ROS2_JAZZY", "ROS2_IRON", "ROS2_HUMBLE", "ROS2_GALACTIC", "ROS2_FOXY")
    for store_name in preferred_store_names:
        if hasattr(Stores, store_name):
            return get_typestore(getattr(Stores, store_name))

    for store in Stores:
        if store.name.startswith("ROS2_"):
            return get_typestore(store)

    return None


def _open_reader(bag_path: Path):
    any_reader_cls = require_rosbags()
    reader_kwargs = {}
    default_typestore = _build_default_ros2_typestore()

    try:
        supports_default_typestore = "default_typestore" in inspect.signature(any_reader_cls).parameters
    except (TypeError, ValueError):
        supports_default_typestore = False

    if supports_default_typestore and default_typestore is not None:
        reader_kwargs["default_typestore"] = default_typestore

    return any_reader_cls([bag_path], **reader_kwargs)


def _decode_raw_image_to_rgb(msg) -> Optional[np.ndarray]:
    cv = require_cv2()
    np_mod = require_numpy()

    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = msg.encoding.lower()
    step = int(msg.step)
    raw = np_mod.frombuffer(msg.data, dtype=np_mod.uint8)
    if step <= 0:
        return None

    expected = height * step
    if raw.size < expected:
        return None
    raw = raw[:expected].reshape(height, step)

    if encoding == "rgb8":
        return raw[:, : width * 3].reshape(height, width, 3).copy()
    if encoding in ("bgr8", "8uc3"):
        img = raw[:, : width * 3].reshape(height, width, 3)
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if encoding == "rgba8":
        img = raw[:, : width * 4].reshape(height, width, 4)
        return cv.cvtColor(img, cv.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        img = raw[:, : width * 4].reshape(height, width, 4)
        return cv.cvtColor(img, cv.COLOR_BGRA2RGB)
    if encoding in ("mono8", "8uc1"):
        mono = raw[:, :width]
        return cv.cvtColor(mono, cv.COLOR_GRAY2RGB)
    if encoding in ("mono16", "16uc1", "16sc1"):
        raw16 = np_mod.frombuffer(msg.data, dtype=np_mod.uint16)
        if raw16.size < height * width:
            return None
        img16 = raw16[: height * width].reshape(height, width)
        img8 = (img16 / 256.0).astype(np_mod.uint8)
        return cv.cvtColor(img8, cv.COLOR_GRAY2RGB)
    if encoding in ("yuyv", "yuyv422", "yuv422", "uyvy"):
        img = raw[:, : width * 2].reshape(height, width, 2)
        code = cv.COLOR_YUV2RGB_YUY2 if encoding != "uyvy" else cv.COLOR_YUV2RGB_UYVY
        return cv.cvtColor(img, code)
    return None


def _decode_compressed_image_to_rgb(msg) -> Optional[np.ndarray]:
    cv = require_cv2()
    np_mod = require_numpy()
    buf = np_mod.frombuffer(msg.data, dtype=np_mod.uint8)
    bgr = cv.imdecode(buf, cv.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)


def _iter_image_messages(bag_path: Path, topic: str):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    try:
        with _open_reader(bag_path) as reader:
            connections = [
                c
                for c in reader.connections
                if c.topic == topic and c.msgtype in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
            ]
            if not connections:
                raise RuntimeError(f"Image topic not found: topic={topic} path={bag_path}")

            for frame_id, (conn, timestamp, raw) in enumerate(reader.messages(connections=connections)):
                msg = reader.deserialize(raw, conn.msgtype)
                image = None
                if conn.msgtype == "sensor_msgs/msg/Image":
                    image = _decode_raw_image_to_rgb(msg)
                elif conn.msgtype == "sensor_msgs/msg/CompressedImage":
                    image = _decode_compressed_image_to_rgb(msg)
                if image is not None:
                    yield frame_id, timestamp, image
    except Exception as exc:
        msg = str(exc)
        if "default_typestore" in msg and "no type definitions" in msg.lower():
            raise RuntimeError(
                "Bag contains no type definitions and could not load a default ROS2 typestore. "
                "Please update rosbags and ensure rosbags.typesys Stores/get_typestore are available."
            ) from exc
        raise


def validate_ratios(args: argparse.Namespace) -> None:
    ratios = [args.top_ratio, args.bottom_ratio, args.left_ratio, args.right_ratio]
    if any(r < 0.0 or r >= 1.0 for r in ratios):
        raise ValueError("Each crop ratio must be within [0, 1).")
    if args.top_ratio + args.bottom_ratio >= 1.0:
        raise ValueError("top_ratio + bottom_ratio must be < 1.0")
    if args.left_ratio + args.right_ratio >= 1.0:
        raise ValueError("left_ratio + right_ratio must be < 1.0")
    if args.video_fps <= 0:
        raise ValueError("--video_fps must be > 0")
    if args.video_start < 0:
        raise ValueError("--video_start must be >= 0")
    if args.video_end is not None and args.video_end < args.video_start:
        raise ValueError("--video_end must be >= --video_start")
    if args.video_step <= 0:
        raise ValueError("--video_step must be >= 1")


def build_label(args: argparse.Namespace) -> str:
    if args.label:
        return args.label
    return (
        f"top={args.top_ratio:.3f} bottom={args.bottom_ratio:.3f} "
        f"left={args.left_ratio:.3f} right={args.right_ratio:.3f}"
    )


def overlay_crop_region(image_rgb, args: argparse.Namespace, frame_id: int, timestamp_ns: int, topic: str, bag_name: str):
    cv = require_cv2()
    np_mod = require_numpy()

    canvas = image_rgb.copy()
    h, w = canvas.shape[:2]

    x0 = int(round(w * args.left_ratio))
    x1 = int(round(w * (1.0 - args.right_ratio)))
    y0 = int(round(h * args.top_ratio))
    y1 = int(round(h * (1.0 - args.bottom_ratio)))

    overlay = canvas.copy()
    shade_color = np_mod.array([255, 80, 80], dtype=np_mod.uint8)
    if args.shade_alpha > 0.0:
        if y0 > 0:
            overlay[:y0, :, :] = ((1.0 - args.shade_alpha) * overlay[:y0, :, :] + args.shade_alpha * shade_color).astype(np_mod.uint8)
        if y1 < h:
            overlay[y1:, :, :] = ((1.0 - args.shade_alpha) * overlay[y1:, :, :] + args.shade_alpha * shade_color).astype(np_mod.uint8)
        if x0 > 0:
            overlay[y0:y1, :x0, :] = (
                (1.0 - args.shade_alpha) * overlay[y0:y1, :x0, :] + args.shade_alpha * shade_color
            ).astype(np_mod.uint8)
        if x1 < w:
            overlay[y0:y1, x1:, :] = (
                (1.0 - args.shade_alpha) * overlay[y0:y1, x1:, :] + args.shade_alpha * shade_color
            ).astype(np_mod.uint8)
        canvas = overlay

    cv.rectangle(canvas, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), (80, 255, 80), args.line_thickness)
    cv.putText(
        canvas,
        f"{bag_name} | {topic} | frame={frame_id} | ts={timestamp_ns}",
        (12, 24),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        canvas,
        build_label(args),
        (12, 48),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )
    cv.putText(
        canvas,
        f"crop_px: top={y0} bottom={h - y1} left={x0} right={w - x1}",
        (12, 72),
        cv.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv.LINE_AA,
    )

    if args.resize_width is not None and args.resize_width > 0 and args.resize_width != w:
        new_w = int(args.resize_width)
        new_h = max(1, int(round(h * (new_w / float(w)))))
        canvas = cv.resize(canvas, (new_w, new_h), interpolation=cv.INTER_AREA)

    return canvas


def collect_frames(bag_path: Path, topic: str, args: argparse.Namespace):
    frames = []
    for frame_id, timestamp_ns, image_rgb in _iter_image_messages(bag_path, topic):
        if frame_id < args.video_start:
            continue
        if args.video_end is not None and frame_id > args.video_end:
            break
        if (frame_id - args.video_start) % args.video_step != 0:
            continue
        frames.append((frame_id, timestamp_ns, image_rgb))
    if not frames:
        raise RuntimeError("No image frames matched the current selection.")
    return frames


def save_mp4(frames_rgb, out_path: Path, fps: float) -> None:
    cv = require_cv2()
    first = frames_rgb[0]
    height, width = first.shape[:2]
    writer = cv.VideoWriter(
        str(out_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Failed to open MP4 writer.")
    try:
        for frame_rgb in frames_rgb:
            writer.write(cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR))
    finally:
        writer.release()


def save_gif(frames_rgb, out_path: Path, fps: float) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for GIF export. Install with: pip install pillow") from exc

    pil_frames = [Image.fromarray(frame_rgb) for frame_rgb in frames_rgb]
    duration_ms = int(round(1000.0 / fps))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=max(1, duration_ms),
        loop=0,
    )


def main() -> None:
    args = parse_args()
    validate_ratios(args)
    bag_path = normalize_bag_path(args.bag)
    out_path = Path(args.output).expanduser().resolve()
    suffix = out_path.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError("--output must end with .mp4 or .gif")

    frames = collect_frames(bag_path, args.topic, args)
    rendered_frames = [
        overlay_crop_region(image_rgb, args, frame_id, timestamp_ns, args.topic, bag_path.name)
        for frame_id, timestamp_ns, image_rgb in frames
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".mp4":
        save_mp4(rendered_frames, out_path, args.video_fps)
    else:
        save_gif(rendered_frames, out_path, args.video_fps)

    print(f"[INFO] Saved {len(rendered_frames)} frames to {out_path}")


if __name__ == "__main__":
    main()

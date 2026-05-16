#!/usr/bin/env python3
"""rosbag の image + camera_info + scan から LiDAR 点群を画像へ投影して可視化する。"""

from __future__ import annotations

import argparse
import inspect
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    AnyReader = object


BOARD_THICKNESS_M = 0.002
LENS_LENGTH_M = 0.014
DEFAULT_CAMERA_INTEROCULAR_M = 0.05
OPTICAL_RPY = (-math.pi / 2.0, 0.0, -math.pi / 2.0)


@dataclass(frozen=True)
class HeaderStampedMessage:
    frame_index: int
    header_stamp_ns: int
    recorded_stamp_ns: int
    frame_id: str
    msg: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project LaserScan points onto camera images from a rosbag for "
            "visual LiDAR-camera alignment debugging."
        )
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory, metadata.yaml, or rosbag1 .bag")
    parser.add_argument("--image-topic", required=True, help="Image topic name")
    parser.add_argument(
        "--camera-info-topic",
        default=None,
        help="CameraInfo topic name. Defaults to the image topic's sibling /camera_info.",
    )
    parser.add_argument("--scan-topic", default="/scan", help="LaserScan topic name")
    parser.add_argument("--tf-static-topic", default="/tf_static", help="Static TF topic name")
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Image frame index to visualize (0-origin). Use -1 for the last selected frame.",
    )
    parser.add_argument("--output", default=None, help="Optional PNG output path for single-frame mode")
    parser.add_argument(
        "--video-output",
        default=None,
        help="Optional MP4 output path. If set, video mode is enabled.",
    )
    parser.add_argument("--video-fps", type=float, default=10.0, help="Output video FPS")
    parser.add_argument("--video-start", type=int, default=0, help="Start frame index for video mode")
    parser.add_argument("--video-end", type=int, default=None, help="End frame index for video mode (inclusive)")
    parser.add_argument("--video-step", type=int, default=1, help="Frame stride for video mode")
    parser.add_argument("--point-radius", type=int, default=2, help="Projected point radius in pixels")
    parser.add_argument(
        "--color-mode",
        choices=("range", "index"),
        default="range",
        help="Color projected points by range/depth or by LaserScan beam index.",
    )
    parser.add_argument(
        "--print-index-colors",
        action="store_true",
        help="Print start/mid/end beam indices and their colors in index mode.",
    )
    parser.add_argument(
        "--highlight-index",
        action="append",
        default=[],
        help="Highlight a beam index in index mode. Accepts integers or start/mid/end. Repeatable.",
    )
    parser.add_argument(
        "--max-range",
        type=float,
        default=12.0,
        help="Discard scan points farther than this range before projection [m].",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=0.05,
        help="Discard scan points nearer than this range before projection [m].",
    )
    parser.add_argument(
        "--camera-side",
        choices=("auto", "left", "right"),
        default="auto",
        help="Fallback optical-frame side when TF static does not include camera optical frames.",
    )
    parser.add_argument("--base-frame", default="base_link", help="Base frame name for fallback extrinsics")
    parser.add_argument("--lidar-frame", default="laser", help="LiDAR frame name for fallback extrinsics")
    parser.add_argument(
        "--camera-frame",
        default="camera_camera_link",
        help="Camera body frame name for fallback extrinsics",
    )
    parser.add_argument("--lidar-x", type=float, default=0.2725)
    parser.add_argument("--lidar-y", type=float, default=0.0)
    parser.add_argument("--lidar-z", type=float, default=0.0257)
    parser.add_argument("--lidar-roll", type=float, default=0.0)
    parser.add_argument("--lidar-pitch", type=float, default=0.0)
    parser.add_argument("--lidar-yaw", type=float, default=0.0)
    parser.add_argument("--camera-x", type=float, default=0.2075)
    parser.add_argument("--camera-y", type=float, default=0.019)
    parser.add_argument("--camera-z", type=float, default=0.065)
    parser.add_argument("--camera-roll", type=float, default=0.0)
    parser.add_argument("--camera-pitch", type=float, default=0.0)
    parser.add_argument("--camera-yaw", type=float, default=0.0)
    parser.add_argument("--camera-mount-to-lens-x", type=float, default=0.0)
    parser.add_argument("--camera-mount-to-lens-y", type=float, default=-0.0075)
    parser.add_argument("--camera-mount-to-lens-z", type=float, default=0.0)
    parser.add_argument("--camera-interocular", type=float, default=DEFAULT_CAMERA_INTEROCULAR_M)
    parser.add_argument(
        "--ignore-bag-tf-static",
        action="store_true",
        help="Ignore /tf_static in the bag and use fallback config extrinsics only.",
    )
    parser.add_argument(
        "--use-recorded-timestamp",
        action="store_true",
        help="Use rosbag recorded timestamps instead of header.stamp for nearest scan matching.",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open an interactive window in single-frame mode")
    return parser.parse_args()


def require_numpy():
    if np is None:
        raise RuntimeError("numpy is required. Install with: pip install numpy")
    return np


def require_cv2():
    if cv2 is None:
        raise RuntimeError("opencv-python is required. Install with: pip install opencv-python")
    return cv2


def require_rosbags():
    if AnyReader is object:
        raise RuntimeError("rosbags is required. Install with: pip install rosbags")
    return AnyReader


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


def derive_camera_info_topic(image_topic: str) -> str:
    if image_topic.endswith("/image_raw"):
        return image_topic[: -len("/image_raw")] + "/camera_info"
    if image_topic.endswith("/image_rect_raw"):
        return image_topic[: -len("/image_rect_raw")] + "/camera_info"
    if image_topic.endswith("/image_compressed"):
        return image_topic[: -len("/image_compressed")] + "/camera_info"
    return image_topic.rstrip("/") + "/camera_info"


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


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def choose_matching_stamp(sample: HeaderStampedMessage, use_recorded_timestamp: bool) -> int:
    return sample.recorded_stamp_ns if use_recorded_timestamp else sample.header_stamp_ns


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


def should_select_image_frame(frame_index: int, args: argparse.Namespace) -> bool:
    if args.video_output:
        if frame_index < args.video_start:
            return False
        if args.video_end is not None and frame_index > args.video_end:
            return False
        return (frame_index - args.video_start) % args.video_step == 0
    if args.frame >= 0:
        return frame_index == args.frame
    return True


def collect_projection_inputs(
    bag_path: Path,
    image_topic: str,
    camera_info_topic: str,
    scan_topic: str,
    tf_static_topic: str,
    args: argparse.Namespace,
) -> tuple[list[HeaderStampedMessage], list[HeaderStampedMessage], list[HeaderStampedMessage], list[object], int]:
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    images: list[HeaderStampedMessage] = []
    scans: list[HeaderStampedMessage] = []
    camera_infos: list[HeaderStampedMessage] = []
    tf_static_msgs: list[object] = []
    image_frame_count = 0

    try:
        with _open_reader(bag_path) as reader:
            connections = [
                c
                for c in reader.connections
                if (c.topic == image_topic and c.msgtype in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage"))
                or (c.topic == camera_info_topic and c.msgtype == "sensor_msgs/msg/CameraInfo")
                or (c.topic == scan_topic and c.msgtype == "sensor_msgs/msg/LaserScan")
                or (not args.ignore_bag_tf_static and c.topic == tf_static_topic and c.msgtype == "tf2_msgs/msg/TFMessage")
            ]
            if not connections:
                raise RuntimeError("No matching topics were found in the bag.")

            for conn, recorded_stamp_ns, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)

                if conn.topic == image_topic:
                    selected = should_select_image_frame(image_frame_count, args)
                    image = None
                    if selected:
                        if conn.msgtype == "sensor_msgs/msg/Image":
                            image = _decode_raw_image_to_rgb(msg)
                        else:
                            image = _decode_compressed_image_to_rgb(msg)
                    if selected and image is not None:
                        sample = HeaderStampedMessage(
                            frame_index=image_frame_count,
                            header_stamp_ns=stamp_to_ns(msg.header.stamp),
                            recorded_stamp_ns=int(recorded_stamp_ns),
                            frame_id=str(msg.header.frame_id),
                            msg=image,
                        )
                        if args.video_output or args.frame >= 0:
                            images.append(sample)
                        else:
                            images = [sample]
                    image_frame_count += 1
                    continue

                if conn.topic == camera_info_topic:
                    camera_infos.append(
                        HeaderStampedMessage(
                            frame_index=len(camera_infos),
                            header_stamp_ns=stamp_to_ns(msg.header.stamp),
                            recorded_stamp_ns=int(recorded_stamp_ns),
                            frame_id=str(msg.header.frame_id),
                            msg=msg,
                        )
                    )
                    continue

                if conn.topic == scan_topic:
                    scans.append(
                        HeaderStampedMessage(
                            frame_index=len(scans),
                            header_stamp_ns=stamp_to_ns(msg.header.stamp),
                            recorded_stamp_ns=int(recorded_stamp_ns),
                            frame_id=str(msg.header.frame_id),
                            msg=msg,
                        )
                    )
                    continue

                if not args.ignore_bag_tf_static and conn.topic == tf_static_topic:
                    tf_static_msgs.append(msg)
    except Exception as exc:
        msg = str(exc)
        if "default_typestore" in msg and "no type definitions" in msg.lower():
            raise RuntimeError(
                "Bag contains no type definitions and could not load a default ROS2 typestore. "
                "Please update rosbags and ensure rosbags.typesys Stores/get_typestore are available."
            ) from exc
        raise

    if not images:
        raise RuntimeError(f"No decodable image frames found on {image_topic}")
    if args.frame >= 0 and len(images) != 1:
        raise RuntimeError(f"Requested frame {args.frame} was not found in {image_topic}")
    if not camera_infos:
        raise RuntimeError(f"No CameraInfo messages found on {camera_info_topic}")
    if not scans:
        raise RuntimeError(f"No LaserScan messages found on {scan_topic}")

    return images, scans, camera_infos, tf_static_msgs, image_frame_count


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    np_mod = require_numpy()
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    rx = np_mod.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np_mod.float64)
    ry = np_mod.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np_mod.float64)
    rz = np_mod.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np_mod.float64)
    return rz @ ry @ rx


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    np_mod = require_numpy()
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm == 0.0:
        return np_mod.eye(3, dtype=np_mod.float64)
    x /= norm
    y /= norm
    z /= norm
    w /= norm
    return np_mod.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np_mod.float64,
    )


def make_transform(translation_xyz: tuple[float, float, float], rotation_matrix: np.ndarray) -> np.ndarray:
    np_mod = require_numpy()
    transform = np_mod.eye(4, dtype=np_mod.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np_mod.asarray(translation_xyz, dtype=np_mod.float64)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    np_mod = require_numpy()
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    inv = np_mod.eye(4, dtype=np_mod.float64)
    inv[:3, :3] = rot.T
    inv[:3, 3] = -(rot.T @ trans)
    return inv


def transform_from_tf_msg(transform_msg) -> np.ndarray:
    rotation = transform_msg.rotation
    translation = transform_msg.translation
    return make_transform(
        (float(translation.x), float(translation.y), float(translation.z)),
        quaternion_to_matrix(
            float(rotation.x),
            float(rotation.y),
            float(rotation.z),
            float(rotation.w),
        ),
    )


def build_tf_graph_from_bag(tf_static_msgs: list[object]) -> dict[str, list[tuple[str, np.ndarray]]]:
    graph: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    for msg in tf_static_msgs:
        for transform_stamped in msg.transforms:
            parent = str(transform_stamped.header.frame_id)
            child = str(transform_stamped.child_frame_id)
            transform = transform_from_tf_msg(transform_stamped.transform)
            graph[child].append((parent, transform))
            graph[parent].append((child, invert_transform(transform)))
    return graph


def add_transform_edge(
    graph: dict[str, list[tuple[str, np.ndarray]]],
    parent_frame: str,
    child_frame: str,
    transform_parent_from_child: np.ndarray,
) -> None:
    graph[child_frame].append((parent_frame, transform_parent_from_child))
    graph[parent_frame].append((child_frame, invert_transform(transform_parent_from_child)))


def infer_camera_side(frame_id: str, image_topic: str, camera_side_arg: str) -> str:
    if camera_side_arg in ("left", "right"):
        return camera_side_arg

    candidate = f"{frame_id} {image_topic}".lower()
    if "infra2" in candidate or "right" in candidate:
        return "right"
    return "left"


def build_fallback_tf_graph(
    args: argparse.Namespace,
    camera_optical_frame: str,
    camera_side: str,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    graph: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)

    base_to_laser = make_transform(
        (args.lidar_x, args.lidar_y, args.lidar_z),
        rpy_to_matrix(args.lidar_roll, args.lidar_pitch, args.lidar_yaw),
    )
    add_transform_edge(graph, args.base_frame, args.lidar_frame, base_to_laser)

    base_to_camera_link = make_transform(
        (
            args.camera_x + args.camera_mount_to_lens_x,
            args.camera_y + args.camera_mount_to_lens_y,
            args.camera_z + args.camera_mount_to_lens_z,
        ),
        rpy_to_matrix(args.camera_roll, args.camera_pitch, args.camera_yaw),
    )
    add_transform_edge(graph, args.base_frame, args.camera_frame, base_to_camera_link)

    optical_y = args.camera_interocular / 2.0
    if camera_side == "right":
        optical_y = -optical_y
    camera_link_to_optical = make_transform(
        (BOARD_THICKNESS_M / 2.0 + LENS_LENGTH_M, optical_y, 0.0),
        rpy_to_matrix(*OPTICAL_RPY),
    )
    add_transform_edge(graph, args.camera_frame, camera_optical_frame, camera_link_to_optical)

    return graph


def merged_tf_graph(
    tf_static_msgs: list[object],
    args: argparse.Namespace,
    camera_optical_frame: str,
    camera_side: str,
) -> dict[str, list[tuple[str, np.ndarray]]]:
    graph = build_tf_graph_from_bag(tf_static_msgs)
    fallback_graph = build_fallback_tf_graph(args, camera_optical_frame, camera_side)
    for frame, edges in fallback_graph.items():
        graph[frame].extend(edges)
    return graph


def resolve_transform(
    graph: dict[str, list[tuple[str, np.ndarray]]],
    source_frame: str,
    target_frame: str,
) -> np.ndarray:
    np_mod = require_numpy()
    if source_frame == target_frame:
        return np_mod.eye(4, dtype=np_mod.float64)

    visited = {source_frame}
    queue = deque([(source_frame, np_mod.eye(4, dtype=np_mod.float64))])

    while queue:
        current_frame, transform_current_from_source = queue.popleft()
        for next_frame, transform_next_from_current in graph.get(current_frame, []):
            if next_frame in visited:
                continue
            transform_next_from_source = transform_next_from_current @ transform_current_from_source
            if next_frame == target_frame:
                return transform_next_from_source
            visited.add(next_frame)
            queue.append((next_frame, transform_next_from_source))

    raise RuntimeError(f"Could not resolve transform: {source_frame} -> {target_frame}")


def select_nearest_sample(
    samples: list[HeaderStampedMessage],
    target_stamp_ns: int,
    use_recorded_timestamp: bool,
) -> HeaderStampedMessage:
    if not samples:
        raise RuntimeError("No samples available for nearest lookup.")

    best = min(
        samples,
        key=lambda sample: abs(choose_matching_stamp(sample, use_recorded_timestamp) - target_stamp_ns),
    )
    return best


def scan_to_points_lidar(scan_msg, min_range_m: float, max_range_m: float) -> tuple[np.ndarray, np.ndarray, int]:
    np_mod = require_numpy()
    ranges = np_mod.asarray(scan_msg.ranges, dtype=np_mod.float32)
    beam_indices = np_mod.arange(ranges.shape[0], dtype=np_mod.float32)
    angles = float(scan_msg.angle_min) + beam_indices * float(scan_msg.angle_increment)

    valid = np_mod.isfinite(ranges)
    valid &= ranges >= max(float(scan_msg.range_min), min_range_m)
    valid &= ranges <= min(float(scan_msg.range_max), max_range_m)

    if not np_mod.any(valid):
        return (
            np_mod.empty((0, 3), dtype=np_mod.float32),
            np_mod.empty((0,), dtype=np_mod.float32),
            int(ranges.shape[0]),
        )

    valid_ranges = ranges[valid]
    valid_angles = angles[valid]
    valid_indices = beam_indices[valid]
    points = np_mod.stack(
        [
            valid_ranges * np_mod.cos(valid_angles),
            valid_ranges * np_mod.sin(valid_angles),
            np_mod.zeros_like(valid_ranges),
        ],
        axis=1,
    )
    return points.astype(np_mod.float32), valid_indices.astype(np_mod.float32), int(ranges.shape[0])


def project_points_to_image(
    points_lidar: np.ndarray,
    transform_camera_from_lidar: np.ndarray,
    camera_info_msg,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np_mod = require_numpy()

    if points_lidar.size == 0:
        return (
            np_mod.empty((0, 2), dtype=np_mod.float32),
            np_mod.empty((0,), dtype=np_mod.float32),
            np_mod.empty((0,), dtype=bool),
        )

    points_h = np_mod.concatenate(
        [points_lidar.astype(np_mod.float64), np_mod.ones((points_lidar.shape[0], 1), dtype=np_mod.float64)],
        axis=1,
    )
    points_camera = (transform_camera_from_lidar @ points_h.T).T[:, :3]
    depths = points_camera[:, 2].astype(np_mod.float32)
    positive_depth = depths > 1e-6
    points_camera = points_camera[positive_depth]
    depths = depths[positive_depth]
    if points_camera.size == 0:
        return (
            np_mod.empty((0, 2), dtype=np_mod.float32),
            np_mod.empty((0,), dtype=np_mod.float32),
            positive_depth,
        )

    K = np_mod.asarray(camera_info_msg.k, dtype=np_mod.float64).reshape(3, 3)
    P = np_mod.asarray(camera_info_msg.p, dtype=np_mod.float64).reshape(3, 4)
    D = np_mod.asarray(camera_info_msg.d, dtype=np_mod.float64).reshape(-1)

    use_projection_matrix = np_mod.any(np_mod.abs(P[:3, :3]) > 1e-9)
    if use_projection_matrix:
        fx = P[0, 0]
        fy = P[1, 1]
        cx = P[0, 2]
        cy = P[1, 2]
        u = fx * (points_camera[:, 0] / depths) + cx
        v = fy * (points_camera[:, 1] / depths) + cy
        pixels = np_mod.stack([u, v], axis=1).astype(np_mod.float32)
        return pixels, depths, positive_depth

    if np_mod.allclose(D, 0.0):
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        u = fx * (points_camera[:, 0] / depths) + cx
        v = fy * (points_camera[:, 1] / depths) + cy
        pixels = np_mod.stack([u, v], axis=1).astype(np_mod.float32)
        return pixels, depths, positive_depth

    cv = require_cv2()
    image_points, _ = cv.projectPoints(
        points_camera.reshape(-1, 1, 3).astype(np_mod.float64),
        np_mod.zeros(3, dtype=np_mod.float64),
        np_mod.zeros(3, dtype=np_mod.float64),
        K,
        D,
    )
    return image_points.reshape(-1, 2).astype(np_mod.float32), depths, positive_depth


def range_colors_bgr(depths: np.ndarray, max_range_m: float) -> np.ndarray:
    np_mod = require_numpy()
    cv = require_cv2()

    if depths.size == 0:
        return np_mod.empty((0, 3), dtype=np_mod.uint8)

    norm = np_mod.clip(depths / max(max_range_m, 1e-6), 0.0, 1.0)
    hsv = np_mod.zeros((depths.shape[0], 1, 3), dtype=np_mod.uint8)
    hsv[:, 0, 0] = ((1.0 - norm) * 120.0).astype(np_mod.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return bgr[:, 0, :]


def index_colors_bgr(indices: np.ndarray, total_beams: int) -> np.ndarray:
    np_mod = require_numpy()
    cv = require_cv2()

    if indices.size == 0:
        return np_mod.empty((0, 3), dtype=np_mod.uint8)

    denom = max(int(total_beams) - 1, 1)
    norm_u8 = np_mod.clip(
        np_mod.round(indices.astype(np_mod.float32) / float(denom) * 255.0),
        0.0,
        255.0,
    ).astype(np_mod.uint8)
    grayscale = norm_u8.reshape(-1, 1)
    colormap = getattr(cv, "COLORMAP_TURBO", cv.COLORMAP_JET)
    bgr = cv.applyColorMap(grayscale, colormap)
    return bgr[:, 0, :]


def special_index_map(total_beams: int) -> dict[str, int]:
    last_index = max(int(total_beams) - 1, 0)
    return {
        "start": 0,
        "mid": last_index // 2,
        "end": last_index,
    }


def describe_index_color(index_value: int, total_beams: int) -> tuple[int, int, int]:
    color_bgr = index_colors_bgr(np.asarray([index_value], dtype=np.float32), total_beams)[0]
    return int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0])


def resolve_highlight_requests(highlight_args: list[str], total_beams: int) -> list[tuple[str, int]]:
    resolved: list[tuple[str, int]] = []
    seen_indices: set[int] = set()
    named = special_index_map(total_beams)

    for item in highlight_args:
        key = str(item).strip().lower()
        if not key:
            continue
        if key in named:
            index_value = named[key]
            label = f"{key}:{index_value}"
        else:
            try:
                index_value = int(key)
            except ValueError as exc:
                raise ValueError(f"Invalid --highlight-index value: {item}") from exc
            if index_value < 0 or index_value >= total_beams:
                raise ValueError(
                    f"--highlight-index out of range: {index_value} (total_beams={total_beams})"
                )
            label = f"idx:{index_value}"

        if index_value in seen_indices:
            continue
        seen_indices.add(index_value)
        resolved.append((label, index_value))

    return resolved


def draw_projection_overlay(
    image_rgb: np.ndarray,
    pixels: np.ndarray,
    depths: np.ndarray,
    point_colors_bgr: np.ndarray,
    point_indices: np.ndarray,
    scan_sample: HeaderStampedMessage,
    camera_info_sample: HeaderStampedMessage,
    image_sample: HeaderStampedMessage,
    color_mode: str,
    point_radius: int,
    highlight_requests: list[tuple[str, int]],
) -> np.ndarray:
    cv = require_cv2()
    np_mod = require_numpy()

    canvas = cv.cvtColor(image_rgb.copy(), cv.COLOR_RGB2BGR)
    image_h, image_w = canvas.shape[:2]

    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image_w)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image_h)
    )
    for (u, v), color in zip(pixels[inside], point_colors_bgr[inside]):
        cv.circle(
            canvas,
            (int(round(float(u))), int(round(float(v)))),
            max(1, point_radius),
            tuple(int(c) for c in color),
            -1,
            lineType=cv.LINE_AA,
        )

    if highlight_requests and point_indices.size > 0:
        for label, target_index in highlight_requests:
            nearest_idx = int(np_mod.argmin(np_mod.abs(point_indices - float(target_index))))
            if not inside[nearest_idx]:
                continue
            u, v = pixels[nearest_idx]
            center = (int(round(float(u))), int(round(float(v))))
            color = tuple(int(c) for c in point_colors_bgr[nearest_idx])
            cv.circle(
                canvas,
                center,
                max(4, point_radius + 3),
                (255, 255, 255),
                2,
                lineType=cv.LINE_AA,
            )
            cv.circle(
                canvas,
                center,
                max(2, point_radius + 1),
                color,
                2,
                lineType=cv.LINE_AA,
            )
            text_origin = (center[0] + 8, max(20, center[1] - 8))
            cv.putText(canvas, label, text_origin, cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(canvas, label, text_origin, cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv.LINE_AA)

    info_lines = [
        f"image frame={image_sample.frame_index} topic ts={image_sample.header_stamp_ns}",
        f"scan frame={scan_sample.frame_index} topic ts={scan_sample.header_stamp_ns}",
        f"camera_info ts={camera_info_sample.header_stamp_ns} frame={camera_info_sample.frame_id}",
        f"projected points={int(np_mod.count_nonzero(inside))}/{len(pixels)} color_mode={color_mode}",
    ]

    y = 26
    for line in info_lines:
        cv.putText(canvas, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(canvas, line, (12, y), cv.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1, cv.LINE_AA)
        y += 24

    return canvas


def build_single_projection_frame(
    image_sample: HeaderStampedMessage,
    scans: list[HeaderStampedMessage],
    camera_infos: list[HeaderStampedMessage],
    tf_static_msgs: list[object],
    args: argparse.Namespace,
) -> np.ndarray:
    match_stamp_ns = choose_matching_stamp(image_sample, args.use_recorded_timestamp)
    scan_sample = select_nearest_sample(scans, match_stamp_ns, args.use_recorded_timestamp)
    camera_info_sample = select_nearest_sample(camera_infos, match_stamp_ns, args.use_recorded_timestamp)

    camera_optical_frame = camera_info_sample.frame_id or image_sample.frame_id or "camera_infra1_optical_frame"
    lidar_frame = scan_sample.frame_id or args.lidar_frame
    camera_side = infer_camera_side(camera_optical_frame, args.image_topic, args.camera_side)
    graph = merged_tf_graph(tf_static_msgs, args, camera_optical_frame, camera_side)
    transform_camera_from_lidar = resolve_transform(graph, lidar_frame, camera_optical_frame)

    points_lidar, point_indices, total_beams = scan_to_points_lidar(
        scan_sample.msg, args.min_range, args.max_range
    )
    pixels, depths, positive_depth = project_points_to_image(
        points_lidar, transform_camera_from_lidar, camera_info_sample.msg
    )
    visible_point_indices = point_indices[positive_depth]
    if args.color_mode == "index":
        point_colors_bgr = index_colors_bgr(visible_point_indices, total_beams)
        highlight_requests = resolve_highlight_requests(args.highlight_index, total_beams)
    else:
        point_colors_bgr = range_colors_bgr(depths, args.max_range)
        highlight_requests = []

    return draw_projection_overlay(
        image_sample.msg,
        pixels,
        depths,
        point_colors_bgr,
        visible_point_indices,
        scan_sample,
        camera_info_sample,
        image_sample,
        args.color_mode,
        args.point_radius,
        highlight_requests,
    )


def save_single_frame(frame_bgr: np.ndarray, output_path: str | None) -> Path | None:
    if output_path is None:
        return None
    cv = require_cv2()
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv.imwrite(str(out_path), frame_bgr):
        raise RuntimeError(f"Failed to save output image: {out_path}")
    return out_path


def export_video(
    frames_bgr: list[np.ndarray],
    output_path: str,
    video_fps: float,
) -> Path:
    cv = require_cv2()
    if video_fps <= 0:
        raise ValueError("--video-fps must be > 0")
    if not frames_bgr:
        raise RuntimeError("No frames available for video export.")

    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames_bgr[0].shape[:2]
    writer = cv.VideoWriter(
        str(out_path),
        cv.VideoWriter_fourcc(*"mp4v"),
        float(video_fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    try:
        for frame in frames_bgr:
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Video frames must all share the same size.")
            writer.write(frame)
    finally:
        writer.release()
    return out_path


def main() -> None:
    args = parse_args()
    bag_path = normalize_bag_path(args.bag)
    camera_info_topic = args.camera_info_topic or derive_camera_info_topic(args.image_topic)

    images, scans, camera_infos, tf_static_msgs, image_frame_count = collect_projection_inputs(
        bag_path=bag_path,
        image_topic=args.image_topic,
        camera_info_topic=camera_info_topic,
        scan_topic=args.scan_topic,
        tf_static_topic=args.tf_static_topic,
        args=args,
    )

    if args.color_mode == "index" and args.print_index_colors:
        total_beams = len(scans[0].msg.ranges)
        for name, index_value in special_index_map(total_beams).items():
            rgb = describe_index_color(index_value, total_beams)
            print(f"[INFO] index_color {name} index={index_value} rgb={rgb}")

    if args.video_output:
        frames_bgr = [
            build_single_projection_frame(image_sample, scans, camera_infos, tf_static_msgs, args)
            for image_sample in images
        ]
        out_path = export_video(frames_bgr, args.video_output, args.video_fps)
        print(f"[INFO] bag={bag_path}")
        print(f"[INFO] image_topic={args.image_topic} selected_frames={len(images)} total_image_frames={image_frame_count}")
        print(f"[INFO] scan_topic={args.scan_topic} camera_info_topic={camera_info_topic}")
        print(f"[INFO] output={out_path}")
        return

    image_sample = images[-1] if args.frame < 0 else images[0]
    frame_bgr = build_single_projection_frame(image_sample, scans, camera_infos, tf_static_msgs, args)
    out_path = save_single_frame(frame_bgr, args.output)

    print(f"[INFO] bag={bag_path}")
    print(f"[INFO] image_topic={args.image_topic} frame={image_sample.frame_index} total_image_frames={image_frame_count}")
    print(f"[INFO] scan_topic={args.scan_topic} camera_info_topic={camera_info_topic}")
    if out_path is not None:
        print(f"[INFO] output={out_path}")

    if not args.no_show:
        cv = require_cv2()
        cv.imshow("lidar_camera_projection", frame_bgr)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()

import argparse
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def _decode_raw_image_to_rgb(msg) -> Optional[np.ndarray]:
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = msg.encoding.lower()
    step = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    if step <= 0 or raw.size < height * step:
        return None
    raw = raw[: height * step].reshape(height, step)

    if encoding == "rgb8":
        return raw[:, : width * 3].reshape(height, width, 3).copy()
    if encoding in ("bgr8", "8uc3"):
        return cv2.cvtColor(raw[:, : width * 3].reshape(height, width, 3), cv2.COLOR_BGR2RGB)
    if encoding == "rgba8":
        return cv2.cvtColor(raw[:, : width * 4].reshape(height, width, 4), cv2.COLOR_RGBA2RGB)
    if encoding == "bgra8":
        return cv2.cvtColor(raw[:, : width * 4].reshape(height, width, 4), cv2.COLOR_BGRA2RGB)
    if encoding in ("mono8", "8uc1"):
        return cv2.cvtColor(raw[:, :width], cv2.COLOR_GRAY2RGB)
    if encoding in ("mono16", "16uc1", "16sc1"):
        raw16 = np.frombuffer(msg.data, dtype=np.uint16)
        if raw16.size < height * width:
            return None
        img8 = (raw16[: height * width].reshape(height, width) / 256.0).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)
    if encoding in ("yuyv", "yuyv422", "yuv422", "uyvy"):
        img = raw[:, : width * 2].reshape(height, width, 2)
        code = cv2.COLOR_YUV2RGB_UYVY if encoding == "uyvy" else cv2.COLOR_YUV2RGB_YUY2
        return cv2.cvtColor(img, code)

    return None


def _decode_compressed_image_to_rgb(msg) -> Optional[np.ndarray]:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_images_as_png(images: List[np.ndarray], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    for i, image_rgb in enumerate(images):
        cv2.imwrite(str(image_dir / f"{i:06d}.png"), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def _stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def _pose_to_xyyaw(pose) -> Tuple[float, float, float]:
    return (
        float(pose.position.x),
        float(pose.position.y),
        _yaw_from_quat(pose.orientation),
    )


def _append_pose_msg(msg, msgtype: str, timestamp: int, pose_rows: List[Tuple[int, float, float, float]]) -> None:
    if msgtype == "geometry_msgs/msg/PoseStamped":
        stamp = _stamp_to_ns(msg.header.stamp) if getattr(msg, "header", None) else timestamp
        pose_rows.append((stamp, *_pose_to_xyyaw(msg.pose)))
    elif msgtype == "geometry_msgs/msg/PoseWithCovarianceStamped":
        stamp = _stamp_to_ns(msg.header.stamp) if getattr(msg, "header", None) else timestamp
        pose_rows.append((stamp, *_pose_to_xyyaw(msg.pose.pose)))
    elif msgtype == "nav_msgs/msg/Odometry":
        stamp = _stamp_to_ns(msg.header.stamp) if getattr(msg, "header", None) else timestamp
        pose_rows.append((stamp, *_pose_to_xyyaw(msg.pose.pose)))
    elif msgtype == "nav_msgs/msg/Path":
        for i, pose_stamped in enumerate(msg.poses):
            stamp = _stamp_to_ns(pose_stamped.header.stamp)
            if stamp == 0:
                stamp = timestamp + i
            pose_rows.append((stamp, *_pose_to_xyyaw(pose_stamped.pose)))


def _trajectory_from_pose_index(
    poses: np.ndarray,
    start_idx: int,
    target_distances: Sequence[float],
) -> Optional[np.ndarray]:
    future = poses[start_idx:, 1:3]
    if len(future) < 2:
        return None

    deltas = np.diff(future, axis=0)
    seg_lengths = np.linalg.norm(deltas, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if cumulative[-1] < target_distances[-1]:
        return None

    global_points = []
    for distance in target_distances:
        idx = int(np.searchsorted(cumulative, distance, side="left"))
        if idx == 0:
            global_points.append(future[0])
            continue
        prev_d = cumulative[idx - 1]
        next_d = cumulative[idx]
        alpha = 0.0 if next_d <= prev_d else (distance - prev_d) / (next_d - prev_d)
        global_points.append(future[idx - 1] + alpha * (future[idx] - future[idx - 1]))

    x0, y0, yaw0 = poses[start_idx, 1], poses[start_idx, 2], poses[start_idx, 3]
    cos_yaw = np.cos(yaw0)
    sin_yaw = np.sin(yaw0)
    local_points = []
    for x, y in global_points:
        dx = x - x0
        dy = y - y0
        local_x = cos_yaw * dx + sin_yaw * dy
        local_y = -sin_yaw * dx + cos_yaw * dy
        local_points.append([local_x, local_y])

    return np.asarray(local_points, dtype=np.float32)


def extract_and_save_per_bag(
    bag_path: Path,
    output_dir: str,
    image_topic: str,
    pose_topic: str,
    image_storage: str,
    target_distances: Sequence[float],
    max_pose_time_diff_ns: int,
) -> None:
    pid = os.getpid()
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_data, image_times = [], []
    pose_rows: List[Tuple[int, float, float, float]] = []

    try:
        with AnyReader([bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic in (image_topic, pose_topic)]

            for conn, timestamp, raw in reader.messages(connections=connections):
                msg = reader.deserialize(raw, conn.msgtype)

                if conn.topic == image_topic:
                    image_rgb = None
                    if conn.msgtype == "sensor_msgs/msg/Image":
                        image_rgb = _decode_raw_image_to_rgb(msg)
                    elif conn.msgtype == "sensor_msgs/msg/CompressedImage":
                        image_rgb = _decode_compressed_image_to_rgb(msg)

                    if image_rgb is not None:
                        image_data.append(image_rgb.astype(np.uint8))
                        image_times.append(timestamp)
                elif conn.topic == pose_topic:
                    _append_pose_msg(msg, conn.msgtype, timestamp, pose_rows)

    except Exception as e:
        print(f"[PID:{pid} ERROR] {bag_name}: Failed to read bag file. {e}")
        return

    if len(image_data) == 0 or len(pose_rows) == 0:
        print(f"[PID:{pid} WARN] Skip {bag_name}: images={len(image_data)}, poses={len(pose_rows)}")
        return

    sorted_pose_rows = sorted(pose_rows, key=lambda row: row[0])
    deduped_pose_rows = []
    for row in sorted_pose_rows:
        if deduped_pose_rows and row == deduped_pose_rows[-1]:
            continue
        deduped_pose_rows.append(row)
    poses = np.asarray(deduped_pose_rows, dtype=np.float64)
    image_times_np = np.asarray(image_times, dtype=np.int64)

    synced_images, trajectories = [], []
    pose_times = poses[:, 0]
    for i, image_time in enumerate(image_times_np):
        idx_pose = int(np.searchsorted(pose_times, image_time))
        candidates = [idx for idx in (idx_pose - 1, idx_pose) if 0 <= idx < len(pose_times)]
        if not candidates:
            continue
        nearest_idx = min(candidates, key=lambda idx: abs(pose_times[idx] - image_time))
        if abs(pose_times[nearest_idx] - image_time) > max_pose_time_diff_ns:
            continue

        trajectory = _trajectory_from_pose_index(poses, nearest_idx, target_distances)
        if trajectory is None:
            continue

        synced_images.append(image_data[i])
        trajectories.append(trajectory)

    if not synced_images:
        print(f"[PID:{pid} WARN] Skip {bag_name}: no images had enough future path.")
        return

    if image_storage == "npy":
        np.save(out_dir / "images.npy", np.asarray(synced_images, dtype=np.uint8))
    elif image_storage == "png":
        _save_images_as_png(synced_images, out_dir / "images")
    else:
        print(f"[PID:{pid} ERROR] Unsupported image_storage: {image_storage}")
        return

    np.save(out_dir / "trajectories.npy", np.asarray(trajectories, dtype=np.float32))
    np.save(out_dir / "target_distances.npy", np.asarray(target_distances, dtype=np.float32))

    print(f"[PID:{pid} SAVE] {bag_name}: {len(synced_images)} samples saved to {out_dir}")


def _parse_target_distances(args) -> List[float]:
    if args.target_distances:
        distances = [float(v) for v in args.target_distances]
    else:
        distances = np.linspace(args.min_distance, args.max_distance, args.num_points).tolist()
    if any(d <= 0.0 for d in distances) or sorted(distances) != distances:
        raise ValueError("Target distances must be positive and sorted ascending.")
    return distances


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract camera images and future local trajectories from rosbags."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bags_dir", help="Path to directory containing rosbag folders (searches recursively)")
    group.add_argument("--seq_dirs", nargs="+", help="List of specific sequence directories to process")

    parser.add_argument("--outdir", required=True, help="Output root directory")
    parser.add_argument("--image_topic", default="/realsense2_camera/color/image_raw", help="Image topic name")
    parser.add_argument(
        "--pose_topic",
        default="/visual_slam/tracking/odometry",
        help=(
            "Pose source topic. Supported: nav_msgs/Path, nav_msgs/Odometry, "
            "geometry_msgs/PoseStamped, geometry_msgs/PoseWithCovarianceStamped."
        ),
    )
    parser.add_argument("--image_storage", default="npy", choices=["npy", "png"])
    parser.add_argument("--num_points", type=int, default=20)
    parser.add_argument("--min_distance", type=float, default=0.5)
    parser.add_argument("--max_distance", type=float, default=10.0)
    parser.add_argument("--target_distances", type=float, nargs="+", default=None)
    parser.add_argument("--max_pose_time_diff", type=float, default=0.2, help="Seconds")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

    target_distances = _parse_target_distances(args)
    max_pose_time_diff_ns = int(args.max_pose_time_diff * 1_000_000_000)

    bag_dirs = []
    if args.bags_dir:
        bags_dir_path = Path(args.bags_dir).expanduser().resolve()
        for p in bags_dir_path.rglob("metadata.yaml"):
            bag_dirs.append(p.parent)
    else:
        for seq_path_str in args.seq_dirs:
            seq_path = Path(seq_path_str).expanduser().resolve()
            if (seq_path / "metadata.yaml").is_file():
                bag_dirs.append(seq_path)

    if not bag_dirs:
        print("[ERROR] No valid rosbag directories found.")
        return

    print(f"[INFO] Found {len(bag_dirs)} rosbag directories to process.")
    print(f"[INFO] Pose topic: {args.pose_topic}")
    print(f"[INFO] Target distances: {target_distances}")

    tasks = [
        (
            p,
            args.outdir,
            args.image_topic,
            args.pose_topic,
            args.image_storage,
            target_distances,
            max_pose_time_diff_ns,
        )
        for p in sorted(bag_dirs)
    ]

    num_workers = args.workers if args.workers else min(max(1, (os.cpu_count() or 1) - 1), 8)
    print(f"[INFO] Starting parallel processing with {num_workers} workers...")

    try:
        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(extract_and_save_per_bag, tasks)
        print("[INFO] All processing finished.")
    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()

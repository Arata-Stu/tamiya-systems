import argparse
import multiprocessing
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from rosbags.highlevel import AnyReader


def _decode_raw_image_to_rgb(msg) -> Optional[np.ndarray]:
    """sensor_msgs/Image を RGB uint8 (H, W, 3) に変換する。"""
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = msg.encoding.lower()
    step = int(msg.step)
    raw = np.frombuffer(msg.data, dtype=np.uint8)

    if step <= 0:
        return None

    expected = height * step
    if raw.size < expected:
        return None
    raw = raw[:expected].reshape(height, step)

    if encoding in ("rgb8",):
        img = raw[:, : width * 3].reshape(height, width, 3)
        return img.copy()

    if encoding in ("bgr8", "8uc3"):
        img = raw[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if encoding in ("rgba8",):
        img = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if encoding in ("bgra8",):
        img = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)

    if encoding in ("mono8", "8uc1"):
        mono = raw[:, :width]
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2RGB)

    if encoding in ("mono16", "16uc1", "16sc1"):
        raw16 = np.frombuffer(msg.data, dtype=np.uint16)
        if raw16.size < height * width:
            return None
        img16 = raw16[: height * width].reshape(height, width)
        img8 = (img16 / 256.0).astype(np.uint8)
        return cv2.cvtColor(img8, cv2.COLOR_GRAY2RGB)

    if encoding in ("yuyv", "yuyv422", "yuv422", "uyvy"):
        img = raw[:, : width * 2].reshape(height, width, 2)
        code = cv2.COLOR_YUV2RGB_YUY2 if encoding != "uyvy" else cv2.COLOR_YUV2RGB_UYVY
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
        out_path = image_dir / f"{i:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))


def extract_and_save_per_bag(
    bag_path: Path,
    output_dir: str,
    image_topic: str,
    cmd_topic: str,
    image_storage: str,
) -> None:
    pid = os.getpid()
    bag_path = Path(bag_path).expanduser().resolve()
    bag_name = bag_path.name
    out_dir = Path(output_dir) / bag_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image_data, image_times = [], []
    cmd_data, cmd_times = [], []

    try:
        with AnyReader([bag_path]) as reader:
            connections = [c for c in reader.connections if c.topic in (image_topic, cmd_topic)]

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

                elif conn.topic == cmd_topic and conn.msgtype == "ackermann_msgs/msg/AckermannDriveStamped":
                    cmd_data.append(np.array([msg.drive.steering_angle, msg.drive.speed], dtype=np.float32))
                    cmd_times.append(timestamp)

    except Exception as e:
        print(f"[PID:{pid} ERROR] {bag_name}: Failed to read bag file. {e}")
        return

    if len(image_data) == 0 or len(cmd_data) == 0:
        print(f"[PID:{pid} WARN] Skip {bag_name}: images={len(image_data)}, cmds={len(cmd_data)}")
        return

    image_times = np.array(image_times)
    cmd_data = np.array(cmd_data, dtype=np.float32)
    cmd_times = np.array(cmd_times)

    synced_images, synced_steers, synced_speeds = [], [], []

    for i, image_time in enumerate(image_times):
        idx_cmd = int(np.argmin(np.abs(cmd_times - image_time)))
        synced_images.append(image_data[i])
        synced_steers.append(cmd_data[idx_cmd][0])
        synced_speeds.append(cmd_data[idx_cmd][1])

    if image_storage == "npy":
        np.save(out_dir / "images.npy", np.array(synced_images, dtype=np.uint8))
    elif image_storage == "png":
        _save_images_as_png(synced_images, out_dir / "images")
    else:
        print(f"[PID:{pid} ERROR] Unsupported image_storage: {image_storage}")
        return

    np.save(out_dir / "steers.npy", np.array(synced_steers, dtype=np.float32))
    np.save(out_dir / "speeds.npy", np.array(synced_speeds, dtype=np.float32))

    print(f"[PID:{pid} SAVE] {bag_name}: {len(synced_images)} samples saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract and synchronize camera images and AckermannDriveStamped data from rosbags."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bags_dir", help="Path to directory containing rosbag folders (searches recursively)")
    group.add_argument("--seq_dirs", nargs="+", help="List of specific sequence directories to process")

    parser.add_argument("--outdir", required=True, help="Output root directory")
    parser.add_argument("--image_topic", default="/camera/left/image_raw", help="Image topic name")
    parser.add_argument("--cmd_topic", default="/jetracer/cmd_drive", help="AckermannDriveStamped topic name")
    parser.add_argument(
        "--image_storage",
        default="npy",
        choices=["npy", "png"],
        help="Image storage format. npy is faster for training, png is better for manual inspection.",
    )
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers")
    args = parser.parse_args()

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

    tasks = [(p, args.outdir, args.image_topic, args.cmd_topic, args.image_storage) for p in sorted(bag_dirs)]

    if args.workers:
        num_workers = args.workers
    else:
        num_workers = min(max(1, (os.cpu_count() or 1) - 1), 8)

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

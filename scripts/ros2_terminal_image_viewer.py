#!/usr/bin/env python3
"""View ROS 2 image topics directly in a kitty-compatible terminal."""

from __future__ import annotations

import argparse
import base64
import math
import signal
import struct
import sys
import time
import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image

try:
    import cv2
except ImportError:  # pragma: no cover - ROS container should have OpenCV.
    cv2 = None


ColorImage = np.ndarray


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    rows = []
    stride = width * 4
    for y in range(height):
        rows.append(b"\x00" + rgba[y * stride : (y + 1) * stride])
    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)),
            png_chunk(b"IDAT", zlib.compress(raw, 5)),
            png_chunk(b"IEND", b""),
        ]
    )


def raw_image_to_rgb(msg: Image) -> ColorImage:
    encoding = msg.encoding.lower()
    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = memoryview(msg.data)

    if encoding in ("rgb8", "bgr8"):
        row_bytes = width * 3
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, :row_bytes].reshape(height, width, 3)
        if encoding == "bgr8":
            return arr[:, :, ::-1].copy()
        return arr.copy()

    if encoding in ("rgba8", "bgra8"):
        row_bytes = width * 4
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, :row_bytes].reshape(height, width, 4)
        if encoding == "bgra8":
            return arr[:, :, [2, 1, 0]].copy()
        return arr[:, :, :3].copy()

    if encoding in ("mono8", "8uc1"):
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, :width]
        return np.repeat(arr[:, :, None], 3, axis=2)

    if encoding in ("mono16", "16uc1"):
        row_items = step // 2
        arr = np.frombuffer(data, dtype=np.uint16).reshape(height, row_items)[:, :width]
        scaled = normalize_numeric_image(arr)
        return np.repeat(scaled[:, :, None], 3, axis=2)

    if encoding == "32fc1":
        row_items = step // 4
        arr = np.frombuffer(data, dtype=np.float32).reshape(height, row_items)[:, :width]
        scaled = normalize_numeric_image(arr)
        return np.repeat(scaled[:, :, None], 3, axis=2)

    if cv2 is not None and encoding in ("yuv422", "yuyv"):
        row_bytes = width * 2
        arr = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, :row_bytes].reshape(height, width, 2)
        return cv2.cvtColor(arr, cv2.COLOR_YUV2RGB_YUY2)

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


def normalize_numeric_image(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    low, high = np.percentile(finite, [1.0, 99.0])
    if math.isclose(float(low), float(high)):
        high = low + 1.0
    scaled = np.clip((arr.astype(np.float32) - low) * 255.0 / (high - low), 0, 255)
    return scaled.astype(np.uint8)


def compressed_image_to_rgb(msg: CompressedImage) -> ColorImage:
    if cv2 is None:
        raise RuntimeError("compressed image topics require opencv-python")
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode compressed image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_to_fit(rgb: ColorImage, max_width: int, max_height: int) -> ColorImage:
    height, width = rgb.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 0.999:
        return rgb
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    if cv2 is not None:
        return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)

    ys = (np.arange(new_size[1]) / scale).astype(np.int32).clip(0, height - 1)
    xs = (np.arange(new_size[0]) / scale).astype(np.int32).clip(0, width - 1)
    return rgb[ys[:, None], xs[None, :], :]


def rgb_to_png(rgb: ColorImage) -> bytes:
    if cv2 is not None:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".png", bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if ok:
            return bytes(encoded)

    alpha = np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)
    rgba = np.concatenate([rgb, alpha], axis=2)
    height, width = rgba.shape[:2]
    return encode_png_rgba(width, height, rgba.tobytes())


def write_kitty_image(png: bytes, width: int, height: int) -> None:
    payload = base64.b64encode(png)
    chunk_size = 4096
    out = sys.stdout.buffer
    for start in range(0, len(payload), chunk_size):
        chunk = payload[start : start + chunk_size]
        more = 1 if start + chunk_size < len(payload) else 0
        if start == 0:
            header = f"\033_Ga=T,f=100,t=d,s={width},v={height},m={more};".encode()
        else:
            header = f"\033_Gm={more};".encode()
        out.write(header + chunk + b"\033\\")
    out.flush()


@dataclass
class ViewerStats:
    frames: int = 0
    dropped: int = 0
    last_error: str = ""
    last_stamp: float = 0.0


class TerminalImageViewer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("terminal_image_viewer")
        self.args = args
        self.stats = ViewerStats()
        self.last_render_time = 0.0
        self.min_period = 1.0 / max(args.max_fps, 0.1)

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE,
        )

        if args.compressed:
            self.subscription = self.create_subscription(CompressedImage, args.topic, self.on_compressed_image, qos)
        else:
            self.subscription = self.create_subscription(Image, args.topic, self.on_image, qos)

        self.get_logger().info(f"subscribed to {args.topic} ({'CompressedImage' if args.compressed else 'Image'})")

    def on_image(self, msg: Image) -> None:
        self.render(lambda: raw_image_to_rgb(msg), f"{msg.width}x{msg.height} {msg.encoding}")

    def on_compressed_image(self, msg: CompressedImage) -> None:
        self.render(lambda: compressed_image_to_rgb(msg), msg.format or "compressed")

    def render(self, decode: Callable[[], ColorImage], label: str) -> None:
        now = time.monotonic()
        if now - self.last_render_time < self.min_period:
            self.stats.dropped += 1
            return
        self.last_render_time = now

        try:
            rgb = decode()
            rgb = resize_to_fit(rgb, self.args.width, self.args.height)
            png = rgb_to_png(rgb)
            height, width = rgb.shape[:2]
            sys.stdout.write("\033[H\033[2J")
            sys.stdout.write(f"topic={self.args.topic} frame={self.stats.frames + 1} size={width}x{height} source={label}\n")
            sys.stdout.flush()
            write_kitty_image(png, width, height)
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.stats.frames += 1
            self.stats.last_stamp = now
        except Exception as exc:  # noqa: BLE001 - keep viewer alive during topic experiments.
            self.stats.last_error = str(exc)
            self.get_logger().warn(self.stats.last_error)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/image", help="image topic to subscribe")
    parser.add_argument("--compressed", action="store_true", help="subscribe as sensor_msgs/CompressedImage")
    parser.add_argument("--width", type=int, default=960, help="maximum rendered image width in pixels")
    parser.add_argument("--height", type=int, default=540, help="maximum rendered image height in pixels")
    parser.add_argument("--max-fps", type=float, default=5.0, help="terminal update limit")
    parser.add_argument("--best-effort", action="store_true", help="use best-effort QoS, useful for camera topics")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TerminalImageViewer(args)
    stop = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    sys.stdout.write("\033[?1049h\033[?25l\033[H\033[2J")
    sys.stdout.flush()
    try:
        while rclpy.ok() and not stop:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""View a ROS 2 LaserScan topic directly in a kitty-compatible terminal."""

from __future__ import annotations

import argparse
import base64
import math
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan


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


class TerminalScanViewer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("terminal_scan_viewer")
        self.args = args
        self.latest_scan: LaserScan | None = None
        self.frame_count = 0
        self.last_scan_monotonic = 0.0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(LaserScan, args.topic, self.on_scan, qos)
        self.create_timer(1.0 / max(args.max_fps, 0.2), self.render)
        self.get_logger().info(f"subscribed to {args.topic} (LaserScan)")

    def on_scan(self, msg: LaserScan) -> None:
        self.latest_scan = msg
        self.last_scan_monotonic = time.monotonic()

    def render(self) -> None:
        if self.latest_scan is None:
            sys.stdout.write(f"\033[H\033[2Jwaiting for LaserScan on {self.args.topic}...\n")
            sys.stdout.flush()
            return

        scan = self.latest_scan
        canvas = np.full((self.args.height, self.args.width, 3), (24, 25, 28), dtype=np.uint8)
        center = (self.args.width // 2, int(self.args.height * 0.58))
        max_range = self.args.range
        if max_range <= 0:
            finite = [float(r) for r in scan.ranges if math.isfinite(float(r)) and float(r) > scan.range_min]
            max_range = min(float(scan.range_max), np.percentile(finite, 98) * 1.08) if finite else float(scan.range_max)
            if not math.isfinite(max_range) or max_range <= scan.range_min:
                max_range = max(1.0, float(scan.range_max) if math.isfinite(scan.range_max) else 10.0)

        radius_px = min(self.args.width * 0.45, self.args.height * 0.50)
        meters_to_px = radius_px / max_range

        self.draw_grid(canvas, center, meters_to_px, max_range)
        points, invalid_count, clipped_count = self.scan_points(scan, center, meters_to_px, max_range)

        if self.args.connect and len(points) >= 2:
            cv2.polylines(canvas, [np.asarray(points, dtype=np.int32)], False, (50, 165, 255), 1, cv2.LINE_AA)
        for point in points:
            cv2.circle(canvas, point, self.args.point_radius, (40, 190, 255), -1, cv2.LINE_AA)

        cv2.circle(canvas, center, 6, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.arrowedLine(canvas, center, (center[0] + 42, center[1]), (70, 235, 125), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.putText(canvas, "+x", (center[0] + 50, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (70, 235, 125), 1, cv2.LINE_AA)

        self.frame_count += 1
        age = time.monotonic() - self.last_scan_monotonic
        status = (
            f"topic={self.args.topic} frame={self.frame_count} scan_frame={scan.header.frame_id or '?'} "
            f"points={len(points)} invalid={invalid_count} clipped={clipped_count} range={max_range:.2f}m age={age:.2f}s"
        )
        cv2.rectangle(canvas, (0, 0), (self.args.width, 30), (16, 17, 20), -1)
        cv2.putText(canvas, status[:180], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression])
        if not ok:
            self.get_logger().warn("failed to encode terminal scan frame")
            return
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()
        write_kitty_image(bytes(encoded), self.args.width, self.args.height)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def draw_grid(self, canvas: np.ndarray, center: tuple[int, int], meters_to_px: float, max_range: float) -> None:
        grid_color = (64, 67, 74)
        axis_color = (90, 96, 106)
        label_color = (172, 176, 184)
        step = self.args.grid_step
        if step <= 0:
            step = nice_grid_step(max_range)

        r = step
        while r <= max_range + 1e-6:
            radius = int(round(r * meters_to_px))
            cv2.circle(canvas, center, radius, grid_color, 1, cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{r:g}m",
                (center[0] + radius + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                label_color,
                1,
                cv2.LINE_AA,
            )
            r += step

        for deg in range(-150, 181, 30):
            a = math.radians(deg)
            end = (
                int(round(center[0] + math.cos(a) * max_range * meters_to_px)),
                int(round(center[1] - math.sin(a) * max_range * meters_to_px)),
            )
            cv2.line(canvas, center, end, axis_color if deg in (0, 90, -90, 180) else grid_color, 1, cv2.LINE_AA)

    def scan_points(
        self,
        scan: LaserScan,
        center: tuple[int, int],
        meters_to_px: float,
        max_range: float,
    ) -> tuple[list[tuple[int, int]], int, int]:
        points: list[tuple[int, int]] = []
        invalid_count = 0
        clipped_count = 0
        angle = float(scan.angle_min)
        stride = max(1, self.args.stride)
        for idx, raw_range in enumerate(scan.ranges):
            if idx % stride != 0:
                angle += scan.angle_increment
                continue
            r = float(raw_range)
            if not math.isfinite(r) or r < scan.range_min:
                invalid_count += 1
                angle += scan.angle_increment
                continue
            if r > max_range:
                clipped_count += 1
                if not self.args.show_clipped:
                    angle += scan.angle_increment
                    continue
                r = max_range
            x = center[0] + math.cos(angle) * r * meters_to_px
            y = center[1] - math.sin(angle) * r * meters_to_px
            points.append((int(round(x)), int(round(y))))
            angle += scan.angle_increment
        return points, invalid_count, clipped_count


def nice_grid_step(max_range: float) -> float:
    if max_range <= 3.0:
        return 0.5
    if max_range <= 8.0:
        return 1.0
    if max_range <= 20.0:
        return 2.0
    return 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/scan")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=700)
    parser.add_argument("--max-fps", type=float, default=5.0)
    parser.add_argument("--best-effort", action="store_true")
    parser.add_argument("--range", type=float, default=0.0, help="visible range in meters; 0 means auto")
    parser.add_argument("--grid-step", type=float, default=0.0, help="range grid step in meters; 0 means auto")
    parser.add_argument("--stride", type=int, default=1, help="draw every Nth scan point")
    parser.add_argument("--point-radius", type=int, default=2)
    parser.add_argument("--connect", action="store_true", help="connect adjacent scan points")
    parser.add_argument("--show-clipped", action="store_true", help="draw out-of-range points on the outer circle")
    parser.add_argument("--png-compression", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TerminalScanViewer(args)
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

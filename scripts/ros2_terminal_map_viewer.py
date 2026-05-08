#!/usr/bin/env python3
"""View a ROS 2 2D map and localization overlays in a kitty terminal."""

from __future__ import annotations

import argparse
import base64
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import LaserScan

try:
    import cv2
except ImportError as exc:  # pragma: no cover - target ROS image has OpenCV.
    raise SystemExit("ros2_terminal_map_viewer.py requires opencv-python") from exc

try:
    import tf2_ros
except ImportError:  # pragma: no cover - keep map/pose viewer usable without TF.
    tf2_ros = None


Color = tuple[int, int, int]


def yaw_from_quat(q: object) -> float:
    x = float(q.x)
    y = float(q.y)
    z = float(q.z)
    w = float(q.w)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def yaw_from_transform(transform: object) -> float:
    return yaw_from_quat(transform.rotation)


def transform_xy_yaw(x: float, y: float, yaw: float, transform: object) -> tuple[float, float, float]:
    tx = float(transform.translation.x)
    ty = float(transform.translation.y)
    theta = yaw_from_transform(transform)
    c = math.cos(theta)
    s = math.sin(theta)
    return tx + c * x - s * y, ty + s * x + c * y, yaw + theta


def transform_xy(x: float, y: float, transform: object) -> tuple[float, float]:
    tx, ty, _ = transform_xy_yaw(x, y, 0.0, transform)
    return tx, ty


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
class MapState:
    frame_id: str
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    width: int
    height: int
    image_bgr: np.ndarray


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float
    frame_id: str
    stamp: object
    covariance: Optional[list[float]] = None


class TerminalMapViewer(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("terminal_map_viewer")
        self.args = args
        self.map_state: Optional[MapState] = None
        self.pose: Optional[Pose2D] = None
        self.particles: Optional[PoseArray] = None
        self.scan: Optional[LaserScan] = None
        self.path: Optional[Path] = None
        self.frame_count = 0
        self.last_status = ""

        self.tf_buffer = None
        self.tf_listener = None
        if tf2_ros is not None and not args.no_tf:
            self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=20.0))
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        default_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE,
        )

        if args.map_qos in ("transient_local", "both"):
            self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, map_qos)
        if args.map_qos in ("volatile", "both"):
            volatile_map_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, volatile_map_qos)
        if args.pose_topic:
            self.create_subscription(PoseWithCovarianceStamped, args.pose_topic, self.on_pose_cov, default_qos)
        if args.pose_stamped_topic:
            self.create_subscription(PoseStamped, args.pose_stamped_topic, self.on_pose_stamped, default_qos)
        if args.particles_topic:
            self.create_subscription(PoseArray, args.particles_topic, self.on_particles, sensor_qos)
        if args.scan_topic:
            self.create_subscription(LaserScan, args.scan_topic, self.on_scan, sensor_qos)
        if args.path_topic:
            self.create_subscription(Path, args.path_topic, self.on_path, default_qos)

        period = 1.0 / max(args.max_fps, 0.2)
        self.create_timer(period, self.render)
        self.get_logger().info(
            "subscribed: map=%s pose=%s pose_stamped=%s particles=%s scan=%s path=%s"
            % (
                args.map_topic,
                args.pose_topic,
                args.pose_stamped_topic,
                args.particles_topic or "-",
                args.scan_topic or "-",
                args.path_topic or "-",
            )
        )

    def on_map(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        image = np.zeros((msg.info.height, msg.info.width, 3), dtype=np.uint8)
        image[data < 0] = (178, 178, 178)
        image[data == 0] = (245, 245, 245)
        occupied = data > 0
        image[occupied] = np.clip(235 - data[occupied] * 2, 25, 235).astype(np.uint8)[:, None]
        image = np.flipud(image)

        self.map_state = MapState(
            frame_id=msg.header.frame_id or self.args.map_frame,
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            origin_yaw=yaw_from_quat(msg.info.origin.orientation),
            width=int(msg.info.width),
            height=int(msg.info.height),
            image_bgr=image,
        )

    def on_pose_cov(self, msg: PoseWithCovarianceStamped) -> None:
        pose = msg.pose.pose
        self.pose = Pose2D(
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw_from_quat(pose.orientation),
            frame_id=msg.header.frame_id or self.args.map_frame,
            stamp=msg.header.stamp,
            covariance=list(msg.pose.covariance),
        )

    def on_pose_stamped(self, msg: PoseStamped) -> None:
        pose = msg.pose
        self.pose = Pose2D(
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw_from_quat(pose.orientation),
            frame_id=msg.header.frame_id or self.args.map_frame,
            stamp=msg.header.stamp,
        )

    def on_particles(self, msg: PoseArray) -> None:
        self.particles = msg

    def on_scan(self, msg: LaserScan) -> None:
        self.scan = msg

    def on_path(self, msg: Path) -> None:
        self.path = msg

    def lookup_transform(self, source_frame: str, stamp: object) -> Optional[object]:
        if self.map_state is None or not source_frame or source_frame == self.map_state.frame_id:
            return None
        if self.tf_buffer is None:
            return None
        try:
            when = Time.from_msg(stamp) if stamp is not None else Time()
            return self.tf_buffer.lookup_transform(
                self.map_state.frame_id,
                source_frame,
                when,
                timeout=Duration(seconds=self.args.tf_timeout),
            ).transform
        except Exception as exc:  # noqa: BLE001 - rendering should survive missing TF.
            self.last_status = f"TF unavailable: {source_frame}->{self.map_state.frame_id}: {exc}"
            try:
                return self.tf_buffer.lookup_transform(
                    self.map_state.frame_id,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=self.args.tf_timeout),
                ).transform
            except Exception:
                return None

    def pose_in_map(self, pose: Pose2D) -> Optional[Pose2D]:
        if self.map_state is None:
            return None
        tf = self.lookup_transform(pose.frame_id, pose.stamp)
        if tf is None:
            if pose.frame_id and pose.frame_id != self.map_state.frame_id and not self.args.assume_same_frame:
                return None
            return pose
        x, y, yaw = transform_xy_yaw(pose.x, pose.y, pose.yaw, tf)
        return Pose2D(x=x, y=y, yaw=yaw, frame_id=self.map_state.frame_id, stamp=pose.stamp, covariance=pose.covariance)

    def world_to_map_px(self, x: float, y: float) -> tuple[int, int]:
        assert self.map_state is not None
        dx = x - self.map_state.origin_x
        dy = y - self.map_state.origin_y
        c = math.cos(-self.map_state.origin_yaw)
        s = math.sin(-self.map_state.origin_yaw)
        mx = (c * dx - s * dy) / self.map_state.resolution
        my = (s * dx + c * dy) / self.map_state.resolution
        return int(round(mx)), int(round(self.map_state.height - 1 - my))

    def map_px_to_view_px(self, px: int, py: int, scale: float, pad_x: int, pad_y: int) -> tuple[int, int]:
        return int(round(px * scale + pad_x)), int(round(py * scale + pad_y))

    def draw_polyline_world(self, canvas: np.ndarray, points: list[tuple[float, float]], color: Color, thickness: int, scale: float, pad_x: int, pad_y: int) -> None:
        if len(points) < 2:
            return
        pixels = [self.map_px_to_view_px(*self.world_to_map_px(x, y), scale, pad_x, pad_y) for x, y in points]
        cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)

    def draw_pose(self, canvas: np.ndarray, pose: Pose2D, scale: float, pad_x: int, pad_y: int) -> None:
        px, py = self.map_px_to_view_px(*self.world_to_map_px(pose.x, pose.y), scale, pad_x, pad_y)
        length = max(22, int(0.55 / max(self.map_state.resolution, 1e-6) * scale))
        end = (int(px + math.cos(pose.yaw) * length), int(py - math.sin(pose.yaw) * length))
        cv2.arrowedLine(canvas, (px, py), end, (30, 30, 245), max(2, int(4 * scale)), cv2.LINE_AA, tipLength=0.35)
        cv2.circle(canvas, (px, py), max(4, int(7 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), max(3, int(4 * scale)), (30, 30, 245), -1, cv2.LINE_AA)
        self.draw_covariance(canvas, pose, scale, pad_x, pad_y)

    def draw_covariance(self, canvas: np.ndarray, pose: Pose2D, scale: float, pad_x: int, pad_y: int) -> None:
        if not pose.covariance:
            return
        cov = np.asarray([[pose.covariance[0], pose.covariance[1]], [pose.covariance[6], pose.covariance[7]]], dtype=np.float64)
        if not np.all(np.isfinite(cov)):
            return
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 0.0)
        if vals.max() <= 1e-9:
            return
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]
        axes_m = 2.0 * np.sqrt(vals)
        axes_px = tuple(max(2, int(a / self.map_state.resolution * scale)) for a in axes_m)
        if axes_px[0] > max(canvas.shape[:2]) * 2:
            return
        angle = -math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
        center = self.map_px_to_view_px(*self.world_to_map_px(pose.x, pose.y), scale, pad_x, pad_y)
        overlay = canvas.copy()
        cv2.ellipse(overlay, center, axes_px, angle, 0, 360, (50, 90, 255), 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

    def draw_particles(self, canvas: np.ndarray, scale: float, pad_x: int, pad_y: int) -> None:
        if self.particles is None or self.map_state is None:
            return
        tf = self.lookup_transform(self.particles.header.frame_id, self.particles.header.stamp)
        for i, pose in enumerate(self.particles.poses):
            if i % max(1, self.args.particle_stride) != 0:
                continue
            x = float(pose.position.x)
            y = float(pose.position.y)
            yaw = yaw_from_quat(pose.orientation)
            if tf is not None:
                x, y, yaw = transform_xy_yaw(x, y, yaw, tf)
            elif self.particles.header.frame_id != self.map_state.frame_id and not self.args.assume_same_frame:
                continue
            px, py = self.map_px_to_view_px(*self.world_to_map_px(x, y), scale, pad_x, pad_y)
            cv2.circle(canvas, (px, py), self.args.particle_radius, (35, 125, 255), -1, cv2.LINE_AA)
            if self.args.particle_heading:
                head = (int(px + math.cos(yaw) * 8), int(py - math.sin(yaw) * 8))
                cv2.line(canvas, (px, py), head, (35, 125, 255), 1, cv2.LINE_AA)

    def draw_scan(self, canvas: np.ndarray, scale: float, pad_x: int, pad_y: int) -> None:
        if self.scan is None or self.map_state is None:
            return
        tf = self.lookup_transform(self.scan.header.frame_id, self.scan.header.stamp)
        if tf is None and self.scan.header.frame_id != self.map_state.frame_id and not self.args.assume_same_frame:
            return
        points = []
        angle = float(self.scan.angle_min)
        stride = max(1, self.args.scan_stride)
        for idx, r in enumerate(self.scan.ranges):
            if idx % stride != 0:
                angle += self.scan.angle_increment
                continue
            rr = float(r)
            if math.isfinite(rr) and self.scan.range_min <= rr <= self.scan.range_max:
                x = math.cos(angle) * rr
                y = math.sin(angle) * rr
                if tf is not None:
                    x, y = transform_xy(x, y, tf)
                points.append(self.map_px_to_view_px(*self.world_to_map_px(x, y), scale, pad_x, pad_y))
            angle += self.scan.angle_increment
        if points:
            pts = np.asarray(points, dtype=np.int32)
            cv2.drawMarker(canvas, tuple(pts[0]), (255, 130, 20), cv2.MARKER_CROSS, 6, 1, cv2.LINE_AA)
            for x, y in pts:
                cv2.circle(canvas, (int(x), int(y)), self.args.scan_radius, (245, 145, 30), -1, cv2.LINE_AA)

    def draw_path(self, canvas: np.ndarray, scale: float, pad_x: int, pad_y: int) -> None:
        if self.path is None or self.map_state is None or not self.path.poses:
            return
        points = []
        for stamped in self.path.poses:
            x = float(stamped.pose.position.x)
            y = float(stamped.pose.position.y)
            tf = self.lookup_transform(stamped.header.frame_id or self.path.header.frame_id, stamped.header.stamp)
            if tf is not None:
                x, y = transform_xy(x, y, tf)
            elif (stamped.header.frame_id or self.path.header.frame_id) != self.map_state.frame_id and not self.args.assume_same_frame:
                continue
            points.append((x, y))
        self.draw_polyline_world(canvas, points, (40, 180, 80), 3, scale, pad_x, pad_y)

    def render(self) -> None:
        if self.map_state is None:
            sys.stdout.write("\033[H\033[2Jwaiting for OccupancyGrid on %s...\n" % self.args.map_topic)
            sys.stdout.flush()
            return

        start = time.monotonic()
        map_img = self.map_state.image_bgr
        scale = min(self.args.width / self.map_state.width, self.args.height / self.map_state.height)
        scale = max(scale, 0.01)
        view_w = max(1, int(round(self.map_state.width * scale)))
        view_h = max(1, int(round(self.map_state.height * scale)))
        canvas = np.full((self.args.height, self.args.width, 3), 28, dtype=np.uint8)
        resized = cv2.resize(map_img, (view_w, view_h), interpolation=cv2.INTER_NEAREST if scale >= 1.0 else cv2.INTER_AREA)
        pad_x = (self.args.width - view_w) // 2
        pad_y = (self.args.height - view_h) // 2
        canvas[pad_y : pad_y + view_h, pad_x : pad_x + view_w] = resized

        self.draw_path(canvas, scale, pad_x, pad_y)
        self.draw_scan(canvas, scale, pad_x, pad_y)
        self.draw_particles(canvas, scale, pad_x, pad_y)
        pose = self.pose_in_map(self.pose) if self.pose is not None else None
        if pose is not None:
            self.draw_pose(canvas, pose, scale, pad_x, pad_y)

        self.frame_count += 1
        status = self.status_line(scale, time.monotonic() - start)
        cv2.rectangle(canvas, (0, 0), (self.args.width, 28), (20, 22, 26), -1)
        cv2.putText(canvas, status[:180], (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression])
        if not ok:
            self.get_logger().warn("failed to encode terminal map frame")
            return
        sys.stdout.write("\033[H\033[2J")
        sys.stdout.flush()
        write_kitty_image(bytes(encoded), self.args.width, self.args.height)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def status_line(self, scale: float, render_sec: float) -> str:
        pose_age = "-"
        if self.pose is not None:
            pose_age = self.pose.frame_id or "?"
        scan_age = self.scan.header.frame_id if self.scan is not None else "-"
        particles = len(self.particles.poses) if self.particles is not None else 0
        return (
            f"frame={self.frame_count} map={self.map_state.width}x{self.map_state.height}@{self.map_state.resolution:.3f}m "
            f"scale={scale:.2f} pose={pose_age} particles={particles} scan={scan_age} render={render_sec * 1000:.1f}ms"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--map-qos", choices=("both", "transient_local", "volatile"), default="both")
    parser.add_argument("--pose-topic", default="/localization_result", help="PoseWithCovarianceStamped topic, empty to disable")
    parser.add_argument("--pose-stamped-topic", default="", help="PoseStamped topic, empty to disable")
    parser.add_argument("--particles-topic", default="/particle_cloud", help="PoseArray topic, empty to disable")
    parser.add_argument("--scan-topic", default="/scan", help="LaserScan topic, empty to disable")
    parser.add_argument("--path-topic", default="", help="nav_msgs/Path topic, empty to disable")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument("--best-effort", action="store_true", help="use best-effort QoS for scan/particles")
    parser.add_argument("--no-tf", action="store_true", help="do not subscribe to TF")
    parser.add_argument("--assume-same-frame", action="store_true", help="draw overlays without TF even if frame_id differs")
    parser.add_argument("--tf-timeout", type=float, default=0.02)
    parser.add_argument("--scan-stride", type=int, default=3)
    parser.add_argument("--scan-radius", type=int, default=2)
    parser.add_argument("--particle-stride", type=int, default=1)
    parser.add_argument("--particle-radius", type=int, default=2)
    parser.add_argument("--particle-heading", action="store_true")
    parser.add_argument("--png-compression", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TerminalMapViewer(args)
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

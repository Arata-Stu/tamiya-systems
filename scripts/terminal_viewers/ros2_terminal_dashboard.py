#!/usr/bin/env python3
"""Custom terminal dashboard for ROS 2 map, localization, scan, image, and sections."""

from __future__ import annotations

import argparse
import math
import select
import signal
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from ros2_terminal_image_viewer import compressed_image_to_rgb, raw_image_to_rgb, resize_to_fit, write_kitty_image
from ros2_terminal_map_viewer import MapState, Pose2D, transform_xy, transform_xy_yaw, yaw_from_quat

try:
    import tf2_ros
except ImportError:  # pragma: no cover
    tf2_ros = None


Color = tuple[int, int, int]


def derive_camera_info_topic(image_topic: str) -> str:
    topic = image_topic.rstrip("/")
    if topic.endswith("/compressed"):
        topic = topic[: -len("/compressed")]
    for suffix in ("/image_rect_raw", "/image_rect", "/image_raw", "/image_color", "/image_mono", "/image"):
        if topic.endswith(suffix):
            return topic[: -len(suffix)] + "/camera_info"
    return topic + "/camera_info"


def transform_xyz(x: float, y: float, z: float, transform: object) -> tuple[float, float, float]:
    tx = float(transform.translation.x)
    ty = float(transform.translation.y)
    tz = float(transform.translation.z)
    qx = float(transform.rotation.x)
    qy = float(transform.rotation.y)
    qz = float(transform.rotation.z)
    qw = float(transform.rotation.w)

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    rx = (1.0 - 2.0 * (yy + zz)) * x + 2.0 * (xy - wz) * y + 2.0 * (xz + wy) * z
    ry = 2.0 * (xy + wz) * x + (1.0 - 2.0 * (xx + zz)) * y + 2.0 * (yz - wx) * z
    rz = 2.0 * (xz - wy) * x + 2.0 * (yz + wx) * y + (1.0 - 2.0 * (xx + yy)) * z
    return tx + rx, ty + ry, tz + rz


@dataclass
class DashboardState:
    map_state: Optional[MapState] = None
    localization: Optional[Pose2D] = None
    amcl_pose: Optional[Pose2D] = None
    initial_pose: Optional[Pose2D] = None
    scan: Optional[LaserScan] = None
    image_rgb: Optional[np.ndarray] = None
    camera_info: Optional[CameraInfo] = None
    particles: Optional[PoseArray] = None
    path: Optional[Path] = None
    odom_speed_mps: Optional[float] = None
    odom_linear_x: Optional[float] = None
    odom_linear_y: Optional[float] = None
    odom_angular_z: Optional[float] = None
    odom_frame_id: str = ""
    odom_child_frame_id: str = ""
    current_section: str = "-"
    frames: int = 0
    image_frame_id: str = ""
    image_stamp: Optional[object] = None
    last_image_error: str = ""
    last_projection_error: str = ""


class TerminalDashboard(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("terminal_dashboard")
        self.args = args
        self.state = DashboardState()
        self.section_markers: dict[tuple[str, int], Marker] = {}
        self.current_section_marker: Optional[Marker] = None
        self.running = True
        self.paused = False
        self.toggles = {
            "map": True,
            "localization": True,
            "amcl": True,
            "initialpose": True,
            "scan": True,
            "image": bool(args.image_topic),
            "sections": True,
            "gates": True,
            "particles": False,
            "path": True,
        }
        self.last_render_time = 0.0
        self.last_key_status = ""

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
        volatile_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        reliable_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE)
        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT if args.best_effort else ReliabilityPolicy.RELIABLE,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        if args.map_topic:
            self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, map_qos)
            self.create_subscription(OccupancyGrid, args.map_topic, self.on_map, volatile_map_qos)
        if args.localization_topic:
            self.create_subscription(PoseWithCovarianceStamped, args.localization_topic, self.on_localization, reliable_qos)
        if args.amcl_pose_topic:
            self.create_subscription(PoseWithCovarianceStamped, args.amcl_pose_topic, self.on_amcl_pose, reliable_qos)
        if args.initial_pose_topic:
            self.create_subscription(PoseWithCovarianceStamped, args.initial_pose_topic, self.on_initial_pose, reliable_qos)
        if args.scan_topic:
            self.create_subscription(LaserScan, args.scan_topic, self.on_scan, sensor_qos)
        if args.odom_topic:
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, sensor_qos)
        if args.image_topic:
            if args.compressed_image:
                self.create_subscription(CompressedImage, args.image_topic, self.on_compressed_image, sensor_qos)
            else:
                self.create_subscription(Image, args.image_topic, self.on_image, sensor_qos)
        if args.image_topic and args.camera_info_topic:
            self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, sensor_qos)
        if args.particles_topic:
            self.create_subscription(PoseArray, args.particles_topic, self.on_particles, sensor_qos)
        if args.path_topic:
            self.create_subscription(Path, args.path_topic, self.on_path, reliable_qos)
        if args.section_markers_topic:
            self.create_subscription(MarkerArray, args.section_markers_topic, self.on_section_markers, marker_qos)
        if args.current_section_marker_topic:
            self.create_subscription(Marker, args.current_section_marker_topic, self.on_current_section_marker, marker_qos)
        if args.current_section_topic:
            self.create_subscription(String, args.current_section_topic, self.on_current_section, marker_qos)

    def on_map(self, msg: OccupancyGrid) -> None:
        data = np.asarray(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        image = np.zeros((msg.info.height, msg.info.width, 3), dtype=np.uint8)
        image[data < 0] = (178, 178, 178)
        image[data == 0] = (245, 245, 245)
        occupied = data > 0
        image[occupied] = np.clip(235 - data[occupied] * 2, 25, 235).astype(np.uint8)[:, None]
        self.state.map_state = MapState(
            frame_id=msg.header.frame_id or self.args.map_frame,
            resolution=float(msg.info.resolution),
            origin_x=float(msg.info.origin.position.x),
            origin_y=float(msg.info.origin.position.y),
            origin_yaw=yaw_from_quat(msg.info.origin.orientation),
            width=int(msg.info.width),
            height=int(msg.info.height),
            image_bgr=np.flipud(image),
        )

    def on_localization(self, msg: PoseWithCovarianceStamped) -> None:
        self.state.localization = self.pose_from_cov_msg(msg)

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self.state.amcl_pose = self.pose_from_cov_msg(msg)

    def on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self.state.initial_pose = self.pose_from_cov_msg(msg)

    def pose_from_cov_msg(self, msg: PoseWithCovarianceStamped) -> Pose2D:
        pose = msg.pose.pose
        return Pose2D(
            x=float(pose.position.x),
            y=float(pose.position.y),
            yaw=yaw_from_quat(pose.orientation),
            frame_id=msg.header.frame_id or self.args.map_frame,
            stamp=msg.header.stamp,
            covariance=list(msg.pose.covariance),
        )

    def on_scan(self, msg: LaserScan) -> None:
        self.state.scan = msg

    def on_odom(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        linear_x = float(twist.linear.x)
        linear_y = float(twist.linear.y)
        self.state.odom_speed_mps = math.hypot(linear_x, linear_y)
        self.state.odom_linear_x = linear_x
        self.state.odom_linear_y = linear_y
        self.state.odom_angular_z = float(twist.angular.z)
        self.state.odom_frame_id = msg.header.frame_id
        self.state.odom_child_frame_id = msg.child_frame_id

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.state.camera_info = msg
        self.state.last_projection_error = ""

    def on_image(self, msg: Image) -> None:
        try:
            self.state.image_rgb = raw_image_to_rgb(msg)
            self.state.image_frame_id = msg.header.frame_id
            self.state.image_stamp = msg.header.stamp
            self.state.last_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_image_error = str(exc)

    def on_compressed_image(self, msg: CompressedImage) -> None:
        try:
            self.state.image_rgb = compressed_image_to_rgb(msg)
            self.state.image_frame_id = msg.header.frame_id
            self.state.image_stamp = msg.header.stamp
            self.state.last_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_image_error = str(exc)

    def on_particles(self, msg: PoseArray) -> None:
        self.state.particles = msg

    def on_path(self, msg: Path) -> None:
        self.state.path = msg

    def on_current_section(self, msg: String) -> None:
        self.state.current_section = msg.data or "-"

    def on_section_markers(self, msg: MarkerArray) -> None:
        for marker in msg.markers:
            self.store_marker(marker, current=False)

    def on_current_section_marker(self, msg: Marker) -> None:
        self.store_marker(msg, current=True)

    def store_marker(self, marker: Marker, current: bool) -> None:
        if marker.action == Marker.DELETEALL:
            self.current_section_marker = None if current else self.current_section_marker
            if not current:
                self.section_markers.clear()
            return
        if marker.action == Marker.DELETE:
            if current:
                self.current_section_marker = None
            else:
                self.section_markers.pop((marker.ns, marker.id), None)
            return
        if marker.action != Marker.ADD:
            return
        if current:
            self.current_section_marker = marker
        else:
            self.section_markers[(marker.ns, marker.id)] = marker

    def lookup_transform_between(self, target_frame: str, source_frame: str, stamp: object) -> Optional[object]:
        if not target_frame or not source_frame or source_frame == target_frame or self.tf_buffer is None:
            return None
        try:
            when = Time.from_msg(stamp) if stamp is not None else Time()
            return self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                when,
                timeout=Duration(seconds=self.args.tf_timeout),
            ).transform
        except Exception:
            try:
                return self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    Time(),
                    timeout=Duration(seconds=self.args.tf_timeout),
                ).transform
            except Exception:
                return None

    def lookup_transform(self, source_frame: str, stamp: object) -> Optional[object]:
        map_state = self.state.map_state
        if map_state is None:
            return None
        return self.lookup_transform_between(map_state.frame_id, source_frame, stamp)

    def pose_in_map(self, pose: Optional[Pose2D]) -> Optional[Pose2D]:
        map_state = self.state.map_state
        if pose is None or map_state is None:
            return None
        tf = self.lookup_transform(pose.frame_id, pose.stamp)
        if tf is None:
            if pose.frame_id and pose.frame_id != map_state.frame_id and not self.args.assume_same_frame:
                return None
            return pose
        x, y, yaw = transform_xy_yaw(pose.x, pose.y, pose.yaw, tf)
        return Pose2D(x=x, y=y, yaw=yaw, frame_id=map_state.frame_id, stamp=pose.stamp, covariance=pose.covariance)

    def world_to_map_px(self, x: float, y: float) -> tuple[int, int]:
        map_state = self.state.map_state
        assert map_state is not None
        dx = x - map_state.origin_x
        dy = y - map_state.origin_y
        c = math.cos(-map_state.origin_yaw)
        s = math.sin(-map_state.origin_yaw)
        mx = (c * dx - s * dy) / map_state.resolution
        my = (s * dx + c * dy) / map_state.resolution
        return int(round(mx)), int(round(map_state.height - 1 - my))

    def map_px_to_view_px(self, px: int, py: int, scale: float, ox: int, oy: int) -> tuple[int, int]:
        return int(round(px * scale + ox)), int(round(py * scale + oy))

    def world_to_view_px(self, x: float, y: float, scale: float, ox: int, oy: int) -> tuple[int, int]:
        return self.map_px_to_view_px(*self.world_to_map_px(x, y), scale, ox, oy)

    def draw_map_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        x0, y0, w, h = rect
        map_state = self.state.map_state
        if map_state is None or not self.toggles["map"]:
            self.draw_empty_panel(canvas, rect, "waiting for /map")
            return

        scale = min(w / map_state.width, h / map_state.height)
        view_w = max(1, int(round(map_state.width * scale)))
        view_h = max(1, int(round(map_state.height * scale)))
        ox = x0 + (w - view_w) // 2
        oy = y0 + (h - view_h) // 2
        resized = cv2.resize(map_state.image_bgr, (view_w, view_h), interpolation=cv2.INTER_NEAREST if scale >= 1 else cv2.INTER_AREA)
        canvas[y0 : y0 + h, x0 : x0 + w] = (34, 35, 38)
        canvas[oy : oy + view_h, ox : ox + view_w] = resized

        if self.toggles["sections"]:
            self.draw_sections(canvas, scale, ox, oy)
        if self.toggles["path"]:
            self.draw_path(canvas, scale, ox, oy)
        if self.toggles["scan"]:
            self.draw_scan_on_map(canvas, scale, ox, oy)
        if self.toggles["particles"]:
            self.draw_particles(canvas, scale, ox, oy)
        if self.toggles["localization"]:
            pose = self.pose_in_map(self.state.localization)
            if pose is not None:
                self.draw_pose(canvas, pose, scale, ox, oy, (45, 70, 245), "GL")
        if self.toggles["amcl"]:
            pose = self.pose_in_map(self.state.amcl_pose)
            if pose is not None:
                self.draw_pose(canvas, pose, scale, ox, oy, (65, 205, 80), "AMCL")
        if self.toggles["initialpose"]:
            pose = self.pose_in_map(self.state.initial_pose)
            if pose is not None:
                self.draw_pose(canvas, pose, scale, ox, oy, (40, 220, 235), "INIT")

    def draw_pose(self, canvas: np.ndarray, pose: Pose2D, scale: float, ox: int, oy: int, color: Color, label: str) -> None:
        map_state = self.state.map_state
        assert map_state is not None
        px, py = self.world_to_view_px(pose.x, pose.y, scale, ox, oy)
        length = max(20, int(0.50 / max(map_state.resolution, 1e-6) * scale))
        end = (int(px + math.cos(pose.yaw) * length), int(py - math.sin(pose.yaw) * length))
        cv2.arrowedLine(canvas, (px, py), end, color, max(2, int(4 * scale)), cv2.LINE_AA, tipLength=0.35)
        cv2.circle(canvas, (px, py), max(5, int(7 * scale)), (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), max(3, int(4 * scale)), color, -1, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (px + 8, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def draw_scan_on_map(self, canvas: np.ndarray, scale: float, ox: int, oy: int) -> None:
        scan = self.state.scan
        map_state = self.state.map_state
        if scan is None or map_state is None:
            return
        tf = self.lookup_transform(scan.header.frame_id, scan.header.stamp)
        if tf is None and scan.header.frame_id != map_state.frame_id and not self.args.assume_same_frame:
            return
        angle = float(scan.angle_min)
        for idx, raw in enumerate(scan.ranges):
            if idx % max(1, self.args.scan_stride) != 0:
                angle += scan.angle_increment
                continue
            r = float(raw)
            if math.isfinite(r) and scan.range_min <= r <= scan.range_max:
                x = math.cos(angle) * r
                y = math.sin(angle) * r
                if tf is not None:
                    x, y = transform_xy(x, y, tf)
                px, py = self.world_to_view_px(x, y, scale, ox, oy)
                cv2.circle(canvas, (px, py), self.args.scan_radius, (245, 145, 30), -1, cv2.LINE_AA)
            angle += scan.angle_increment

    def draw_particles(self, canvas: np.ndarray, scale: float, ox: int, oy: int) -> None:
        particles = self.state.particles
        map_state = self.state.map_state
        if particles is None or map_state is None:
            return
        tf = self.lookup_transform(particles.header.frame_id, particles.header.stamp)
        for idx, pose in enumerate(particles.poses):
            if idx % max(1, self.args.particle_stride) != 0:
                continue
            x = float(pose.position.x)
            y = float(pose.position.y)
            if tf is not None:
                x, y = transform_xy(x, y, tf)
            elif particles.header.frame_id != map_state.frame_id and not self.args.assume_same_frame:
                continue
            px, py = self.world_to_view_px(x, y, scale, ox, oy)
            cv2.circle(canvas, (px, py), 2, (35, 125, 255), -1, cv2.LINE_AA)

    def draw_path(self, canvas: np.ndarray, scale: float, ox: int, oy: int) -> None:
        path = self.state.path
        map_state = self.state.map_state
        if path is None or map_state is None or not path.poses:
            return
        pixels = []
        for stamped in path.poses:
            frame_id = stamped.header.frame_id or path.header.frame_id
            x = float(stamped.pose.position.x)
            y = float(stamped.pose.position.y)
            tf = self.lookup_transform(frame_id, stamped.header.stamp)
            if tf is not None:
                x, y = transform_xy(x, y, tf)
            elif frame_id != map_state.frame_id and not self.args.assume_same_frame:
                continue
            pixels.append(self.world_to_view_px(x, y, scale, ox, oy))
        if len(pixels) >= 2:
            cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], False, (40, 180, 80), 3, cv2.LINE_AA)

    def marker_color(self, marker: Marker, fallback: Color) -> Color:
        if marker.color.a <= 0:
            return fallback
        return (
            int(max(0.0, min(1.0, marker.color.b)) * 255),
            int(max(0.0, min(1.0, marker.color.g)) * 255),
            int(max(0.0, min(1.0, marker.color.r)) * 255),
        )

    def draw_sections(self, canvas: np.ndarray, scale: float, ox: int, oy: int) -> None:
        for marker in sorted(self.section_markers.values(), key=lambda m: (m.ns, m.id)):
            if not self.toggles["gates"] and marker.ns.startswith("section_gate"):
                continue
            self.draw_marker(canvas, marker, scale, ox, oy)
        if self.current_section_marker is not None:
            self.draw_marker(canvas, self.current_section_marker, scale, ox, oy, current=True)

    def draw_marker(self, canvas: np.ndarray, marker: Marker, scale: float, ox: int, oy: int, current: bool = False) -> None:
        map_state = self.state.map_state
        if map_state is None:
            return
        frame_id = marker.header.frame_id or map_state.frame_id
        tf = self.lookup_transform(frame_id, marker.header.stamp)
        if tf is None and frame_id != map_state.frame_id and not self.args.assume_same_frame:
            return
        color = self.marker_color(marker, (30, 70, 245) if current else (255, 180, 70))
        thickness = max(1, int(round(float(marker.scale.x or 0.05) / map_state.resolution * scale)))
        if current:
            thickness = max(thickness, 3)
        if marker.type in (Marker.LINE_STRIP, Marker.LINE_LIST):
            pixels = []
            marker_yaw = yaw_from_quat(marker.pose.orientation)
            c = math.cos(marker_yaw)
            s = math.sin(marker_yaw)
            for point in marker.points:
                x = float(marker.pose.position.x) + c * float(point.x) - s * float(point.y)
                y = float(marker.pose.position.y) + s * float(point.x) + c * float(point.y)
                if tf is not None:
                    x, y = transform_xy(x, y, tf)
                pixels.append(self.world_to_view_px(x, y, scale, ox, oy))
            if marker.type == Marker.LINE_STRIP and len(pixels) >= 2:
                cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)
            elif marker.type == Marker.LINE_LIST:
                for a, b in zip(pixels[0::2], pixels[1::2]):
                    cv2.line(canvas, a, b, color, thickness, cv2.LINE_AA)
        elif marker.type == Marker.TEXT_VIEW_FACING and marker.text:
            x = float(marker.pose.position.x)
            y = float(marker.pose.position.y)
            if tf is not None:
                x, y = transform_xy(x, y, tf)
            px, py = self.world_to_view_px(x, y, scale, ox, oy)
            cv2.putText(canvas, marker.text[:32], (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(canvas, marker.text[:32], (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    def draw_image_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        image = self.state.image_rgb
        if not self.toggles["image"]:
            self.draw_empty_panel(canvas, rect, "image hidden")
            return
        if image is None:
            text = self.state.last_image_error or "waiting for image"
            self.draw_empty_panel(canvas, rect, text)
            return
        x0, y0, w, h = rect
        canvas[y0 : y0 + h, x0 : x0 + w] = (18, 19, 22)
        if self.toggles["scan"]:
            image = self.draw_scan_on_image(image)
        rgb = resize_to_fit(image, w, h)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ih, iw = bgr.shape[:2]
        ox = x0 + (w - iw) // 2
        oy = y0 + (h - ih) // 2
        canvas[oy : oy + ih, ox : ox + iw] = bgr

    def draw_scan_on_image(self, image_rgb: np.ndarray) -> np.ndarray:
        scan = self.state.scan
        if scan is None:
            return image_rgb

        overlay = image_rgb.copy()

        def draw_status(text: str, color: Color) -> np.ndarray:
            cv2.putText(overlay, text[:48], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text[:48], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            return overlay

        camera_info = self.state.camera_info
        if camera_info is None:
            self.state.last_projection_error = "camera_info missing"
            return draw_status("scan proj: camera_info missing", (255, 120, 120))

        camera_frame = camera_info.header.frame_id or self.state.image_frame_id
        if not camera_frame:
            self.state.last_projection_error = "camera frame unavailable"
            return draw_status("scan proj: camera frame unavailable", (255, 120, 120))

        tf = self.lookup_transform_between(camera_frame, scan.header.frame_id, scan.header.stamp)
        if tf is None and scan.header.frame_id != camera_frame and not self.args.assume_same_frame:
            self.state.last_projection_error = f"no TF {scan.header.frame_id}->{camera_frame}"
            return draw_status("scan proj: TF unavailable", (255, 120, 120))

        if len(camera_info.p) >= 12 and (abs(float(camera_info.p[0])) > 1e-9 or abs(float(camera_info.p[5])) > 1e-9):
            fx = float(camera_info.p[0])
            fy = float(camera_info.p[5])
            cx = float(camera_info.p[2])
            cy = float(camera_info.p[6])
        elif len(camera_info.k) >= 9:
            fx = float(camera_info.k[0])
            fy = float(camera_info.k[4])
            cx = float(camera_info.k[2])
            cy = float(camera_info.k[5])
        else:
            self.state.last_projection_error = "camera intrinsics unavailable"
            return draw_status("scan proj: intrinsics unavailable", (255, 120, 120))

        image_h, image_w = overlay.shape[:2]
        visible = 0
        sampled = 0
        angle = float(scan.angle_min)
        stride = max(1, self.args.scan_stride)
        radius = max(1, self.args.scan_radius)

        for idx, raw in enumerate(scan.ranges):
            if idx % stride != 0:
                angle += scan.angle_increment
                continue

            sampled += 1
            r = float(raw)
            if not math.isfinite(r) or r < scan.range_min or r > scan.range_max:
                angle += scan.angle_increment
                continue

            x = math.cos(angle) * r
            y = math.sin(angle) * r
            z = 0.0
            if tf is not None:
                x, y, z = transform_xyz(x, y, z, tf)

            if z <= 1e-6:
                angle += scan.angle_increment
                continue

            u = fx * (x / z) + cx
            v = fy * (y / z) + cy
            if 0.0 <= u < image_w and 0.0 <= v < image_h:
                depth_alpha = min(1.0, max(0.0, z / max(scan.range_max, 1e-3)))
                color = (
                    255,
                    int(round(220.0 - 120.0 * depth_alpha)),
                    int(round(60.0 + 120.0 * depth_alpha)),
                )
                cv2.circle(overlay, (int(round(u)), int(round(v))), radius, color, -1, cv2.LINE_AA)
                visible += 1
            angle += scan.angle_increment

        self.state.last_projection_error = ""
        status = f"scan proj {visible}/{sampled}"
        return draw_status(status, (255, 255, 255))

    def draw_empty_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str) -> None:
        x0, y0, w, h = rect
        canvas[y0 : y0 + h, x0 : x0 + w] = (24, 25, 28)
        cv2.putText(canvas, text[:60], (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 195, 205), 1, cv2.LINE_AA)

    def handle_key(self, key: str) -> None:
        mapping = {
            "m": "map",
            "l": "localization",
            "a": "amcl",
            "u": "initialpose",
            "s": "scan",
            "i": "image",
            "c": "sections",
            "g": "gates",
            "p": "particles",
            "t": "path",
        }
        if key == "q":
            self.running = False
            return
        if key == " ":
            self.paused = not self.paused
            self.last_key_status = "paused" if self.paused else "running"
            return
        if key in mapping:
            name = mapping[key]
            self.toggles[name] = not self.toggles[name]
            self.last_key_status = f"{name}={'on' if self.toggles[name] else 'off'}"

    def render(self) -> None:
        if self.paused:
            return
        now = time.monotonic()
        if now - self.last_render_time < 1.0 / max(self.args.max_fps, 0.2):
            return
        self.last_render_time = now
        self.state.frames += 1

        canvas = np.full((self.args.height, self.args.width, 3), (20, 21, 24), dtype=np.uint8)
        header_h = 52
        gap = 6
        has_image_panel = bool(self.args.image_topic)
        right_w = int(self.args.width * self.args.image_panel_ratio) if has_image_panel else 0
        map_rect = (0, header_h, self.args.width - right_w - (gap if has_image_panel else 0), self.args.height - header_h)
        image_rect = (self.args.width - right_w, header_h, right_w, self.args.height - header_h)
        self.draw_map_panel(canvas, map_rect)
        if has_image_panel:
            self.draw_image_panel(canvas, image_rect)
            cv2.line(canvas, (image_rect[0] - gap // 2, header_h), (image_rect[0] - gap // 2, self.args.height), (55, 58, 64), 1)
        self.draw_header(canvas)

        ok, encoded = cv2.imencode(".png", canvas, [cv2.IMWRITE_PNG_COMPRESSION, self.args.png_compression])
        if not ok:
            return
        sys.stdout.write("\033[H")
        sys.stdout.flush()
        write_kitty_image(bytes(encoded), self.args.width, self.args.height)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def draw_header(self, canvas: np.ndarray) -> None:
        header_h = 52
        cv2.rectangle(canvas, (0, 0), (self.args.width, header_h), (14, 15, 18), -1)
        loc = self.state.localization.frame_id if self.state.localization is not None else "-"
        amcl = self.pose_status(self.state.amcl_pose)
        init = self.pose_status(self.state.initial_pose)
        scan = self.state.scan.header.frame_id if self.state.scan is not None else "-"
        image = "yes" if self.state.image_rgb is not None else "-"
        camera_info = "yes" if self.state.camera_info is not None else "-"
        flags = " ".join(f"{k[0]}:{'1' if v else '0'}" for k, v in self.toggles.items())
        odom = self.odom_status()
        line1 = f"frame={self.state.frames} loc={loc} amcl={amcl} init={init} scan={scan} section={self.state.current_section}"
        line2 = (
            f"odom={odom} img={image} cam={camera_info} tog={flags} "
            f"keys=m/l/a/u/s/i/c/g/p/t space q {self.last_key_status}"
        )
        cv2.putText(canvas, line1[:180], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2[:180], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 220, 228), 1, cv2.LINE_AA)

    def pose_status(self, pose: Optional[Pose2D]) -> str:
        if pose is None:
            return "-"
        if pose.covariance is None or len(pose.covariance) < 36:
            return pose.frame_id
        return f"{pose.frame_id}:{pose.covariance[0]:.2g}/{pose.covariance[7]:.2g}/{pose.covariance[35]:.2g}"

    def odom_status(self) -> str:
        if self.state.odom_speed_mps is None:
            return "-"
        frame_id = self.state.odom_frame_id or "-"
        child_frame_id = self.state.odom_child_frame_id or "-"
        return (
            f"{frame_id}->{child_frame_id} "
            f"v={self.state.odom_speed_mps:.2f}m/s "
            f"vx={self.state.odom_linear_x:.2f} "
            f"vy={self.state.odom_linear_y:.2f} "
            f"wz={self.state.odom_angular_z:.2f}"
        )


class RawTerminal:
    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self) -> str:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return ""
        return sys.stdin.read(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-topic", default="/map")
    parser.add_argument("--localization-topic", default="/localization_result")
    parser.add_argument("--amcl-pose-topic", default="/amcl_pose")
    parser.add_argument("--initial-pose-topic", default="/initialpose")
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument(
        "--odom-topic",
        default="/visual_slam/tracking/odometry",
        help="nav_msgs/Odometry topic used to display planar speed, empty to disable",
    )
    parser.add_argument("--image-topic", default="/camera/left/image_raw")
    parser.add_argument("--camera-info-topic", default=None)
    parser.add_argument("--compressed-image", action="store_true")
    parser.add_argument("--particles-topic", default="/particle_cloud")
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--section-markers-topic", default="/localization/section_markers")
    parser.add_argument("--current-section-marker-topic", default="/localization/current_section_marker")
    parser.add_argument("--current-section-topic", default="/localization/current_section")
    parser.add_argument("--map-frame", default="map")
    parser.add_argument("--width", type=int, default=1400)
    parser.add_argument("--height", type=int, default=850)
    parser.add_argument("--max-fps", type=float, default=3.0)
    parser.add_argument("--best-effort", action="store_true")
    parser.add_argument("--no-tf", action="store_true")
    parser.add_argument("--assume-same-frame", action="store_true")
    parser.add_argument("--tf-timeout", type=float, default=0.02)
    parser.add_argument("--scan-stride", type=int, default=3)
    parser.add_argument("--scan-radius", type=int, default=2)
    parser.add_argument("--particle-stride", type=int, default=1)
    parser.add_argument("--image-panel-ratio", type=float, default=0.34)
    parser.add_argument("--png-compression", type=int, default=3)
    args = parser.parse_args()
    if args.camera_info_topic is None and args.image_topic:
        args.camera_info_topic = derive_camera_info_topic(args.image_topic)
    return args


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = TerminalDashboard(args)

    def request_stop(_signum: int, _frame: object) -> None:
        node.running = False

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    sys.stdout.write("\033[?1049h\033[?25l\033[H\033[2J")
    sys.stdout.flush()
    try:
        with RawTerminal() if sys.stdin.isatty() else null_terminal() as terminal:
            while rclpy.ok() and node.running:
                rclpy.spin_once(node, timeout_sec=0.02)
                key = terminal.read_key()
                if key:
                    node.handle_key(key)
                node.render()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.stdout.write("\033[?25h\033[?1049l")
        sys.stdout.flush()
    return 0


class null_terminal:
    def __enter__(self) -> "null_terminal":
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        return None

    def read_key(self) -> str:
        return ""


if __name__ == "__main__":
    raise SystemExit(main())

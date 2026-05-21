#!/usr/bin/env python3
"""Custom terminal dashboard for ROS 2 map, localization, scan, images, and paths."""

from __future__ import annotations

import argparse
from collections import deque
import math
import os
import re
import select
import signal
import shutil
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


def stamp_to_ns(stamp: object) -> Optional[int]:
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nanosec = getattr(stamp, "nanosec", None)
    if sec is None or nanosec is None:
        return None
    return int(sec) * 1_000_000_000 + int(nanosec)


@dataclass
class TimedImageFrame:
    rgb: np.ndarray
    frame_id: str
    stamp: object


@dataclass
class TimedOdomState:
    stamp: object
    speed_mps: float
    linear_x: float
    linear_y: float
    angular_z: float
    frame_id: str
    child_frame_id: str


@dataclass
class SelectionState:
    reference_source: str = "-"
    reference_stamp: Optional[object] = None
    localization: Optional[Pose2D] = None
    amcl_pose: Optional[Pose2D] = None
    initial_pose: Optional[Pose2D] = None
    scan: Optional[LaserScan] = None
    image: Optional[TimedImageFrame] = None
    crop_image: Optional[TimedImageFrame] = None
    particles: Optional[PoseArray] = None
    odom: Optional[TimedOdomState] = None
    path: Optional[Path] = None
    vo_path: Optional[Path] = None
    global_path: Optional[Path] = None
    local_path: Optional[Path] = None


@dataclass
class DashboardState:
    map_state: Optional[MapState] = None
    localization: Optional[Pose2D] = None
    amcl_pose: Optional[Pose2D] = None
    initial_pose: Optional[Pose2D] = None
    scan: Optional[LaserScan] = None
    image_rgb: Optional[np.ndarray] = None
    crop_image_rgb: Optional[np.ndarray] = None
    camera_info: Optional[CameraInfo] = None
    particles: Optional[PoseArray] = None
    path: Optional[Path] = None
    vo_path: Optional[Path] = None
    global_path: Optional[Path] = None
    local_path: Optional[Path] = None
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
    crop_image_frame_id: str = ""
    crop_image_stamp: Optional[object] = None
    last_crop_image_error: str = ""
    last_projection_error: str = ""


@dataclass
class InputEvent:
    key: str = ""
    mouse_x: Optional[int] = None
    mouse_y: Optional[int] = None
    mouse_button: Optional[int] = None
    mouse_pressed: bool = False


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
            "crop": bool(args.crop_image_topic),
            "sections": True,
            "gates": True,
            "particles": False,
            "path": True,
            "vo_path": True,
            "global_path": True,
            "local_path": True,
        }
        self.last_render_time = 0.0
        self.last_key_status = ""
        self.selection = SelectionState()
        self.last_sync_status = "sync=-"
        self.toggle_button_rects: dict[str, tuple[int, int, int, int]] = {}
        self.sync_tolerance_ns = int(max(0.0, args.sync_tolerance_ms) * 1_000_000.0)
        self.localization_buffer: deque[Pose2D] = deque(maxlen=args.sync_buffer_size)
        self.amcl_pose_buffer: deque[Pose2D] = deque(maxlen=args.sync_buffer_size)
        self.initial_pose_buffer: deque[Pose2D] = deque(maxlen=args.sync_buffer_size)
        self.scan_buffer: deque[LaserScan] = deque(maxlen=args.sync_buffer_size)
        self.image_buffer: deque[TimedImageFrame] = deque(maxlen=args.sync_buffer_size)
        self.crop_image_buffer: deque[TimedImageFrame] = deque(maxlen=args.sync_buffer_size)
        self.camera_info_buffer: deque[CameraInfo] = deque(maxlen=args.sync_buffer_size)
        self.particles_buffer: deque[PoseArray] = deque(maxlen=args.sync_buffer_size)
        state_buffer_size = max(1, args.state_sync_buffer_size)
        self.odom_buffer: deque[TimedOdomState] = deque(maxlen=state_buffer_size)
        self.path_buffer: deque[Path] = deque(maxlen=state_buffer_size)
        self.vo_path_buffer: deque[Path] = deque(maxlen=state_buffer_size)
        self.global_path_buffer: deque[Path] = deque(maxlen=state_buffer_size)
        self.local_path_buffer: deque[Path] = deque(maxlen=state_buffer_size)

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
        crop_sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
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
        if args.crop_image_topic:
            if args.crop_compressed_image:
                self.create_subscription(
                    CompressedImage,
                    args.crop_image_topic,
                    self.on_crop_compressed_image,
                    crop_sensor_qos,
                )
            else:
                self.create_subscription(Image, args.crop_image_topic, self.on_crop_image, crop_sensor_qos)
        if args.image_topic and args.camera_info_topic:
            self.create_subscription(CameraInfo, args.camera_info_topic, self.on_camera_info, sensor_qos)
        if args.particles_topic:
            self.create_subscription(PoseArray, args.particles_topic, self.on_particles, sensor_qos)
        if args.path_topic:
            self.create_subscription(Path, args.path_topic, self.on_path, reliable_qos)
        if args.vo_path_topic:
            self.create_subscription(Path, args.vo_path_topic, self.on_vo_path, reliable_qos)
        if args.global_path_topic:
            self.create_subscription(Path, args.global_path_topic, self.on_global_path, reliable_qos)
        if args.local_path_topic:
            self.create_subscription(Path, args.local_path_topic, self.on_local_path, reliable_qos)
        if args.section_markers_topic:
            self.create_subscription(MarkerArray, args.section_markers_topic, self.on_section_markers, marker_qos)
        if args.current_section_marker_topic:
            self.create_subscription(Marker, args.current_section_marker_topic, self.on_current_section_marker, marker_qos)
        if args.current_section_topic:
            self.create_subscription(String, args.current_section_topic, self.on_current_section, marker_qos)

    def header_toggle_specs(self) -> tuple[tuple[str, str, str], ...]:
        specs = [
            ("map", "m", "Map"),
            ("localization", "l", "GL"),
            ("amcl", "a", "AMCL"),
            ("initialpose", "u", "Init"),
            ("scan", "s", "Scan"),
        ]
        if self.args.image_topic:
            specs.append(("image", "i", "Image"))
        if self.args.crop_image_topic:
            specs.append(("crop", "r", "Crop"))
        specs.extend(
            [
                ("sections", "c", "Sect"),
                ("gates", "g", "Gate"),
            ]
        )
        if self.args.particles_topic:
            specs.append(("particles", "p", "Part"))
        specs.extend(
            [
                ("path", "t", "Slam"),
                ("vo_path", "v", "VO"),
                ("global_path", "y", "Global"),
                ("local_path", "h", "Local"),
            ]
        )
        return tuple(specs)

    def layout_toggle_buttons(self, canvas_width: int) -> list[tuple[str, str, str, tuple[int, int, int, int]]]:
        x = 8
        y = 50
        row_h = 24
        gap_x = 6
        gap_y = 6
        right_margin = 8
        font_scale = 0.48
        layout = []

        for name, key, label in self.header_toggle_specs():
            text = f"{label} [{key}]"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            btn_w = text_w + 16
            btn_h = max(row_h, text_h + baseline + 10)
            if x + btn_w > canvas_width - right_margin:
                x = 8
                y += row_h + gap_y
            layout.append((name, key, text, (x, y, btn_w, btn_h)))
            x += btn_w + gap_x

        return layout

    def header_height(self) -> int:
        layout = self.layout_toggle_buttons(self.args.width)
        if not layout:
            return 52
        bottom = max(rect[1] + rect[3] for _, _, _, rect in layout)
        return max(52, bottom + 8)

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
        self.localization_buffer.append(self.state.localization)

    def on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self.state.amcl_pose = self.pose_from_cov_msg(msg)
        self.amcl_pose_buffer.append(self.state.amcl_pose)

    def on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        self.state.initial_pose = self.pose_from_cov_msg(msg)
        self.initial_pose_buffer.append(self.state.initial_pose)

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
        self.scan_buffer.append(msg)

    def on_odom(self, msg: Odometry) -> None:
        twist = msg.twist.twist
        linear_x = float(twist.linear.x)
        linear_y = float(twist.linear.y)
        speed_mps = math.hypot(linear_x, linear_y)
        self.state.odom_speed_mps = speed_mps
        self.state.odom_linear_x = linear_x
        self.state.odom_linear_y = linear_y
        self.state.odom_angular_z = float(twist.angular.z)
        self.state.odom_frame_id = msg.header.frame_id
        self.state.odom_child_frame_id = msg.child_frame_id
        self.odom_buffer.append(
            TimedOdomState(
                stamp=msg.header.stamp,
                speed_mps=speed_mps,
                linear_x=linear_x,
                linear_y=linear_y,
                angular_z=float(twist.angular.z),
                frame_id=msg.header.frame_id,
                child_frame_id=msg.child_frame_id,
            )
        )

    def on_camera_info(self, msg: CameraInfo) -> None:
        self.state.camera_info = msg
        self.camera_info_buffer.append(msg)
        self.state.last_projection_error = ""

    def on_image(self, msg: Image) -> None:
        try:
            rgb = raw_image_to_rgb(msg)
            self.state.image_rgb = rgb
            self.state.image_frame_id = msg.header.frame_id
            self.state.image_stamp = msg.header.stamp
            self.image_buffer.append(TimedImageFrame(rgb=rgb, frame_id=msg.header.frame_id, stamp=msg.header.stamp))
            self.state.last_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_image_error = str(exc)

    def on_compressed_image(self, msg: CompressedImage) -> None:
        try:
            rgb = compressed_image_to_rgb(msg)
            self.state.image_rgb = rgb
            self.state.image_frame_id = msg.header.frame_id
            self.state.image_stamp = msg.header.stamp
            self.image_buffer.append(TimedImageFrame(rgb=rgb, frame_id=msg.header.frame_id, stamp=msg.header.stamp))
            self.state.last_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_image_error = str(exc)

    def on_crop_image(self, msg: Image) -> None:
        try:
            rgb = raw_image_to_rgb(msg)
            self.state.crop_image_rgb = rgb
            self.state.crop_image_frame_id = msg.header.frame_id
            self.state.crop_image_stamp = msg.header.stamp
            self.crop_image_buffer.append(TimedImageFrame(rgb=rgb, frame_id=msg.header.frame_id, stamp=msg.header.stamp))
            self.state.last_crop_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_crop_image_error = str(exc)

    def on_crop_compressed_image(self, msg: CompressedImage) -> None:
        try:
            rgb = compressed_image_to_rgb(msg)
            self.state.crop_image_rgb = rgb
            self.state.crop_image_frame_id = msg.header.frame_id
            self.state.crop_image_stamp = msg.header.stamp
            self.crop_image_buffer.append(TimedImageFrame(rgb=rgb, frame_id=msg.header.frame_id, stamp=msg.header.stamp))
            self.state.last_crop_image_error = ""
        except Exception as exc:  # noqa: BLE001
            self.state.last_crop_image_error = str(exc)

    def on_particles(self, msg: PoseArray) -> None:
        self.state.particles = msg
        self.particles_buffer.append(msg)

    def on_path(self, msg: Path) -> None:
        self.state.path = msg
        self.path_buffer.append(msg)

    def on_vo_path(self, msg: Path) -> None:
        self.state.vo_path = msg
        self.vo_path_buffer.append(msg)

    def on_global_path(self, msg: Path) -> None:
        self.state.global_path = msg
        self.global_path_buffer.append(msg)

    def on_local_path(self, msg: Path) -> None:
        self.state.local_path = msg
        self.local_path_buffer.append(msg)

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

    def nearest_buffer_item(
        self,
        buffer: object,
        target_stamp: object,
        stamp_getter: object,
        *,
        prefer_past: bool = False,
        allow_future_fallback: bool = True,
        enforce_tolerance: bool = True,
    ) -> Optional[object]:
        if not buffer:
            return None
        target_ns = stamp_to_ns(target_stamp)
        if target_ns is None:
            return buffer[-1]

        if prefer_past:
            best_past = None
            best_past_delta = None
            for item in reversed(buffer):
                item_ns = stamp_to_ns(stamp_getter(item))
                if item_ns is None or item_ns > target_ns:
                    continue
                delta = target_ns - item_ns
                if best_past_delta is None or delta < best_past_delta:
                    best_past = item
                    best_past_delta = delta
                if delta == 0:
                    break
            if best_past is not None:
                if (
                    enforce_tolerance
                    and self.sync_tolerance_ns > 0
                    and best_past_delta is not None
                    and best_past_delta > self.sync_tolerance_ns
                ):
                    return None
                return best_past
            if not allow_future_fallback:
                return None

        best_item = None
        best_delta = None
        for item in reversed(buffer):
            item_ns = stamp_to_ns(stamp_getter(item))
            if item_ns is None:
                continue
            delta = abs(item_ns - target_ns)
            if best_delta is None or delta < best_delta:
                best_item = item
                best_delta = delta
            if delta == 0:
                break

        if best_item is None:
            return None
        if enforce_tolerance and self.sync_tolerance_ns > 0 and best_delta is not None and best_delta > self.sync_tolerance_ns:
            return None
        return best_item

    def latest_buffer_item(self, buffer: object, fallback: Optional[object] = None) -> Optional[object]:
        if buffer:
            return buffer[-1]
        return fallback

    def select_pose_from_buffer(self, buffer: object, fallback: Optional[Pose2D], target_stamp: object) -> Optional[Pose2D]:
        if target_stamp is None:
            return self.latest_buffer_item(buffer, fallback)
        return self.nearest_buffer_item(
            buffer,
            target_stamp,
            lambda pose: pose.stamp,
            prefer_past=True,
        )

    def select_localization_pose(self, target_stamp: object) -> Optional[Pose2D]:
        return self.select_pose_from_buffer(self.localization_buffer, self.state.localization, target_stamp)

    def select_amcl_pose(self, target_stamp: object) -> Optional[Pose2D]:
        return self.select_pose_from_buffer(self.amcl_pose_buffer, self.state.amcl_pose, target_stamp)

    def select_initial_pose(self, target_stamp: object) -> Optional[Pose2D]:
        return self.select_pose_from_buffer(self.initial_pose_buffer, self.state.initial_pose, target_stamp)

    def select_scan(self, target_stamp: object) -> Optional[LaserScan]:
        if target_stamp is None:
            return self.latest_buffer_item(self.scan_buffer, self.state.scan)
        return self.nearest_buffer_item(
            self.scan_buffer,
            target_stamp,
            lambda msg: msg.header.stamp,
            prefer_past=True,
        )

    def select_particles(self, target_stamp: object) -> Optional[PoseArray]:
        if target_stamp is None:
            return self.latest_buffer_item(self.particles_buffer, self.state.particles)
        return self.nearest_buffer_item(
            self.particles_buffer,
            target_stamp,
            lambda msg: msg.header.stamp,
            prefer_past=True,
        )

    def select_timed_message(
        self,
        buffer: object,
        fallback: Optional[object],
        target_stamp: object,
        *,
        enforce_tolerance: bool = True,
        allow_future_fallback: bool = True,
    ) -> Optional[object]:
        if target_stamp is None:
            return self.latest_buffer_item(buffer, fallback)
        return self.nearest_buffer_item(
            buffer,
            target_stamp,
            lambda msg: msg.header.stamp,
            prefer_past=True,
            allow_future_fallback=allow_future_fallback,
            enforce_tolerance=enforce_tolerance,
        )

    def select_image_frame(self, target_stamp: object) -> Optional[TimedImageFrame]:
        if target_stamp is None:
            return self.latest_buffer_item(self.image_buffer)
        return self.nearest_buffer_item(
            self.image_buffer,
            target_stamp,
            lambda frame: frame.stamp,
            prefer_past=True,
        )

    def select_crop_image_frame(self, target_stamp: object) -> Optional[TimedImageFrame]:
        if target_stamp is None:
            return self.latest_buffer_item(self.crop_image_buffer)
        return self.nearest_buffer_item(
            self.crop_image_buffer,
            target_stamp,
            lambda frame: frame.stamp,
            prefer_past=True,
        )

    def select_scan_for_image(self, image_stamp: object) -> Optional[LaserScan]:
        return self.select_scan(image_stamp)

    def select_camera_info_for_image(self, image_stamp: object) -> Optional[CameraInfo]:
        nearest = self.nearest_buffer_item(
            self.camera_info_buffer,
            image_stamp,
            lambda msg: msg.header.stamp,
            prefer_past=True,
        )
        if nearest is not None:
            return nearest
        return self.camera_info_buffer[-1] if self.camera_info_buffer else self.state.camera_info

    def select_odom(self, target_stamp: object) -> Optional[TimedOdomState]:
        selected = self.select_timed_message(
            self.odom_buffer,
            None,
            target_stamp,
            enforce_tolerance=False,
            allow_future_fallback=False,
        )
        return selected if isinstance(selected, TimedOdomState) else None

    def select_path_message(self, buffer: object, fallback: Optional[Path], target_stamp: object) -> Optional[Path]:
        selected = self.select_timed_message(
            buffer,
            fallback,
            target_stamp,
            enforce_tolerance=False,
            allow_future_fallback=False,
        )
        return selected if isinstance(selected, Path) else None

    def select_reference(self) -> tuple[str, Optional[object]]:
        if self.toggles["localization"] and self.localization_buffer:
            return "loc", self.localization_buffer[-1].stamp
        if self.toggles["amcl"] and self.amcl_pose_buffer:
            return "amcl", self.amcl_pose_buffer[-1].stamp
        if self.toggles["initialpose"] and self.initial_pose_buffer:
            return "init", self.initial_pose_buffer[-1].stamp
        if self.toggles["scan"] and self.scan_buffer:
            return "scan", self.scan_buffer[-1].header.stamp
        if self.toggles["image"] and self.image_buffer:
            return "img", self.image_buffer[-1].stamp
        if self.toggles["crop"] and self.crop_image_buffer:
            return "crop", self.crop_image_buffer[-1].stamp
        if self.toggles["particles"] and self.particles_buffer:
            return "particles", self.particles_buffer[-1].header.stamp
        return "-", None

    def stamp_delta_ms_text(self, stamp: object, reference_stamp: object) -> str:
        stamp_ns = stamp_to_ns(stamp)
        reference_ns = stamp_to_ns(reference_stamp)
        if stamp_ns is None or reference_ns is None:
            return "-"
        return str(int(round(abs(stamp_ns - reference_ns) / 1_000_000.0)))

    def update_selection_state(self) -> None:
        reference_source, reference_stamp = self.select_reference()
        self.selection = SelectionState(
            reference_source=reference_source,
            reference_stamp=reference_stamp,
            localization=self.select_localization_pose(reference_stamp),
            amcl_pose=self.select_amcl_pose(reference_stamp),
            initial_pose=self.select_initial_pose(reference_stamp),
            scan=self.select_scan(reference_stamp),
            image=self.select_image_frame(reference_stamp),
            crop_image=self.select_crop_image_frame(reference_stamp),
            particles=self.select_particles(reference_stamp),
            odom=self.select_odom(reference_stamp),
            path=self.select_path_message(self.path_buffer, self.state.path, reference_stamp),
            vo_path=self.select_path_message(self.vo_path_buffer, self.state.vo_path, reference_stamp),
            global_path=self.select_path_message(self.global_path_buffer, self.state.global_path, reference_stamp),
            local_path=self.select_path_message(self.local_path_buffer, self.state.local_path, reference_stamp),
        )

        parts = [f"ref={reference_source}"]
        if reference_stamp is not None:
            scan_stamp = self.selection.scan.header.stamp if self.selection.scan is not None else None
            parts.append(f"scan={self.stamp_delta_ms_text(scan_stamp, reference_stamp) if scan_stamp is not None else '-'}")
            loc_stamp = self.selection.localization.stamp if self.selection.localization is not None else None
            parts.append(f"loc={self.stamp_delta_ms_text(loc_stamp, reference_stamp) if loc_stamp is not None else '-'}")
            if self.toggles["image"]:
                image_stamp = self.selection.image.stamp if self.selection.image is not None else None
                parts.append(f"img={self.stamp_delta_ms_text(image_stamp, reference_stamp) if image_stamp is not None else '-'}")
            if self.toggles["crop"]:
                crop_stamp = self.selection.crop_image.stamp if self.selection.crop_image is not None else None
                parts.append(f"crop={self.stamp_delta_ms_text(crop_stamp, reference_stamp) if crop_stamp is not None else '-'}")
        self.last_sync_status = "sync " + " ".join(parts)

    def lookup_transform_between_exact(self, target_frame: str, source_frame: str, stamp: object) -> Optional[object]:
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
            return None

    def lookup_transform_between(self, target_frame: str, source_frame: str, stamp: object) -> Optional[object]:
        exact = self.lookup_transform_between_exact(target_frame, source_frame, stamp)
        if exact is not None:
            return exact
        if not self.args.allow_latest_tf_fallback or self.tf_buffer is None:
            return None
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
        exact = self.lookup_transform_between_exact(map_state.frame_id, source_frame, stamp)
        if exact is not None:
            return exact
        if not self.args.allow_latest_tf_fallback:
            return None
        return self.lookup_transform_between(map_state.frame_id, source_frame, None)

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
        selected_scan = self.selection.scan
        selected_particles = self.selection.particles

        if self.toggles["sections"]:
            self.draw_sections(canvas, scale, ox, oy)
        if self.toggles["path"]:
            self.draw_path(canvas, scale, ox, oy, self.selection.path, (40, 180, 80))
        if self.toggles["vo_path"]:
            self.draw_path(canvas, scale, ox, oy, self.selection.vo_path, (245, 140, 35))
        if self.toggles["global_path"]:
            self.draw_path(canvas, scale, ox, oy, self.selection.global_path, (180, 40, 180))
        if self.toggles["local_path"]:
            self.draw_path(canvas, scale, ox, oy, self.selection.local_path, (40, 180, 180))
        if self.toggles["scan"]:
            self.draw_scan_on_map(canvas, scale, ox, oy, selected_scan)
        if self.toggles["particles"]:
            self.draw_particles(canvas, scale, ox, oy, selected_particles)
        if self.toggles["localization"]:
            pose = self.pose_in_map(self.selection.localization)
            if pose is not None:
                self.draw_pose(canvas, pose, scale, ox, oy, (45, 70, 245), "GL")
        if self.toggles["amcl"]:
            pose = self.pose_in_map(self.selection.amcl_pose)
            if pose is not None:
                self.draw_pose(canvas, pose, scale, ox, oy, (65, 205, 80), "AMCL")
        if self.toggles["initialpose"]:
            pose = self.pose_in_map(self.selection.initial_pose)
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

    def draw_scan_on_map(self, canvas: np.ndarray, scale: float, ox: int, oy: int, scan: Optional[LaserScan]) -> None:
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

    def draw_particles(self, canvas: np.ndarray, scale: float, ox: int, oy: int, particles: Optional[PoseArray]) -> None:
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

    def draw_path(self, canvas: np.ndarray, scale: float, ox: int, oy: int, path: Optional[Path], color: Color) -> None:
        map_state = self.state.map_state
        if path is None or map_state is None or not path.poses:
            return
        stride = max(1, len(path.poses) // max(1, self.args.path_max_points))
        sampled_poses = path.poses[::stride]
        if len(sampled_poses) < 2 and len(path.poses) >= 2:
            sampled_poses = [path.poses[0], path.poses[-1]]

        pixels = []
        tf_cache: dict[tuple[str, Optional[int]], Optional[object]] = {}
        for stamped in sampled_poses:
            frame_id = stamped.header.frame_id or path.header.frame_id
            x = float(stamped.pose.position.x)
            y = float(stamped.pose.position.y)
            if not frame_id or frame_id == map_state.frame_id:
                pixels.append(self.world_to_view_px(x, y, scale, ox, oy))
                continue

            pose_stamp = stamped.header.stamp
            pose_stamp_ns = stamp_to_ns(pose_stamp)
            if pose_stamp_ns is None or pose_stamp_ns == 0:
                pose_stamp = path.header.stamp
                pose_stamp_ns = stamp_to_ns(pose_stamp)

            cache_key = (frame_id, pose_stamp_ns)
            if cache_key not in tf_cache:
                tf_cache[cache_key] = self.lookup_transform(frame_id, pose_stamp)
            tf = tf_cache[cache_key]
            if tf is not None:
                x, y = transform_xy(x, y, tf)
            elif not self.args.assume_same_frame:
                continue
            pixels.append(self.world_to_view_px(x, y, scale, ox, oy))
        if len(pixels) >= 2:
            cv2.polylines(canvas, [np.asarray(pixels, dtype=np.int32)], False, color, 3, cv2.LINE_AA)

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
        selected_image = self.selection.image
        if not self.toggles["image"]:
            self.draw_empty_panel(canvas, rect, "image hidden", "image")
            return
        if selected_image is None:
            text = self.state.last_image_error or ("no synced image" if self.image_buffer else "waiting for image")
            self.draw_empty_panel(canvas, rect, text, "image")
            return
        x0, y0, w, h = rect
        canvas[y0 : y0 + h, x0 : x0 + w] = (18, 19, 22)
        image = selected_image.rgb
        if self.toggles["scan"]:
            image = self.draw_scan_on_image(image, selected_image)
        rgb = resize_to_fit(image, w, h)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ih, iw = bgr.shape[:2]
        ox = x0 + (w - iw) // 2
        oy = y0 + (h - ih) // 2
        canvas[oy : oy + ih, ox : ox + iw] = bgr
        self.draw_panel_title(canvas, rect, "image")

    def draw_crop_image_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int]) -> None:
        selected_crop = self.selection.crop_image
        if not self.toggles["crop"]:
            self.draw_empty_panel(canvas, rect, "crop hidden", "crop")
            return
        if selected_crop is None:
            text = self.state.last_crop_image_error or ("no synced crop image" if self.crop_image_buffer else "waiting for crop image")
            self.draw_empty_panel(canvas, rect, text, "crop")
            return
        x0, y0, w, h = rect
        canvas[y0 : y0 + h, x0 : x0 + w] = (18, 19, 22)
        image = selected_crop.rgb
        rgb = resize_to_fit(image, w, h)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        ih, iw = bgr.shape[:2]
        ox = x0 + (w - iw) // 2
        oy = y0 + (h - ih) // 2
        canvas[oy : oy + ih, ox : ox + iw] = bgr
        self.draw_panel_title(canvas, rect, "crop")

    def draw_scan_on_image(self, image_rgb: np.ndarray, image_frame: TimedImageFrame) -> np.ndarray:
        scan = self.select_scan_for_image(image_frame.stamp)
        overlay = image_rgb.copy()

        def draw_status(text: str, color: Color) -> np.ndarray:
            cv2.putText(overlay, text[:48], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, text[:48], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
            return overlay

        if scan is None:
            self.state.last_projection_error = "no synced scan"
            return draw_status("scan proj: no synced scan", (255, 120, 120))

        camera_info = self.select_camera_info_for_image(image_frame.stamp)
        if camera_info is None:
            self.state.last_projection_error = "camera_info missing"
            return draw_status("scan proj: camera_info missing", (255, 120, 120))

        camera_frame = camera_info.header.frame_id or image_frame.frame_id
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

    def draw_panel_title(self, canvas: np.ndarray, rect: tuple[int, int, int, int], title: str) -> None:
        x0, y0, _, _ = rect
        cv2.putText(canvas, title[:24], (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, title[:24], (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 235, 242), 1, cv2.LINE_AA)

    def draw_empty_panel(self, canvas: np.ndarray, rect: tuple[int, int, int, int], text: str, title: str = "") -> None:
        x0, y0, w, h = rect
        canvas[y0 : y0 + h, x0 : x0 + w] = (24, 25, 28)
        if title:
            self.draw_panel_title(canvas, rect, title)
        text_y = y0 + (54 if title else 30)
        cv2.putText(canvas, text[:60], (x0 + 12, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190, 195, 205), 1, cv2.LINE_AA)

    def toggle_named(self, name: str, source: str = "") -> None:
        self.toggles[name] = not self.toggles[name]
        state = "on" if self.toggles[name] else "off"
        prefix = f"{source} " if source else ""
        self.last_key_status = f"{prefix}{name}={state}"

    def handle_key(self, key: str) -> None:
        mapping = {
            "m": "map",
            "l": "localization",
            "a": "amcl",
            "u": "initialpose",
            "s": "scan",
            "i": "image",
            "r": "crop",
            "c": "sections",
            "g": "gates",
            "p": "particles",
            "t": "path",
            "v": "vo_path",
            "y": "global_path",
            "h": "local_path",
        }
        if key == "q":
            self.running = False
            return
        if key == " ":
            self.paused = not self.paused
            self.last_key_status = "paused" if self.paused else "running"
            return
        if key in mapping:
            self.toggle_named(mapping[key], "key")

    def mouse_to_canvas(self, x: int, y: int) -> tuple[int, int]:
        cols, lines = shutil.get_terminal_size(fallback=(80, 24))
        if 0 <= x < self.args.width and 0 <= y < self.args.height:
            return x, y
        if 0 <= x < cols and 0 <= y < lines:
            mapped_x = int(round((x + 0.5) * self.args.width / max(1, cols)))
            mapped_y = int(round((y + 0.5) * self.args.height / max(1, lines)))
            return mapped_x, mapped_y
        return x, y

    def handle_mouse(self, x: int, y: int, button: int, pressed: bool) -> None:
        if button != 0 or not pressed:
            return
        canvas_x, canvas_y = self.mouse_to_canvas(x, y)
        for name, rect in self.toggle_button_rects.items():
            rx, ry, rw, rh = rect
            if rx <= canvas_x < rx + rw and ry <= canvas_y < ry + rh:
                self.toggle_named(name, "click")
                return

    def render(self) -> None:
        if self.paused:
            return
        now = time.monotonic()
        if now - self.last_render_time < 1.0 / max(self.args.max_fps, 0.2):
            return
        self.last_render_time = now
        self.state.frames += 1
        self.update_selection_state()

        canvas = np.full((self.args.height, self.args.width, 3), (20, 21, 24), dtype=np.uint8)
        header_h = self.header_height()
        gap = 6
        has_main_image_panel = bool(self.args.image_topic)
        has_crop_image_panel = bool(self.args.crop_image_topic)
        has_right_panel = has_main_image_panel or has_crop_image_panel
        right_w = int(self.args.width * self.args.image_panel_ratio) if has_right_panel else 0
        map_rect = (0, header_h, self.args.width - right_w - (gap if has_right_panel else 0), self.args.height - header_h)
        self.draw_map_panel(canvas, map_rect)
        if has_right_panel:
            image_x0 = self.args.width - right_w
            image_y0 = header_h
            image_h = self.args.height - header_h
            if has_main_image_panel and has_crop_image_panel:
                split_gap = gap
                top_h = max(1, (image_h - split_gap) // 2)
                bottom_h = max(1, image_h - top_h - split_gap)
                image_rect = (image_x0, image_y0, right_w, top_h)
                crop_rect = (image_x0, image_y0 + top_h + split_gap, right_w, bottom_h)
                self.draw_image_panel(canvas, image_rect)
                self.draw_crop_image_panel(canvas, crop_rect)
                cv2.line(canvas, (image_x0, crop_rect[1] - split_gap // 2), (self.args.width, crop_rect[1] - split_gap // 2), (55, 58, 64), 1)
            elif has_main_image_panel:
                image_rect = (image_x0, image_y0, right_w, image_h)
                self.draw_image_panel(canvas, image_rect)
            else:
                crop_rect = (image_x0, image_y0, right_w, image_h)
                self.draw_crop_image_panel(canvas, crop_rect)
            cv2.line(canvas, (image_x0 - gap // 2, header_h), (image_x0 - gap // 2, self.args.height), (55, 58, 64), 1)
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
        header_h = self.header_height()
        cv2.rectangle(canvas, (0, 0), (self.args.width, header_h), (14, 15, 18), -1)
        loc = self.pose_status(self.selection.localization)
        amcl = self.pose_status(self.selection.amcl_pose)
        init = self.pose_status(self.selection.initial_pose)
        scan = self.selection.scan.header.frame_id if self.selection.scan is not None else ("stale" if self.scan_buffer else "-")
        image = "ok" if self.selection.image is not None else ("stale" if self.image_buffer else "-")
        crop = "ok" if self.selection.crop_image is not None else ("stale" if self.crop_image_buffer else "-")
        camera_info = "yes" if self.state.camera_info is not None else "-"
        odom = self.odom_status(self.selection.odom)
        line1 = f"frame={self.state.frames} loc={loc} amcl={amcl} init={init} scan={scan} section={self.state.current_section}"
        line2 = (
            f"odom={odom} img={image} crop={crop} cam={camera_info} {self.last_sync_status} "
            f"click buttons below or use keys, space pause, q quit {self.last_key_status}"
        )
        cv2.putText(canvas, line1[:180], (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        cv2.putText(canvas, line2[:180], (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 220, 228), 1, cv2.LINE_AA)
        self.draw_toggle_buttons(canvas)

    def draw_toggle_buttons(self, canvas: np.ndarray) -> None:
        self.toggle_button_rects = {}
        font_scale = 0.48
        for name, _key, text, rect in self.layout_toggle_buttons(self.args.width):
            x, y, w, h = rect
            self.toggle_button_rects[name] = rect
            enabled = self.toggles.get(name, False)
            bg = (58, 116, 210) if enabled else (46, 48, 54)
            border = (135, 180, 255) if enabled else (92, 96, 106)
            fg = (245, 247, 250) if enabled else (200, 204, 212)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), bg, -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)
            text_y = y + h - 7
            cv2.putText(canvas, text, (x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (x + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, 1, cv2.LINE_AA)

    def toggle_status_summary(self) -> str:
        labels = (
            ("m", "map"),
            ("l", "localization"),
            ("a", "amcl"),
            ("u", "initialpose"),
            ("s", "scan"),
            ("i", "image"),
            ("r", "crop"),
            ("c", "sections"),
            ("g", "gates"),
            ("p", "particles"),
            ("t", "path"),
            ("v", "vo_path"),
            ("y", "global_path"),
            ("h", "local_path"),
        )
        return " ".join(f"{key}:{'1' if self.toggles.get(name, False) else '0'}" for key, name in labels)

    def pose_status(self, pose: Optional[Pose2D]) -> str:
        if pose is None:
            return "-"
        if pose.covariance is None or len(pose.covariance) < 36:
            return pose.frame_id
        return f"{pose.frame_id}:{pose.covariance[0]:.2g}/{pose.covariance[7]:.2g}/{pose.covariance[35]:.2g}"

    def odom_status(self, odom: Optional[TimedOdomState]) -> str:
        if odom is None:
            return "-"
        frame_id = odom.frame_id or "-"
        child_frame_id = odom.child_frame_id or "-"
        return (
            f"{frame_id}->{child_frame_id} "
            f"v={odom.speed_mps:.2f}m/s "
            f"vx={odom.linear_x:.2f} "
            f"vy={odom.linear_y:.2f} "
            f"wz={odom.angular_z:.2f}"
        )


class RawTerminal:
    mouse_event_re = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([mM])")

    def __init__(self, enable_mouse: bool = True) -> None:
        self.enable_mouse = enable_mouse
        self.buffer = ""

    def __enter__(self) -> "RawTerminal":
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        if self.enable_mouse:
            sys.stdout.write("\033[?1000h\033[?1006h\033[?1016h")
            sys.stdout.flush()
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.enable_mouse:
            sys.stdout.write("\033[?1016l\033[?1006l\033[?1000l")
            sys.stdout.flush()
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_events(self) -> list[InputEvent]:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if ready:
            self.buffer += os.read(self.fd, 256).decode("utf-8", errors="ignore")

        events: list[InputEvent] = []
        while self.buffer:
            if not self.buffer.startswith("\x1b"):
                events.append(InputEvent(key=self.buffer[0]))
                self.buffer = self.buffer[1:]
                continue

            if self.buffer.startswith("\x1b[<"):
                match = self.mouse_event_re.match(self.buffer)
                if match is None:
                    if "M" not in self.buffer and "m" not in self.buffer:
                        break
                    self.buffer = self.buffer[1:]
                    continue
                code = int(match.group(1))
                x = max(0, int(match.group(2)) - 1)
                y = max(0, int(match.group(3)) - 1)
                pressed = match.group(4) == "M"
                button = code & 0b11
                events.append(
                    InputEvent(
                        mouse_x=x,
                        mouse_y=y,
                        mouse_button=button,
                        mouse_pressed=pressed,
                    )
                )
                self.buffer = self.buffer[match.end() :]
                continue

            if self.buffer.startswith("\x1b["):
                if len(self.buffer) < 3:
                    break
                seq_end = None
                for idx, ch in enumerate(self.buffer[2:], start=2):
                    if ch.isalpha() or ch == "~":
                        seq_end = idx
                        break
                if seq_end is None:
                    break
                self.buffer = self.buffer[seq_end + 1 :]
                continue

            if len(self.buffer) == 1:
                break
            self.buffer = self.buffer[1:]

        return events


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
    parser.add_argument(
        "--crop-image-topic",
        default="/perception/crop/image",
        help="Image topic for perception crop preview",
    )
    parser.add_argument(
        "--crop-compressed-image",
        action="store_true",
        help="Treat --crop-image-topic as sensor_msgs/CompressedImage",
    )
    parser.add_argument("--particles-topic", default="/particle_cloud")
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--vo-path-topic", default="/visual_slam/tracking/vo_path")
    parser.add_argument("--global-path-topic", default="/planning/global_raceline")
    parser.add_argument("--local-path-topic", default="/autonomous/trajectory")
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
    parser.add_argument(
        "--allow-latest-tf-fallback",
        action="store_true",
        help="if exact-timestamp TF is unavailable, fall back to latest TF instead of hiding the overlay",
    )
    parser.add_argument(
        "--sync-buffer-size",
        type=int,
        default=24,
        help="recent message count kept per topic for nearest-timestamp matching",
    )
    parser.add_argument(
        "--state-sync-buffer-size",
        type=int,
        default=180,
        help="recent message count kept for path and odom history when replaying delayed logs",
    )
    parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=120.0,
        help="max timestamp delta for nearest-message matching before overlays are hidden",
    )
    parser.add_argument("--scan-stride", type=int, default=3)
    parser.add_argument("--scan-radius", type=int, default=2)
    parser.add_argument("--particle-stride", type=int, default=1)
    parser.add_argument("--path-max-points", type=int, default=600)
    parser.add_argument("--image-panel-ratio", type=float, default=0.34)
    parser.add_argument("--png-compression", type=int, default=3)
    parser.add_argument(
        "--no-mouse",
        action="store_true",
        help="disable terminal mouse support and use keyboard toggles only",
    )
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
        with RawTerminal(enable_mouse=not args.no_mouse) if sys.stdin.isatty() else null_terminal() as terminal:
            while rclpy.ok() and node.running:
                rclpy.spin_once(node, timeout_sec=0.02)
                for event in terminal.read_events():
                    if event.key:
                        node.handle_key(event.key)
                    elif event.mouse_x is not None and event.mouse_y is not None and event.mouse_button is not None:
                        node.handle_mouse(event.mouse_x, event.mouse_y, event.mouse_button, event.mouse_pressed)
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

    def read_events(self) -> list[InputEvent]:
        return []


if __name__ == "__main__":
    raise SystemExit(main())

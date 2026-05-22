#!/usr/bin/env python3
"""Publish lane markers and a primary centerline path from an editable HD map YAML."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import rclpy
import yaml
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker, MarkerArray


Point3 = Tuple[float, float, float]


@dataclass
class Lane:
    lane_id: str
    closed_loop: bool
    left_bound: List[Point3]
    right_bound: List[Point3]
    centerline: List[Point3]


@dataclass
class HdMap:
    frame_id: str
    primary_lane_id: str
    lanes: List[Lane]

    def primary_lane(self) -> Optional[Lane]:
        for lane in self.lanes:
            if lane.lane_id == self.primary_lane_id:
                return lane
        return self.lanes[0] if self.lanes else None


def read_points(rows: object) -> List[Point3]:
    points: List[Point3] = []
    if not isinstance(rows, list):
        return points
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        z = float(row[2]) if len(row) >= 3 else 0.0
        points.append((float(row[0]), float(row[1]), z))
    return points


def load_hd_map(path: Path, frame_override: str) -> HdMap:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"HD map YAML root must be a mapping: {path}")

    raw_lanes = data.get("lanes", [])
    if not isinstance(raw_lanes, list):
        raise RuntimeError(f"HD map YAML lanes must be a list: {path}")

    lanes: List[Lane] = []
    for lane_index, raw_lane in enumerate(raw_lanes, start=1):
        if not isinstance(raw_lane, dict):
            continue
        lanes.append(
            Lane(
                lane_id=str(raw_lane.get("id") or f"lane_{lane_index:03d}"),
                closed_loop=bool(raw_lane.get("closed_loop", True)),
                left_bound=read_points(raw_lane.get("left_bound", [])),
                right_bound=read_points(raw_lane.get("right_bound", [])),
                centerline=read_points(raw_lane.get("centerline", [])),
            )
        )

    if not lanes:
        raise RuntimeError(f"HD map YAML has no lanes: {path}")

    primary_lane_id = str(data.get("primary_lane_id") or lanes[0].lane_id)
    if primary_lane_id not in {lane.lane_id for lane in lanes}:
        primary_lane_id = lanes[0].lane_id

    return HdMap(
        frame_id=frame_override or str(data.get("frame_id") or "map"),
        primary_lane_id=primary_lane_id,
        lanes=lanes,
    )


def transient_local_qos() -> QoSProfile:
    return QoSProfile(
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def color_for_field(marker: Marker, field_name: str) -> None:
    marker.color.a = 0.95
    if field_name == "left_bound":
        marker.color.r = 0.15
        marker.color.g = 0.95
        marker.color.b = 0.25
    elif field_name == "right_bound":
        marker.color.r = 0.95
        marker.color.g = 0.20
        marker.color.b = 0.95
    else:
        marker.color.r = 1.0
        marker.color.g = 0.85
        marker.color.b = 0.05


def to_geometry_point(point: Point3, z_offset: float) -> Point:
    msg = Point()
    msg.x = point[0]
    msg.y = point[1]
    msg.z = point[2] + z_offset
    return msg


def yaw_at(points: Sequence[Point3], index: int, closed_loop: bool) -> float:
    if len(points) < 2:
        return 0.0
    if closed_loop and len(points) >= 3:
        prev_point = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
    elif index == len(points) - 1:
        prev_point = points[index - 1]
        next_point = points[index]
    else:
        prev_point = points[index]
        next_point = points[index + 1]
    return math.atan2(next_point[1] - prev_point[1], next_point[0] - prev_point[0])


class HdMapPublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("hd_map_publisher")

        self.hd_map_yaml_path = self.declare_parameter("hd_map_yaml_path", "").value
        self.frame_id_override = self.declare_parameter("frame_id_override", "").value
        self.publish_rate_hz = max(float(self.declare_parameter("publish_rate_hz", 1.0).value), 0.1)
        self.retry_interval_sec = max(float(self.declare_parameter("retry_interval_sec", 2.0).value), 0.1)
        self.publish_lane_markers = bool(self.declare_parameter("publish_lane_markers", True).value)
        self.publish_primary_path = bool(self.declare_parameter("publish_primary_path", True).value)
        self.marker_line_width_m = max(
            float(self.declare_parameter("marker_line_width_m", 0.03).value),
            0.001,
        )
        self.primary_marker_scale = max(
            float(self.declare_parameter("primary_marker_scale", 1.35).value),
            1.0,
        )
        self.marker_z_offset_m = float(self.declare_parameter("marker_z_offset_m", 0.02).value)
        self.path_z_offset_m = float(self.declare_parameter("path_z_offset_m", 0.0).value)

        qos = transient_local_qos()
        self.marker_pub = (
            self.create_publisher(MarkerArray, "lane_markers", qos)
            if self.publish_lane_markers
            else None
        )
        self.path_pub = (
            self.create_publisher(PathMsg, "primary_centerline_path", qos)
            if self.publish_primary_path
            else None
        )

        self.hd_map: Optional[HdMap] = None
        self.primary_path_warning_logged = False
        self.last_load_attempt = self.get_clock().now()
        self.try_load_hd_map()
        self.timer = self.create_timer(1.0 / self.publish_rate_hz, self.publish_outputs)

    def try_load_hd_map(self) -> bool:
        self.last_load_attempt = self.get_clock().now()
        if not self.hd_map_yaml_path:
            self.get_logger().warn("Parameter hd_map_yaml_path is empty. Waiting for a YAML path.")
            return False

        path = Path(self.hd_map_yaml_path).expanduser()
        if not path.is_file():
            self.get_logger().error(f"HD map YAML not found: {path}")
            return False

        try:
            self.hd_map = load_hd_map(path.resolve(), str(self.frame_id_override))
        except Exception as exc:
            self.get_logger().error(f"Failed to load HD map YAML {path}: {exc}")
            self.hd_map = None
            return False

        primary_lane = self.hd_map.primary_lane()
        primary_points = len(primary_lane.centerline) if primary_lane is not None else 0
        self.get_logger().info(
            f"Loaded HD map YAML: {path} "
            f"(frame={self.hd_map.frame_id}, lanes={len(self.hd_map.lanes)}, "
            f"primary={self.hd_map.primary_lane_id}, primary_points={primary_points})"
        )
        return True

    def publish_outputs(self) -> None:
        if self.hd_map is None:
            now = self.get_clock().now()
            elapsed = max(0.0, (now.nanoseconds - self.last_load_attempt.nanoseconds) / 1.0e9)
            if elapsed >= self.retry_interval_sec:
                self.try_load_hd_map()
            return

        stamp = self.get_clock().now().to_msg()
        if self.marker_pub is not None:
            self.marker_pub.publish(self.build_lane_markers(stamp))
        if self.path_pub is not None:
            path = self.build_primary_path(stamp)
            if path is not None:
                self.path_pub.publish(path)

    def build_lane_markers(self, stamp) -> MarkerArray:
        marker_array = MarkerArray()
        marker_id = 0
        for lane in self.hd_map.lanes:
            for field_name in ("left_bound", "right_bound", "centerline"):
                points = getattr(lane, field_name)
                if len(points) < 2:
                    continue
                marker = Marker()
                marker.header.frame_id = self.hd_map.frame_id
                marker.header.stamp = stamp
                marker.ns = f"hd_map/{lane.lane_id}/{field_name}"
                marker.id = marker_id
                marker_id += 1
                marker.type = Marker.LINE_STRIP
                marker.action = Marker.ADD
                marker.pose.orientation.w = 1.0
                marker.scale.x = self.marker_line_width_m
                if lane.lane_id == self.hd_map.primary_lane_id:
                    marker.scale.x *= self.primary_marker_scale
                color_for_field(marker, field_name)
                marker.points = [to_geometry_point(point, self.marker_z_offset_m) for point in points]
                if lane.closed_loop and len(points) >= 3:
                    marker.points.append(to_geometry_point(points[0], self.marker_z_offset_m))
                marker_array.markers.append(marker)
        return marker_array

    def build_primary_path(self, stamp) -> Optional[PathMsg]:
        lane = self.hd_map.primary_lane()
        if lane is None or len(lane.centerline) < 2:
            if not self.primary_path_warning_logged:
                self.get_logger().warn(
                    "Primary lane has fewer than two centerline points. Skip primary path output."
                )
                self.primary_path_warning_logged = True
            return None

        path = PathMsg()
        path.header.frame_id = self.hd_map.frame_id
        path.header.stamp = stamp
        for index, point in enumerate(lane.centerline):
            yaw = yaw_at(lane.centerline, index, lane.closed_loop)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2] + self.path_z_offset_m
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)
        return path


def main() -> None:
    rclpy.init()
    node = HdMapPublisherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

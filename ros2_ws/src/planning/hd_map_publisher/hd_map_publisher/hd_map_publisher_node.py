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
class SectionGate:
    gate_id: str
    lane_id: str
    s_m: float
    line: List[Point3]


@dataclass
class Section:
    section_id: str
    lane_id: str
    start_s_m: float
    end_s_m: float
    speed_override_mps: Optional[float]


@dataclass
class HdMap:
    frame_id: str
    primary_lane_id: str
    lanes: List[Lane]
    section_gates: List[SectionGate]
    sections: List[Section]

    def primary_lane(self) -> Optional[Lane]:
        for lane in self.lanes:
            if lane.lane_id == self.primary_lane_id:
                return lane
        return self.lanes[0] if self.lanes else None

    def lane_by_id(self, lane_id: str) -> Optional[Lane]:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return None


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


def read_float_or_none(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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

    section_gates: List[SectionGate] = []
    raw_gates = data.get("section_gates", [])
    if isinstance(raw_gates, list):
        for gate_index, raw_gate in enumerate(raw_gates, start=1):
            if not isinstance(raw_gate, dict):
                continue
            line = read_points(raw_gate.get("line", []))
            if len(line) < 2:
                continue
            section_gates.append(
                SectionGate(
                    gate_id=str(raw_gate.get("id") or f"gate_{gate_index:03d}"),
                    lane_id=str(raw_gate.get("lane_id") or primary_lane_id),
                    s_m=float(raw_gate.get("s_m", 0.0)),
                    line=line[:2],
                )
            )

    sections: List[Section] = []
    raw_sections = data.get("sections", [])
    if isinstance(raw_sections, list):
        for section_index, raw_section in enumerate(raw_sections, start=1):
            if not isinstance(raw_section, dict):
                continue
            try:
                start_s_m = float(raw_section.get("start_s_m", 0.0))
                end_s_m = float(raw_section.get("end_s_m", 0.0))
            except (TypeError, ValueError):
                continue
            sections.append(
                Section(
                    section_id=str(raw_section.get("id") or f"section_{section_index:03d}"),
                    lane_id=str(raw_section.get("lane_id") or primary_lane_id),
                    start_s_m=start_s_m,
                    end_s_m=end_s_m,
                    speed_override_mps=read_float_or_none(raw_section.get("speed_override_mps")),
                )
            )

    return HdMap(
        frame_id=frame_override or str(data.get("frame_id") or "map"),
        primary_lane_id=primary_lane_id,
        lanes=lanes,
        section_gates=section_gates,
        sections=sections,
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


def distance_2d(a: Point3, b: Point3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def cumulative_s(points: Sequence[Point3], closed_loop: bool) -> Tuple[List[float], float]:
    if not points:
        return [], 0.0
    s_values = [0.0]
    total = 0.0
    for index in range(1, len(points)):
        total += distance_2d(points[index - 1], points[index])
        s_values.append(total)
    if closed_loop and len(points) >= 3:
        total += distance_2d(points[-1], points[0])
    return s_values, total


def interpolate_at_s(points: Sequence[Point3], s_values: Sequence[float], total_length: float, target_s: float, closed_loop: bool) -> Point3:
    if not points:
        return (0.0, 0.0, 0.0)
    if len(points) == 1 or total_length <= 1.0e-9:
        return points[0]
    target = target_s % total_length if closed_loop else max(0.0, min(target_s, total_length))
    segment_count = len(points) if closed_loop and len(points) >= 3 else len(points) - 1
    for index in range(segment_count):
        next_index = (index + 1) % len(points)
        start_s = s_values[index]
        segment_length = distance_2d(points[index], points[next_index])
        end_s = start_s + segment_length
        if index == segment_count - 1 and closed_loop:
            end_s = total_length
        if start_s - 1.0e-9 <= target <= end_s + 1.0e-9:
            ratio = 0.0 if segment_length <= 1.0e-9 else (target - start_s) / segment_length
            a = points[index]
            b = points[next_index]
            return (
                a[0] + (b[0] - a[0]) * ratio,
                a[1] + (b[1] - a[1]) * ratio,
                a[2] + (b[2] - a[2]) * ratio,
            )
    return points[-1]


def section_points(points: Sequence[Point3], closed_loop: bool, start_s: float, end_s: float) -> List[Point3]:
    s_values, total_length = cumulative_s(points, closed_loop)
    if len(points) < 2 or total_length <= 1.0e-9:
        return []

    def collect_range(start: float, end: float) -> List[Point3]:
        collected = [interpolate_at_s(points, s_values, total_length, start, closed_loop)]
        for point, point_s in zip(points, s_values):
            if start < point_s < end:
                collected.append(point)
        collected.append(interpolate_at_s(points, s_values, total_length, end, closed_loop))
        return collected

    start = start_s % total_length if closed_loop else max(0.0, min(start_s, total_length))
    end = end_s % total_length if closed_loop else max(0.0, min(end_s, total_length))
    if closed_loop and start > end:
        return collect_range(start, total_length) + collect_range(0.0, end)[1:]
    return collect_range(start, end)


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
        self.publish_section_markers = bool(self.declare_parameter("publish_section_markers", True).value)
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
        self.section_marker_pub = (
            self.create_publisher(MarkerArray, "section_markers", qos)
            if self.publish_section_markers
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
            f"primary={self.hd_map.primary_lane_id}, primary_points={primary_points}, "
            f"sections={len(self.hd_map.sections)}, gates={len(self.hd_map.section_gates)})"
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
        if self.section_marker_pub is not None:
            self.section_marker_pub.publish(self.build_section_markers(stamp))
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

    def build_section_markers(self, stamp) -> MarkerArray:
        marker_array = MarkerArray()
        marker_id = 0

        for section in self.hd_map.sections:
            lane = self.hd_map.lane_by_id(section.lane_id)
            if lane is None or len(lane.centerline) < 2:
                continue
            points = section_points(lane.centerline, lane.closed_loop, section.start_s_m, section.end_s_m)
            if len(points) < 2:
                continue
            marker = Marker()
            marker.header.frame_id = self.hd_map.frame_id
            marker.header.stamp = stamp
            marker.ns = f"hd_map/section/{section.lane_id}"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.marker_line_width_m * 2.2
            marker.color.a = 0.9
            if section.speed_override_mps is None:
                marker.color.r = 0.1
                marker.color.g = 0.65
                marker.color.b = 1.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.45
                marker.color.b = 0.05
            marker.points = [to_geometry_point(point, self.marker_z_offset_m + 0.05) for point in points]
            marker_array.markers.append(marker)

            label = Marker()
            label.header.frame_id = self.hd_map.frame_id
            label.header.stamp = stamp
            label.ns = f"hd_map/section_label/{section.lane_id}"
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.orientation.w = 1.0
            mid = points[len(points) // 2]
            label.pose.position.x = mid[0]
            label.pose.position.y = mid[1]
            label.pose.position.z = mid[2] + self.marker_z_offset_m + 0.25
            label.scale.z = 0.22
            label.color.r = marker.color.r
            label.color.g = marker.color.g
            label.color.b = marker.color.b
            label.color.a = 1.0
            if section.speed_override_mps is None:
                label.text = section.section_id
            else:
                label.text = f"{section.section_id} {section.speed_override_mps:.2f}m/s"
            marker_array.markers.append(label)

        for gate in self.hd_map.section_gates:
            if len(gate.line) < 2:
                continue
            marker = Marker()
            marker.header.frame_id = self.hd_map.frame_id
            marker.header.stamp = stamp
            marker.ns = f"hd_map/section_gate/{gate.lane_id}"
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.LINE_LIST
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.marker_line_width_m * 2.6
            marker.color.r = 1.0
            marker.color.g = 0.72
            marker.color.b = 0.1
            marker.color.a = 1.0
            marker.points = [to_geometry_point(point, self.marker_z_offset_m + 0.08) for point in gate.line[:2]]
            marker_array.markers.append(marker)

            label = Marker()
            label.header.frame_id = self.hd_map.frame_id
            label.header.stamp = stamp
            label.ns = f"hd_map/section_gate_label/{gate.lane_id}"
            label.id = marker_id
            marker_id += 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.orientation.w = 1.0
            label.pose.position.x = 0.5 * (gate.line[0][0] + gate.line[1][0])
            label.pose.position.y = 0.5 * (gate.line[0][1] + gate.line[1][1])
            label.pose.position.z = 0.5 * (gate.line[0][2] + gate.line[1][2]) + self.marker_z_offset_m + 0.28
            label.scale.z = 0.2
            label.color.r = 1.0
            label.color.g = 0.9
            label.color.b = 0.25
            label.color.a = 1.0
            label.text = gate.gate_id
            marker_array.markers.append(label)

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
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

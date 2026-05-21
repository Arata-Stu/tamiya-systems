#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener


def parse_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip()
    return data


def parse_origin(origin_value: object) -> tuple[float, float, float]:
    if isinstance(origin_value, str):
        origin_value = literal_eval(origin_value)
    if isinstance(origin_value, (list, tuple)) and len(origin_value) >= 2:
        yaw = float(origin_value[2]) if len(origin_value) >= 3 else 0.0
        return float(origin_value[0]), float(origin_value[1]), yaw
    raise ValueError("origin must be a list-like value [x, y, yaw]")


@dataclass
class RasterGeometry:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float


@dataclass
class MapYamlTemplate:
    negate: float = 0.0
    occupied_thresh: float = 0.65
    free_thresh: float = 0.196


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, transform_msg) -> np.ndarray:
    rotation = transform_msg.transform.rotation
    translation = transform_msg.transform.translation
    rot = quaternion_to_rotation_matrix(rotation.x, rotation.y, rotation.z, rotation.w)
    translated = points @ rot.T
    translated[:, 0] += translation.x
    translated[:, 1] += translation.y
    translated[:, 2] += translation.z
    return translated


def points_to_pixels(points_xy: np.ndarray, geometry: RasterGeometry) -> np.ndarray:
    dx = points_xy[:, 0] - geometry.origin_x
    dy = points_xy[:, 1] - geometry.origin_y
    cos_t = math.cos(geometry.origin_yaw)
    sin_t = math.sin(geometry.origin_yaw)
    grid_x = (cos_t * dx + sin_t * dy) / geometry.resolution
    grid_y = (-sin_t * dx + cos_t * dy) / geometry.resolution
    img_x = np.round(grid_x).astype(np.int32)
    img_y = np.round((geometry.height - 1) - grid_y).astype(np.int32)
    return np.column_stack([img_x, img_y])


def filter_pixels_inside(pixels: np.ndarray, width: int, height: int) -> np.ndarray:
    if pixels.size == 0:
        return pixels
    mask = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < height)
    )
    return pixels[mask]


def write_map_yaml(
    output_yaml: Path,
    image_path: Path,
    geometry: RasterGeometry,
    template: MapYamlTemplate,
) -> None:
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    image_value = image_path.name if image_path.parent == output_yaml.parent else str(image_path)
    lines = [
        f'image: "{image_value}"',
        f"resolution: {geometry.resolution:.12g}",
        f"origin: [{geometry.origin_x:.12g}, {geometry.origin_y:.12g}, {geometry.origin_yaw:.12g}]",
        f"negate: {template.negate:.12g}",
        f"occupied_thresh: {template.occupied_thresh:.12g}",
        f"free_thresh: {template.free_thresh:.12g}",
    ]
    output_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


class LandmarkRasterExporter(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("landmark_raster_exporter")
        self.args = args
        self.latest_landmarks: Optional[PointCloud2] = None
        self.latest_path: Optional[PathMsg] = None
        self.completed = False
        self.start_time_ns = self.get_clock().now().nanoseconds

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PointCloud2, args.landmarks_topic, self.on_landmarks, 10)
        if args.path_topic:
            self.create_subscription(PathMsg, args.path_topic, self.on_path, 10)

        self.timer = self.create_timer(0.2, self.try_export)

    def finish(self, message: str, *, error: bool = False) -> None:
        if self.completed:
            return
        self.completed = True
        self.timer.cancel()
        if error:
            self.get_logger().error(message)
        else:
            self.get_logger().info(message)
        rclpy.shutdown()

    def on_landmarks(self, msg: PointCloud2) -> None:
        self.latest_landmarks = msg

    def on_path(self, msg: PathMsg) -> None:
        self.latest_path = msg

    def read_landmark_points(self, msg: PointCloud2) -> np.ndarray:
        rows = list(point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if not rows:
            return np.empty((0, 3), dtype=np.float64)
        points = np.asarray(rows, dtype=np.float64)
        if self.args.min_z is not None:
            points = points[points[:, 2] >= self.args.min_z]
        if self.args.max_z is not None:
            points = points[points[:, 2] <= self.args.max_z]
        return points

    def transform_if_needed(self, points: np.ndarray, source_frame: str) -> np.ndarray:
        if points.size == 0 or not self.args.target_frame or self.args.target_frame == source_frame:
            return points
        transform = self.tf_buffer.lookup_transform(
            self.args.target_frame, source_frame, Time()
        )
        return transform_points(points, transform)

    def path_points_in_target_frame(self, msg: PathMsg) -> np.ndarray:
        if not msg.poses:
            return np.empty((0, 3), dtype=np.float64)

        points = np.array(
            [
                [pose.pose.position.x, pose.pose.position.y, pose.pose.position.z]
                for pose in msg.poses
            ],
            dtype=np.float64,
        )
        source_frame = msg.header.frame_id
        return self.transform_if_needed(points, source_frame)

    def resolve_geometry(self, landmark_xy: np.ndarray, path_xy: np.ndarray) -> tuple[RasterGeometry, MapYamlTemplate]:
        if self.args.reference_yaml:
            ref_yaml = Path(self.args.reference_yaml).expanduser().resolve()
            ref_data = parse_simple_yaml(ref_yaml)
            ref_image_value = str(ref_data["image"]).strip().strip('"').strip("'")
            ref_image_path = (ref_yaml.parent / ref_image_value).resolve()
            image = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(f"Could not read reference map image: {ref_image_path}")
            origin_x, origin_y, origin_yaw = parse_origin(ref_data["origin"])
            geometry = RasterGeometry(
                width=int(image.shape[1]),
                height=int(image.shape[0]),
                resolution=float(ref_data["resolution"]),
                origin_x=origin_x,
                origin_y=origin_y,
                origin_yaw=origin_yaw,
            )
            template = MapYamlTemplate(
                negate=float(ref_data.get("negate", 0.0)),
                occupied_thresh=float(ref_data.get("occupied_thresh", 0.65)),
                free_thresh=float(ref_data.get("free_thresh", 0.196)),
            )
            return geometry, template

        all_points = [arr for arr in (landmark_xy, path_xy) if arr.size > 0]
        if not all_points:
            raise RuntimeError("No landmarks or path points were available to compute raster bounds.")

        stacked = np.vstack(all_points)
        padding = max(0.0, float(self.args.padding_m))
        min_x = float(np.min(stacked[:, 0]) - padding)
        min_y = float(np.min(stacked[:, 1]) - padding)
        max_x = float(np.max(stacked[:, 0]) + padding)
        max_y = float(np.max(stacked[:, 1]) + padding)
        resolution = max(1.0e-4, float(self.args.resolution))
        width = max(1, int(math.ceil((max_x - min_x) / resolution)) + 1)
        height = max(1, int(math.ceil((max_y - min_y) / resolution)) + 1)
        geometry = RasterGeometry(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=min_x,
            origin_y=min_y,
            origin_yaw=0.0,
        )
        return geometry, MapYamlTemplate()

    def draw_landmarks(self, canvas: np.ndarray, pixels: np.ndarray) -> None:
        if pixels.size == 0:
            return
        if self.args.point_radius_px <= 0:
            canvas[pixels[:, 1], pixels[:, 0]] = self.args.landmark_value
            return
        for px, py in pixels:
            cv2.circle(canvas, (int(px), int(py)), self.args.point_radius_px, self.args.landmark_value, -1, cv2.LINE_AA)

    def draw_path(self, canvas: np.ndarray, pixels: np.ndarray) -> None:
        if pixels.shape[0] < 2:
            return
        cv2.polylines(
            canvas,
            [pixels.reshape((-1, 1, 2)).astype(np.int32)],
            isClosed=False,
            color=int(self.args.path_value),
            thickness=max(1, int(self.args.path_thickness_px)),
            lineType=cv2.LINE_AA,
        )

    def try_export(self) -> None:
        if self.completed:
            return

        elapsed_sec = (self.get_clock().now().nanoseconds - self.start_time_ns) / 1.0e9
        if self.latest_landmarks is None:
            if elapsed_sec > self.args.timeout_sec:
                self.finish("Timed out waiting for landmarks topic.", error=True)
            return

        if self.args.require_path and self.args.path_topic and self.latest_path is None:
            if elapsed_sec > self.args.timeout_sec:
                self.finish("Timed out waiting for path topic.", error=True)
            return

        try:
            landmark_points = self.read_landmark_points(self.latest_landmarks)
            landmark_points = self.transform_if_needed(landmark_points, self.latest_landmarks.header.frame_id)
            path_points = (
                self.path_points_in_target_frame(self.latest_path)
                if self.latest_path is not None and self.args.path_topic
                else np.empty((0, 3), dtype=np.float64)
            )

            landmark_xy = (
                landmark_points[:, :2] if landmark_points.size > 0 else np.empty((0, 2), dtype=np.float64)
            )
            path_xy = path_points[:, :2] if path_points.size > 0 else np.empty((0, 2), dtype=np.float64)
            geometry, yaml_template = self.resolve_geometry(landmark_xy, path_xy)

            canvas = np.full(
                (geometry.height, geometry.width),
                int(self.args.background_value),
                dtype=np.uint8,
            )
            landmark_pixels = filter_pixels_inside(
                points_to_pixels(landmark_xy, geometry), geometry.width, geometry.height
            )
            path_pixels = filter_pixels_inside(
                points_to_pixels(path_xy, geometry), geometry.width, geometry.height
            )

            self.draw_landmarks(canvas, landmark_pixels)
            self.draw_path(canvas, path_pixels)

            output_image = Path(self.args.output_image).expanduser().resolve()
            output_image.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_image), canvas):
                raise RuntimeError(f"Failed to write image: {output_image}")

            if self.args.output_yaml:
                output_yaml = Path(self.args.output_yaml).expanduser().resolve()
                write_map_yaml(output_yaml, output_image, geometry, yaml_template)
                self.get_logger().info(f"Wrote map yaml: {output_yaml}")

            self.finish(
                "Wrote landmark raster: "
                f"{output_image} (landmarks={landmark_pixels.shape[0]}, path_points={path_pixels.shape[0]}, "
                f"frame={self.args.target_frame or self.latest_landmarks.header.frame_id})"
            )
        except TransformException:
            if elapsed_sec > self.args.timeout_sec:
                self.finish("Timed out waiting for TF needed to rasterize landmarks.", error=True)
        except Exception as exc:
            self.finish(f"Failed to export landmark raster: {exc}", error=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export VSLAM landmark visualization to a 2D PNG.")
    parser.add_argument("--landmarks-topic", default="/visual_slam/vis/landmarks_cloud")
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--require-path", action="store_true")
    parser.add_argument("--target-frame", default="map")
    parser.add_argument("--reference-yaml", default="")
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output-yaml", default="")
    parser.add_argument("--resolution", type=float, default=0.02)
    parser.add_argument("--padding-m", type=float, default=0.5)
    parser.add_argument("--background-value", type=int, default=255)
    parser.add_argument("--landmark-value", type=int, default=0)
    parser.add_argument("--path-value", type=int, default=96)
    parser.add_argument("--point-radius-px", type=int, default=1)
    parser.add_argument("--path-thickness-px", type=int, default=1)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    cli_args = parser.parse_args(args=rclpy.utilities.remove_ros_args()[1:])

    if not cli_args.target_frame:
        cli_args.target_frame = ""
    if not cli_args.reference_yaml:
        cli_args.reference_yaml = ""
    if not cli_args.output_yaml:
        cli_args.output_yaml = ""
    if not cli_args.path_topic:
        cli_args.path_topic = ""

    rclpy.init()
    node = LandmarkRasterExporter(cli_args)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()

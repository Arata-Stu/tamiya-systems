#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import math
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


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


def euler_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z


def get_transformation_matrix(x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cr = math.cos(roll)
    sr = math.sin(roll)

    rot = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr]
    ])

    return rot, np.array([x, y, z])


def transform_points(points: np.ndarray, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points
    translated = points @ rot.T
    translated[:, 0] += trans[0]
    translated[:, 1] += trans[1]
    translated[:, 2] += trans[2]
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


def read_landmark_points(landmarks_data: dict, min_z: Optional[float], max_z: Optional[float]) -> np.ndarray:
    import struct
    
    data_bytes = base64.b64decode(landmarks_data["data"])
    point_step = landmarks_data["point_step"]
    num_points = len(data_bytes) // point_step
    
    x_offset, y_offset, z_offset = -1, -1, -1
    for f in landmarks_data["fields"]:
        if f["name"] == "x": x_offset = f["offset"]
        if f["name"] == "y": y_offset = f["offset"]
        if f["name"] == "z": z_offset = f["offset"]
        
    if x_offset == -1 or y_offset == -1 or z_offset == -1:
        return np.empty((0, 3), dtype=np.float64)
        
    points = []
    for i in range(num_points):
        base = i * point_step
        x = struct.unpack_from('<f', data_bytes, base + x_offset)[0]
        y = struct.unpack_from('<f', data_bytes, base + y_offset)[0]
        z = struct.unpack_from('<f', data_bytes, base + z_offset)[0]
        if not (math.isnan(x) or math.isnan(y) or math.isnan(z)):
            points.append([x, y, z])
            
    pts = np.array(points, dtype=np.float64)
    if pts.size > 0:
        if min_z is not None: pts = pts[pts[:, 2] >= min_z]
        if max_z is not None: pts = pts[pts[:, 2] <= max_z]
    return pts


def downsample_points_spatial(points: np.ndarray, cell_size: float) -> np.ndarray:
    if points.size == 0 or cell_size <= 0:
        return points
    grid_coords = np.round(points[:, :2] / cell_size).astype(np.int32)
    _, unique_indices = np.unique(grid_coords, axis=0, return_index=True)
    return points[unique_indices]


def draw_landmarks(canvas: np.ndarray, pixels: np.ndarray, radius_px: int, color: tuple[int, int, int]) -> None:
    if pixels.size == 0:
        return
    if radius_px <= 0:
        canvas[pixels[:, 1], pixels[:, 0]] = color
        return
    for px, py in pixels:
        cv2.circle(canvas, (int(px), int(py)), radius_px, color, -1, cv2.LINE_AA)


def draw_path(canvas: np.ndarray, pixels: np.ndarray, thickness: int, color: tuple[int, int, int]) -> None:
    if pixels.shape[0] < 2:
        return
    cv2.polylines(
        canvas,
        [pixels.reshape((-1, 1, 2)).astype(np.int32)],
        isClosed=False,
        color=color,
        thickness=max(1, thickness),
        lineType=cv2.LINE_AA,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline export of VSLAM landmarks/paths into a PNG aligned to the map.")
    parser.add_argument("--snapshot", required=True, help="Path to ver1_vslam_reference.json")
    parser.add_argument("--alignment", required=True, help="Path to vslam_map_alignment.yaml")
    parser.add_argument("--reference-yaml", required=True, help="Path to the 2D map yaml (e.g. ver1.yaml)")
    parser.add_argument("--output-image", required=True, help="Path to output PNG image")
    parser.add_argument("--output-yaml", default="", help="Path to output YAML image")
    parser.add_argument("--background-value", type=int, default=255)
    parser.add_argument("--landmark-value", type=int, default=0)
    parser.add_argument("--path-value", type=int, default=96)
    parser.add_argument("--point-radius-px", type=int, default=1)
    parser.add_argument("--path-thickness-px", type=int, default=1)
    parser.add_argument("--min-z", type=float, default=None)
    parser.add_argument("--max-z", type=float, default=None)
    parser.add_argument("--landmark-downsample-m", type=float, default=0.1, help="Spatial downsampling cell size in meters (0 to disable)")
    
    args = parser.parse_args()
    
    snapshot_path = Path(args.snapshot).expanduser().resolve()
    if not snapshot_path.exists():
        print(f"Error: Snapshot not found: {snapshot_path}")
        return
        
    alignment_path = Path(args.alignment).expanduser().resolve()
    if not alignment_path.exists():
        print(f"Error: Alignment YAML not found: {alignment_path}")
        return
        
    reference_path = Path(args.reference_yaml).expanduser().resolve()
    if not reference_path.exists():
        print(f"Error: Reference map YAML not found: {reference_path}")
        return

    # Parse Alignment TF (map -> vslam_map)
    alignment_data = parse_simple_yaml(alignment_path)
    
    # Check if there is ros__parameters wrapper
    if "ros__parameters" in alignment_path.read_text():
        # Quick manual parsing for nested ros__parameters
        align_params = {}
        for line in alignment_path.read_text().splitlines():
            line = line.strip()
            if ":" in line and not line.endswith(":"):
                k, v = line.split(":", 1)
                align_params[k.strip()] = v.strip()
    else:
        align_params = alignment_data

    tx = float(align_params.get("x", 0.0))
    ty = float(align_params.get("y", 0.0))
    tz = float(align_params.get("z", 0.0))
    roll = float(align_params.get("roll_rad", 0.0))
    pitch = float(align_params.get("pitch_rad", 0.0))
    yaw = float(align_params.get("yaw_rad", 0.0))
    
    rot, trans = get_transformation_matrix(tx, ty, tz, roll, pitch, yaw)

    # Load Snapshot
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    
    # Process Landmarks
    landmarks_data = snapshot.get("landmarks")
    if landmarks_data:
        landmark_points = read_landmark_points(landmarks_data, args.min_z, args.max_z)
        if args.landmark_downsample_m > 0:
            original_count = landmark_points.shape[0]
            landmark_points = downsample_points_spatial(landmark_points, args.landmark_downsample_m)
            print(f"Downsampled landmarks: {original_count} -> {landmark_points.shape[0]} (cell: {args.landmark_downsample_m}m)")
    else:
        landmark_points = np.empty((0, 3), dtype=np.float64)

    # Process Path
    full_path_data = snapshot.get("full_vslam_path")
    if full_path_data:
        poses = full_path_data.get("poses", [])
        path_points = np.array([
            [float(p["position"]["x"]), float(p["position"]["y"]), float(p["position"]["z"])]
            for p in poses
        ], dtype=np.float64)
    else:
        path_points = np.empty((0, 3), dtype=np.float64)

    # Transform to map frame
    landmark_points_map = transform_points(landmark_points, rot, trans)
    path_points_map = transform_points(path_points, rot, trans)

    # Read geometry from Reference Map
    ref_data = parse_simple_yaml(reference_path)
    ref_image_value = str(ref_data["image"]).strip().strip('"').strip("'")
    ref_image_path = (reference_path.parent / ref_image_value).resolve()
    image = cv2.imread(str(ref_image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Error: Could not read reference map image: {ref_image_path}")
        return
        
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

    # Create Raster (BGR)
    canvas = np.full((geometry.height, geometry.width, 3), 255, dtype=np.uint8)
    
    # Overlay the 2D map faintly in the background
    if image is not None:
        # image is grayscale, convert to BGR
        map_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Fade the 2D map by 60% (mix with white background)
        canvas = cv2.addWeighted(map_bgr, 0.4, canvas, 0.6, 0)
    
    landmark_xy = landmark_points_map[:, :2] if landmark_points_map.size > 0 else np.empty((0, 2))
    path_xy = path_points_map[:, :2] if path_points_map.size > 0 else np.empty((0, 2))
    
    landmark_pixels = filter_pixels_inside(points_to_pixels(landmark_xy, geometry), geometry.width, geometry.height)
    path_pixels = filter_pixels_inside(points_to_pixels(path_xy, geometry), geometry.width, geometry.height)

    # Draw Landmarks in RED (BGR: 0, 0, 255)
    draw_landmarks(canvas, landmark_pixels, args.point_radius_px, (0, 0, 255))
    # Draw Path in BLUE (BGR: 255, 0, 0)
    draw_path(canvas, path_pixels, args.path_thickness_px, (255, 0, 0))

    output_image = Path(args.output_image).expanduser().resolve()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), canvas)
    
    if args.output_yaml:
        output_yaml = Path(args.output_yaml).expanduser().resolve()
        write_map_yaml(output_yaml, output_image, geometry, template)
        print(f"Wrote map yaml: {output_yaml}")

    print(f"Successfully exported offline aligned landmarks: {output_image}")
    print(f"Landmarks: {landmark_pixels.shape[0]}, Path points: {path_pixels.shape[0]}")

if __name__ == "__main__":
    main()

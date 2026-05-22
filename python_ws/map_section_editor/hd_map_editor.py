#!/usr/bin/env python3
"""
Interactive local HD map editor for VSLAM landmark raster backgrounds.

The editor stores lane boundaries and centerlines in world coordinates so the
resulting YAML can be edited by hand. A selected primary lane is also exported
to the F1TENTH-style centerline CSV consumed by data_analysis/generate_raceline.py.
"""

from __future__ import annotations

import argparse
import json
import math
from ast import literal_eval
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


PointPx = Tuple[int, int]
PointM = Tuple[float, float]

POLYLINE_FIELDS = ("centerline", "left_bound", "right_bound")
FIELD_LABELS = {
    "centerline": "centerline",
    "left_bound": "left bound",
    "right_bound": "right bound",
}
FIELD_COLORS = {
    "centerline": (40, 220, 255),
    "left_bound": (80, 220, 80),
    "right_bound": (230, 100, 230),
}


@dataclass
class RasterGeometry:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    map_yaml_path: Path
    image_path: Path

    def pixel_to_world(self, point: PointPx) -> PointM:
        grid_x = float(point[0]) * self.resolution
        grid_y = float((self.height - 1) - point[1]) * self.resolution
        cos_t = math.cos(self.origin_yaw)
        sin_t = math.sin(self.origin_yaw)
        return (
            self.origin_x + cos_t * grid_x - sin_t * grid_y,
            self.origin_y + sin_t * grid_x + cos_t * grid_y,
        )

    def world_to_pixel(self, point: Sequence[float]) -> PointPx:
        dx = float(point[0]) - self.origin_x
        dy = float(point[1]) - self.origin_y
        cos_t = math.cos(self.origin_yaw)
        sin_t = math.sin(self.origin_yaw)
        grid_x = (cos_t * dx + sin_t * dy) / self.resolution
        grid_y = (-sin_t * dx + cos_t * dy) / self.resolution
        u = int(round(grid_x))
        v = int(round((self.height - 1) - grid_y))
        return (
            max(0, min(self.width - 1, u)),
            max(0, min(self.height - 1, v)),
        )


@dataclass
class LaneDraft:
    lane_id: str
    closed_loop: bool = True
    centerline: List[PointPx] = field(default_factory=list)
    left_bound: List[PointPx] = field(default_factory=list)
    right_bound: List[PointPx] = field(default_factory=list)

    def points(self, field_name: str) -> List[PointPx]:
        return getattr(self, field_name)


def _strip_yaml_scalar(value: object) -> object:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in ("'", '"'):
        return stripped[1:-1]
    return stripped


def _parse_flat_yaml(path: Path) -> Dict[str, object]:
    data: Dict[str, object] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _strip_yaml_scalar(value)
    return data


def load_yaml(path: Path, *, allow_flat_fallback: bool) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    try:
        from omegaconf import OmegaConf

        data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        if isinstance(data, dict):
            return data
    except Exception:
        if not allow_flat_fallback:
            raise RuntimeError(
                f"Could not parse YAML {path}. Install PyYAML or omegaconf to load nested HD map YAML."
            )

    if allow_flat_fallback:
        return _parse_flat_yaml(path)
    raise RuntimeError(f"YAML root must be a mapping: {path}")


def parse_origin(value: object) -> Tuple[float, float, float]:
    if isinstance(value, str):
        value = literal_eval(value)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 2:
        raise RuntimeError("map YAML origin must be [x, y, yaw]")
    yaw = float(value[2]) if len(value) >= 3 else 0.0
    return float(value[0]), float(value[1]), yaw


def resolve_image_path(map_yaml_path: Path, image_value: object) -> Path:
    raw = str(_strip_yaml_scalar(image_value) or "").strip()
    if not raw:
        raise RuntimeError(f"Map YAML has no image entry: {map_yaml_path}")
    image_path = Path(raw).expanduser()
    return image_path.resolve() if image_path.is_absolute() else (map_yaml_path.parent / image_path).resolve()


def load_raster_geometry(map_yaml_path: Path) -> Tuple[RasterGeometry, np.ndarray]:
    map_yaml_path = map_yaml_path.expanduser().resolve()
    data = load_yaml(map_yaml_path, allow_flat_fallback=True)
    if "resolution" not in data or "origin" not in data:
        raise RuntimeError(f"Map YAML needs resolution and origin: {map_yaml_path}")
    image_path = resolve_image_path(map_yaml_path, data.get("image"))
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read map image: {image_path}")
    origin_x, origin_y, origin_yaw = parse_origin(data["origin"])
    resolution = float(data["resolution"])
    if resolution <= 0.0:
        raise RuntimeError(f"Map YAML resolution must be positive: {map_yaml_path}")
    height, width = image.shape[:2]
    return (
        RasterGeometry(
            width=width,
            height=height,
            resolution=resolution,
            origin_x=origin_x,
            origin_y=origin_y,
            origin_yaw=origin_yaw,
            map_yaml_path=map_yaml_path,
            image_path=image_path,
        ),
        image,
    )


def sanitize_lane_id(value: str, fallback: str) -> str:
    allowed = []
    for char in value.strip():
        if char.isalnum() or char in ("_", "-", "."):
            allowed.append(char)
    result = "".join(allowed)
    return result if result else fallback


def next_lane_id(lanes: Sequence[LaneDraft]) -> str:
    existing = {lane.lane_id for lane in lanes}
    index = 1
    while True:
        candidate = f"lane_{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _point_rows_to_pixels(rows: object, geometry: RasterGeometry) -> List[PointPx]:
    points: List[PointPx] = []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return points
    for row in rows:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 2:
            continue
        points.append(geometry.world_to_pixel((float(row[0]), float(row[1]))))
    return points


def load_hd_map_lanes(path: Path, geometry: RasterGeometry) -> Tuple[List[LaneDraft], str]:
    data = load_yaml(path, allow_flat_fallback=False)
    raw_lanes = data.get("lanes", [])
    if not isinstance(raw_lanes, Sequence) or isinstance(raw_lanes, (str, bytes)):
        raise RuntimeError(f"HD map lanes must be a list: {path}")

    lanes: List[LaneDraft] = []
    for index, raw_lane in enumerate(raw_lanes, start=1):
        if not isinstance(raw_lane, dict):
            continue
        lane_id = sanitize_lane_id(str(raw_lane.get("id", "")), f"lane_{index:03d}")
        lanes.append(
            LaneDraft(
                lane_id=lane_id,
                closed_loop=bool(raw_lane.get("closed_loop", True)),
                centerline=_point_rows_to_pixels(raw_lane.get("centerline", []), geometry),
                left_bound=_point_rows_to_pixels(raw_lane.get("left_bound", []), geometry),
                right_bound=_point_rows_to_pixels(raw_lane.get("right_bound", []), geometry),
            )
        )

    if not lanes:
        lanes = [LaneDraft(lane_id="lane_001")]
    primary_lane_id = sanitize_lane_id(str(data.get("primary_lane_id", "")), lanes[0].lane_id)
    if primary_lane_id not in {lane.lane_id for lane in lanes}:
        primary_lane_id = lanes[0].lane_id
    return lanes, primary_lane_id


def _fmt_float(value: float) -> str:
    normalized = 0.0 if abs(value) < 5.0e-13 else float(value)
    return f"{normalized:.9g}"


def _quote_yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _append_world_polyline(
    lines: List[str],
    field_name: str,
    pixel_points: Sequence[PointPx],
    geometry: RasterGeometry,
) -> None:
    lines.append(f"    {field_name}:")
    if not pixel_points:
        lines[-1] = f"    {field_name}: []"
        return
    for point in pixel_points:
        x, y = geometry.pixel_to_world(point)
        lines.append(f"      - [{_fmt_float(x)}, {_fmt_float(y)}, 0.0]")


def write_hd_map_yaml(
    output_path: Path,
    geometry: RasterGeometry,
    lanes: Sequence[LaneDraft],
    primary_lane_id: str,
    centerline_csv_path: Optional[Path],
) -> None:
    lines = [
        "format: tamiya_local_hd_map_v1",
        "frame_id: map",
        "units: meter",
        f"primary_lane_id: {_quote_yaml_string(primary_lane_id)}",
        "source_raster:",
        f"  map_yaml: {_quote_yaml_string(str(geometry.map_yaml_path))}",
        f"  image: {_quote_yaml_string(str(geometry.image_path))}",
        f"  resolution_m_per_px: {_fmt_float(geometry.resolution)}",
        (
            "  origin_xy_yaw: "
            f"[{_fmt_float(geometry.origin_x)}, {_fmt_float(geometry.origin_y)}, {_fmt_float(geometry.origin_yaw)}]"
        ),
        f"  image_size_px: [{geometry.width}, {geometry.height}]",
    ]
    if centerline_csv_path is not None:
        lines.extend(
            [
                "exports:",
                f"  primary_centerline_csv: {_quote_yaml_string(str(centerline_csv_path))}",
            ]
        )
    lines.append("lanes:")

    for lane in lanes:
        lines.extend(
            [
                f"  - id: {_quote_yaml_string(lane.lane_id)}",
                f"    closed_loop: {'true' if lane.closed_loop else 'false'}",
            ]
        )
        _append_world_polyline(lines, "left_bound", lane.left_bound, geometry)
        _append_world_polyline(lines, "right_bound", lane.right_bound, geometry)
        _append_world_polyline(lines, "centerline", lane.centerline, geometry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _world_xy(points: Sequence[PointPx], geometry: RasterGeometry) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray([geometry.pixel_to_world(point) for point in points], dtype=np.float64)


def _nearest_distances(points: np.ndarray, polyline: np.ndarray, closed_loop: bool) -> np.ndarray:
    if len(polyline) == 0:
        return np.zeros(len(points), dtype=np.float64)
    if len(polyline) == 1:
        deltas = points - polyline[0]
        return np.sqrt(np.sum(deltas * deltas, axis=1))

    starts = polyline if closed_loop else polyline[:-1]
    ends = np.roll(polyline, -1, axis=0) if closed_loop else polyline[1:]
    segments = ends - starts
    segment_len_sq = np.sum(segments * segments, axis=1)
    segment_len_sq[segment_len_sq < 1.0e-12] = 1.0

    rel = points[:, None, :] - starts[None, :, :]
    t = np.sum(rel * segments[None, :, :], axis=2) / segment_len_sq[None, :]
    t = np.clip(t, 0.0, 1.0)
    closest = starts[None, :, :] + t[:, :, None] * segments[None, :, :]
    deltas = points[:, None, :] - closest
    return np.sqrt(np.min(np.sum(deltas * deltas, axis=2), axis=1))


def lane_export_issue(lane: LaneDraft) -> Optional[str]:
    if len(lane.centerline) < 2:
        return "centerline needs at least two points"
    if len(lane.left_bound) < 2:
        return "left bound needs at least two points"
    if len(lane.right_bound) < 2:
        return "right bound needs at least two points"
    return None


def export_centerline_csv(output_path: Path, lane: LaneDraft, geometry: RasterGeometry) -> None:
    issue = lane_export_issue(lane)
    if issue is not None:
        raise RuntimeError(f"Lane {lane.lane_id} cannot export centerline CSV: {issue}.")

    centerline = _world_xy(lane.centerline, geometry)
    left_bound = _world_xy(lane.left_bound, geometry)
    right_bound = _world_xy(lane.right_bound, geometry)
    rows = np.column_stack(
        [
            centerline[:, 0],
            centerline[:, 1],
            _nearest_distances(centerline, right_bound, lane.closed_loop),
            _nearest_distances(centerline, left_bound, lane.closed_loop),
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        np.savetxt(
            file,
            rows,
            fmt="%.6f",
            delimiter=",",
            header="x_m,y_m,w_tr_right_m,w_tr_left_m",
        )


class HdMapEditor:
    def __init__(
        self,
        background: np.ndarray,
        geometry: RasterGeometry,
        output_path: Path,
        centerline_output_path: Path,
        lanes: Sequence[LaneDraft],
        primary_lane_id: str,
        window_width: int,
        window_height: int,
        scale: float,
    ) -> None:
        self.background = background.copy()
        self.geometry = geometry
        self.output_path = output_path
        self.centerline_output_path = centerline_output_path
        self.lanes = list(lanes) if lanes else [LaneDraft(lane_id="lane_001")]
        self.primary_lane_id = primary_lane_id
        self.active_lane_index = 0
        self.active_field = "centerline"

        self.window_name = "Local HD Map Editor"
        self.window_width = max(480, int(window_width))
        self.window_height = max(320, int(window_height))
        self.min_scale = 0.1
        self.max_scale = 24.0
        self.scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_offset = (0, 0)
        self.dragging_index: Optional[int] = None
        self.has_unsaved_changes = False
        self.show_help = True

        if scale > 0.0:
            self.scale = max(self.min_scale, min(self.max_scale, float(scale)))
        else:
            self._reset_view()

    @property
    def active_lane(self) -> LaneDraft:
        return self.lanes[self.active_lane_index]

    @property
    def active_points(self) -> List[PointPx]:
        return self.active_lane.points(self.active_field)

    def _scaled_size(self) -> Tuple[int, int]:
        return (
            max(1, int(round(self.geometry.width * self.scale))),
            max(1, int(round(self.geometry.height * self.scale))),
        )

    def _clamp_pan(self) -> None:
        scaled_width, scaled_height = self._scaled_size()
        self.pan_x = max(0, min(self.pan_x, max(0, scaled_width - self.window_width)))
        self.pan_y = max(0, min(self.pan_y, max(0, scaled_height - self.window_height)))

    def _reset_view(self) -> None:
        self.scale = max(
            self.min_scale,
            min(
                self.max_scale,
                min(
                    self.window_width / float(max(1, self.geometry.width)),
                    self.window_height / float(max(1, self.geometry.height)),
                ),
            ),
        )
        self.pan_x = 0
        self.pan_y = 0

    def _zoom_at(self, factor: float, x: int, y: int) -> None:
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, old_scale * factor))
        if abs(new_scale - old_scale) < 1.0e-9:
            return
        u = (self.pan_x + x) / old_scale
        v = (self.pan_y + y) / old_scale
        self.scale = new_scale
        self.pan_x = int(round(u * new_scale - x))
        self.pan_y = int(round(v * new_scale - y))
        self._clamp_pan()

    def _to_map_pixel(self, x: int, y: int) -> PointPx:
        u = int(round((self.pan_x + x) / self.scale))
        v = int(round((self.pan_y + y) / self.scale))
        return (
            max(0, min(self.geometry.width - 1, u)),
            max(0, min(self.geometry.height - 1, v)),
        )

    def _inside_map(self, x: int, y: int) -> bool:
        scaled_width, scaled_height = self._scaled_size()
        sx = self.pan_x + x
        sy = self.pan_y + y
        return 0 <= sx < scaled_width and 0 <= sy < scaled_height

    def _nearest_active_point_index(self, point: PointPx) -> Optional[int]:
        if not self.active_points:
            return None
        threshold = max(2.0, 11.0 / max(self.scale, 1.0e-6))
        best_index: Optional[int] = None
        best_distance = float("inf")
        for index, candidate in enumerate(self.active_points):
            distance = math.hypot(candidate[0] - point[0], candidate[1] - point[1])
            if distance <= threshold and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _draw_text(self, frame: np.ndarray, text: str, origin: PointPx, scale: float, color: Tuple[int, int, int]) -> None:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _draw_panel(self, frame: np.ndarray, height: int) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (min(self.window_width - 8, 1050), min(self.window_height - 8, height)), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.74, frame, 0.26, 0.0, dst=frame)
        cv2.rectangle(frame, (8, 8), (min(self.window_width - 8, 1050), min(self.window_height - 8, height)), (220, 220, 220), 1, cv2.LINE_AA)

    def _draw_polyline(
        self,
        canvas: np.ndarray,
        points: Sequence[PointPx],
        color: Tuple[int, int, int],
        closed: bool,
        point_radius: int,
        thickness: int,
    ) -> None:
        if not points:
            return
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], bool(closed and len(points) >= 3), color, thickness, cv2.LINE_AA)
        if point_radius > 0:
            for point in points:
                cv2.circle(canvas, point, point_radius, color, -1, cv2.LINE_AA)

    def _draw_map(self) -> np.ndarray:
        canvas = self.background.copy()
        shade = np.full_like(canvas, 235)
        canvas = cv2.addWeighted(canvas, 0.82, shade, 0.18, 0.0)

        for lane_index, lane in enumerate(self.lanes):
            active_lane = lane_index == self.active_lane_index
            for field_name in POLYLINE_FIELDS:
                base_color = FIELD_COLORS[field_name]
                if active_lane:
                    color = base_color
                    radius = 4 if field_name == self.active_field else 2
                    thickness = 3 if field_name == self.active_field else 2
                else:
                    color = tuple(int(channel * 0.55) for channel in base_color)
                    radius = 0
                    thickness = 1
                self._draw_polyline(
                    canvas,
                    lane.points(field_name),
                    color,
                    lane.closed_loop,
                    radius,
                    thickness,
                )

        scaled_width, scaled_height = self._scaled_size()
        interpolation = cv2.INTER_NEAREST if self.scale >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(canvas, (scaled_width, scaled_height), interpolation=interpolation)
        self._clamp_pan()
        cropped = scaled[
            self.pan_y : min(scaled_height, self.pan_y + self.window_height),
            self.pan_x : min(scaled_width, self.pan_x + self.window_width),
        ]
        frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        frame[: cropped.shape[0], : cropped.shape[1]] = cropped
        self._draw_hud(frame)
        return frame

    def _draw_hud(self, frame: np.ndarray) -> None:
        if not self.show_help:
            self._draw_panel(frame, 72)
            self._draw_text(
                frame,
                f"{self.active_lane.lane_id}  {FIELD_LABELS[self.active_field]}  i:help  s:save",
                (22, 48),
                0.75,
                (255, 255, 255),
            )
            return

        cursor_px = self._to_map_pixel(self.last_mouse_x, self.last_mouse_y)
        cursor_m = self.geometry.pixel_to_world(cursor_px)
        issue = lane_export_issue(self._lane_by_id(self.primary_lane_id))
        export_state = "ready" if issue is None else issue
        dirty = "unsaved" if self.has_unsaved_changes else "saved"
        loop_state = "closed" if self.active_lane.closed_loop else "open"
        primary_suffix = " primary" if self.active_lane.lane_id == self.primary_lane_id else ""
        self._draw_panel(frame, 220)
        self._draw_text(
            frame,
            (
                f"HD map: {self.active_lane.lane_id}{primary_suffix} "
                f"({self.active_lane_index + 1}/{len(self.lanes)}, {loop_state}, {dirty})"
            ),
            (22, 46),
            0.78,
            (255, 255, 255),
        )
        self._draw_text(
            frame,
            (
                f"Editing {FIELD_LABELS[self.active_field]}: {len(self.active_points)} points   "
                f"zoom {self.scale:.2f}x   map ({cursor_m[0]:.3f}, {cursor_m[1]:.3f}) m"
            ),
            (22, 80),
            0.66,
            FIELD_COLORS[self.active_field],
        )
        self._draw_text(
            frame,
            "L-click:add or drag point  d:delete near/last  u:undo last  1:center  2:left  3:right",
            (22, 114),
            0.62,
            (235, 235, 235),
        )
        self._draw_text(
            frame,
            "n:new lane  [/]:switch lane  p:set primary  o:toggle closed loop  s:save YAML+CSV",
            (22, 146),
            0.62,
            (235, 235, 235),
        )
        self._draw_text(
            frame,
            "Wheel +/-:zoom  Right-drag or H/J/K/L:pan  0:fit  i:help  q/Esc:quit",
            (22, 178),
            0.62,
            (235, 235, 235),
        )
        self._draw_text(
            frame,
            f"Primary CSV export: {export_state}",
            (22, 208),
            0.60,
            (130, 235, 130) if issue is None else (120, 190, 255),
        )

    def _mouse_callback(self, event: int, x: int, y: int, flags: int, _userdata: object) -> None:
        self.last_mouse_x = max(0, min(self.window_width - 1, int(x)))
        self.last_mouse_y = max(0, min(self.window_height - 1, int(y)))

        if event == cv2.EVENT_RBUTTONDOWN:
            self.is_panning = True
            self.pan_start_mouse = (x, y)
            self.pan_start_offset = (self.pan_x, self.pan_y)
            return
        if event == cv2.EVENT_RBUTTONUP:
            self.is_panning = False
            return
        if event == cv2.EVENT_MOUSEWHEEL:
            self._zoom_at(1.15 if flags > 0 else 1.0 / 1.15, x, y)
            return
        if event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            self.pan_x = self.pan_start_offset[0] - (x - self.pan_start_mouse[0])
            self.pan_y = self.pan_start_offset[1] - (y - self.pan_start_mouse[1])
            self._clamp_pan()
            return

        if event == cv2.EVENT_LBUTTONDOWN and self._inside_map(x, y):
            point = self._to_map_pixel(x, y)
            near_index = self._nearest_active_point_index(point)
            if near_index is not None:
                self.dragging_index = near_index
                return
            self.active_points.append(point)
            self.has_unsaved_changes = True
            self.dragging_index = len(self.active_points) - 1
            return

        if event == cv2.EVENT_MOUSEMOVE and self.dragging_index is not None and self._inside_map(x, y):
            self.active_points[self.dragging_index] = self._to_map_pixel(x, y)
            self.has_unsaved_changes = True
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.dragging_index = None

    def _lane_by_id(self, lane_id: str) -> LaneDraft:
        for lane in self.lanes:
            if lane.lane_id == lane_id:
                return lane
        return self.lanes[0]

    def _set_field(self, field_name: str) -> None:
        if field_name in POLYLINE_FIELDS:
            self.active_field = field_name

    def _delete_point(self) -> None:
        if not self.active_points:
            print("[INFO] Active polyline has no points.")
            return
        cursor = self._to_map_pixel(self.last_mouse_x, self.last_mouse_y)
        index = self._nearest_active_point_index(cursor)
        if index is None:
            index = len(self.active_points) - 1
        removed = self.active_points.pop(index)
        self.has_unsaved_changes = True
        print(f"[INFO] Removed {FIELD_LABELS[self.active_field]} point {removed}.")

    def _new_lane(self) -> None:
        lane = LaneDraft(lane_id=next_lane_id(self.lanes))
        self.lanes.append(lane)
        self.active_lane_index = len(self.lanes) - 1
        self.active_field = "centerline"
        self.has_unsaved_changes = True
        print(f"[INFO] Added lane {lane.lane_id}.")

    def _switch_lane(self, delta: int) -> None:
        self.active_lane_index = (self.active_lane_index + delta) % len(self.lanes)
        print(f"[INFO] Active lane: {self.active_lane.lane_id}.")

    def _save(self) -> None:
        write_hd_map_yaml(
            output_path=self.output_path,
            geometry=self.geometry,
            lanes=self.lanes,
            primary_lane_id=self.primary_lane_id,
            centerline_csv_path=self.centerline_output_path,
        )
        print(f"[INFO] Saved HD map YAML: {self.output_path}")
        primary_lane = self._lane_by_id(self.primary_lane_id)
        issue = lane_export_issue(primary_lane)
        if issue is None:
            export_centerline_csv(self.centerline_output_path, primary_lane, self.geometry)
            print(f"[INFO] Exported primary centerline CSV: {self.centerline_output_path}")
        else:
            print(f"[WARN] Saved YAML without primary centerline CSV: {issue}.")
        self.has_unsaved_changes = False

    def _pan_by_key(self, dx: int, dy: int) -> None:
        self.pan_x += dx
        self.pan_y += dy
        self._clamp_pan()

    def _handle_key(self, key: int) -> bool:
        low = key & 0xFF
        if low in (27, ord("q")):
            if self.has_unsaved_changes:
                print("[WARN] Quit with unsaved HD map edits.")
            return False
        if low == ord("1"):
            self._set_field("centerline")
        elif low == ord("2"):
            self._set_field("left_bound")
        elif low == ord("3"):
            self._set_field("right_bound")
        elif low == ord("n"):
            self._new_lane()
        elif low in (ord("["), ord(",")):
            self._switch_lane(-1)
        elif low in (ord("]"), ord(".")):
            self._switch_lane(1)
        elif low == ord("p"):
            self.primary_lane_id = self.active_lane.lane_id
            self.has_unsaved_changes = True
            print(f"[INFO] Primary lane: {self.primary_lane_id}.")
        elif low == ord("o"):
            self.active_lane.closed_loop = not self.active_lane.closed_loop
            self.has_unsaved_changes = True
        elif low == ord("s"):
            self._save()
        elif low in (ord("d"), 8, 127):
            self._delete_point()
        elif low == ord("u"):
            if self.active_points:
                self.active_points.pop()
                self.has_unsaved_changes = True
            else:
                print("[INFO] Active polyline has no points.")
        elif low == ord("i"):
            self.show_help = not self.show_help
        elif low == ord("0"):
            self._reset_view()
        elif low in (ord("+"), ord("=")):
            self._zoom_at(1.15, self.window_width // 2, self.window_height // 2)
        elif low in (ord("-"), ord("_")):
            self._zoom_at(1.0 / 1.15, self.window_width // 2, self.window_height // 2)
        elif low in (ord("h"), 81):
            self._pan_by_key(-60, 0)
        elif low in (ord("l"), 83):
            self._pan_by_key(60, 0)
        elif low in (ord("k"), 82):
            self._pan_by_key(0, -60)
        elif low in (ord("j"), 84):
            self._pan_by_key(0, 60)
        return True

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        print("[INFO] HD map editor started. Press i for help and s to save.")
        keep_running = True
        while keep_running:
            cv2.imshow(self.window_name, self._draw_map())
            key = cv2.waitKeyEx(20)
            if key >= 0:
                keep_running = self._handle_key(key)
        cv2.destroyWindow(self.window_name)


def _default_centerline_output(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_centerline.csv")


def load_or_create_lanes(output_path: Path, geometry: RasterGeometry) -> Tuple[List[LaneDraft], str]:
    if output_path.exists():
        lanes, primary_lane_id = load_hd_map_lanes(output_path, geometry)
        print(f"[INFO] Loaded HD map YAML: {output_path}")
        return lanes, primary_lane_id
    return [LaneDraft(lane_id="lane_001")], "lane_001"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw local HD map lane bounds and centerlines on a map YAML raster."
    )
    parser.add_argument("--map-yaml", required=True, help="Landmark raster or occupancy map YAML used as the editor background.")
    parser.add_argument("--output", required=True, help="Editable HD map YAML output path.")
    parser.add_argument(
        "--centerline-output",
        default="",
        help="Primary lane centerline CSV. Default: <output_stem>_centerline.csv.",
    )
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=0.0, help="Initial image scale. 0 fits the window.")
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="Reload the HD map YAML and export the primary centerline CSV without opening the GUI.",
    )
    return parser


def main() -> int:
    if _IMPORT_ERROR is not None:
        raise SystemExit(f"hd_map_editor.py requires numpy and opencv-python: {_IMPORT_ERROR}")

    args = build_arg_parser().parse_args()
    output_path = Path(args.output).expanduser().resolve()
    centerline_output_path = (
        Path(args.centerline_output).expanduser().resolve()
        if args.centerline_output
        else _default_centerline_output(output_path)
    )
    geometry, background = load_raster_geometry(Path(args.map_yaml))
    lanes, primary_lane_id = load_or_create_lanes(output_path, geometry)

    if args.export_only:
        primary_lane = next((lane for lane in lanes if lane.lane_id == primary_lane_id), lanes[0])
        export_centerline_csv(centerline_output_path, primary_lane, geometry)
        print(f"[INFO] Exported primary centerline CSV: {centerline_output_path}")
        return 0

    editor = HdMapEditor(
        background=background,
        geometry=geometry,
        output_path=output_path,
        centerline_output_path=centerline_output_path,
        lanes=lanes,
        primary_lane_id=primary_lane_id,
        window_width=args.window_width,
        window_height=args.window_height,
        scale=args.scale,
    )
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

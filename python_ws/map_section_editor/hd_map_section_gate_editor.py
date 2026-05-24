#!/usr/bin/env python3
"""Interactive section-gate editor for editable local HD map YAML files."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import cv2
    import numpy as np
    import yaml

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    yaml = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


PointPx = Tuple[int, int]
PointM = Tuple[float, float]


@dataclass
class RasterGeometry:
    width: int
    height: int
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
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
class LaneView:
    lane_id: str
    closed_loop: bool
    centerline: List[PointPx]
    left_bound: List[PointPx]
    right_bound: List[PointPx]


@dataclass
class SectionGate:
    gate_id: str
    lane_id: str
    points: Tuple[PointPx, PointPx]
    s_m: float


def _as_float_triplet(row: object) -> Optional[Tuple[float, float, float]]:
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 2:
        return None
    z = float(row[2]) if len(row) >= 3 else 0.0
    return float(row[0]), float(row[1]), z


def _world_rows_to_pixels(rows: object, geometry: RasterGeometry) -> List[PointPx]:
    points: List[PointPx] = []
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return points
    for row in rows:
        parsed = _as_float_triplet(row)
        if parsed is not None:
            points.append(geometry.world_to_pixel(parsed))
    return points


def _fmt_float(value: float) -> float:
    normalized = 0.0 if abs(value) < 5.0e-13 else float(value)
    return float(f"{normalized:.9g}")


def _distance(a: PointPx, b: PointPx) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _sanitize_id(value: str, fallback: str) -> str:
    allowed = [c for c in value.strip() if c.isalnum() or c in ("_", "-", ".")]
    result = "".join(allowed)
    return result if result else fallback


def _next_gate_id(gates: Sequence[SectionGate]) -> str:
    existing = {gate.gate_id for gate in gates}
    index = 1
    while True:
        candidate = f"gate_{index:03d}"
        if candidate not in existing:
            return candidate
        index += 1


def _source_raster_geometry(data: Dict[str, Any], hd_map_path: Path) -> tuple[RasterGeometry, np.ndarray]:
    source = data.get("source_raster")
    if not isinstance(source, dict):
        raise RuntimeError("HD map YAML has no source_raster block. Recreate it with hd_map_editor.py.")

    image_path = Path(str(source.get("image", ""))).expanduser()
    if not image_path.is_absolute():
        image_path = (hd_map_path.parent / image_path).resolve()
    background = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if background is None:
        raise FileNotFoundError(f"Could not read source raster image: {image_path}")

    origin = source.get("origin_xy_yaw", [0.0, 0.0, 0.0])
    if not isinstance(origin, Sequence) or isinstance(origin, (str, bytes)) or len(origin) < 2:
        raise RuntimeError("source_raster.origin_xy_yaw must be [x, y, yaw]")

    height, width = background.shape[:2]
    image_size = source.get("image_size_px")
    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)) and len(image_size) >= 2:
        width = int(image_size[0])
        height = int(image_size[1])

    geometry = RasterGeometry(
        width=width,
        height=height,
        resolution=float(source.get("resolution_m_per_px", 0.02)),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]) if len(origin) >= 3 else 0.0,
        image_path=image_path,
    )
    return geometry, background


def _load_lanes(data: Dict[str, Any], geometry: RasterGeometry) -> List[LaneView]:
    lanes: List[LaneView] = []
    raw_lanes = data.get("lanes", [])
    if not isinstance(raw_lanes, list):
        return lanes
    for index, raw_lane in enumerate(raw_lanes, start=1):
        if not isinstance(raw_lane, dict):
            continue
        lane_id = _sanitize_id(str(raw_lane.get("id", "")), f"lane_{index:03d}")
        lanes.append(
            LaneView(
                lane_id=lane_id,
                closed_loop=bool(raw_lane.get("closed_loop", True)),
                centerline=_world_rows_to_pixels(raw_lane.get("centerline", []), geometry),
                left_bound=_world_rows_to_pixels(raw_lane.get("left_bound", []), geometry),
                right_bound=_world_rows_to_pixels(raw_lane.get("right_bound", []), geometry),
            )
        )
    return lanes


def _polyline_s(points: Sequence[PointPx], geometry: RasterGeometry, closed_loop: bool) -> tuple[List[float], float]:
    if not points:
        return [], 0.0
    s_values = [0.0]
    total = 0.0
    world = [geometry.pixel_to_world(point) for point in points]
    for i in range(1, len(world)):
        total += math.hypot(world[i][0] - world[i - 1][0], world[i][1] - world[i - 1][1])
        s_values.append(total)
    if closed_loop and len(world) >= 3:
        total += math.hypot(world[0][0] - world[-1][0], world[0][1] - world[-1][1])
    return s_values, total


def _project_point_to_lane_s(point: PointPx, lane: LaneView, geometry: RasterGeometry) -> float:
    if not lane.centerline:
        return 0.0
    if len(lane.centerline) == 1:
        return 0.0

    target = geometry.pixel_to_world(point)
    world = [geometry.pixel_to_world(p) for p in lane.centerline]
    s_values, _total = _polyline_s(lane.centerline, geometry, lane.closed_loop)
    best_distance = float("inf")
    best_s = 0.0
    segment_count = len(world) if lane.closed_loop and len(world) >= 3 else len(world) - 1
    for i in range(segment_count):
        j = (i + 1) % len(world)
        ax, ay = world[i]
        bx, by = world[j]
        vx = bx - ax
        vy = by - ay
        denom = vx * vx + vy * vy
        if denom <= 1.0e-12:
            continue
        t = max(0.0, min(1.0, ((target[0] - ax) * vx + (target[1] - ay) * vy) / denom))
        cx = ax + t * vx
        cy = ay + t * vy
        distance = math.hypot(target[0] - cx, target[1] - cy)
        if distance < best_distance:
            best_distance = distance
            best_s = s_values[i] + math.sqrt(denom) * t
    return best_s


def _gate_midpoint(gate: SectionGate) -> PointPx:
    return (
        int(round((gate.points[0][0] + gate.points[1][0]) * 0.5)),
        int(round((gate.points[0][1] + gate.points[1][1]) * 0.5)),
    )


def _load_gates(data: Dict[str, Any], lanes: Sequence[LaneView], geometry: RasterGeometry) -> List[SectionGate]:
    lane_by_id = {lane.lane_id: lane for lane in lanes}
    gates: List[SectionGate] = []
    raw_gates = data.get("section_gates", [])
    if not isinstance(raw_gates, list):
        return gates
    for index, raw_gate in enumerate(raw_gates, start=1):
        if not isinstance(raw_gate, dict):
            continue
        lane_id = _sanitize_id(str(raw_gate.get("lane_id", "")), lanes[0].lane_id if lanes else "lane_001")
        raw_line = raw_gate.get("line", [])
        if not isinstance(raw_line, Sequence) or isinstance(raw_line, (str, bytes)) or len(raw_line) < 2:
            continue
        p0 = _as_float_triplet(raw_line[0])
        p1 = _as_float_triplet(raw_line[1])
        if p0 is None or p1 is None:
            continue
        points = (geometry.world_to_pixel(p0), geometry.world_to_pixel(p1))
        lane = lane_by_id.get(lane_id)
        default_s = _project_point_to_lane_s(_gate_midpoint(SectionGate("", lane_id, points, 0.0)), lane, geometry) if lane else 0.0
        gates.append(
            SectionGate(
                gate_id=_sanitize_id(str(raw_gate.get("id", "")), f"gate_{index:03d}"),
                lane_id=lane_id,
                points=points,
                s_m=float(raw_gate.get("s_m", default_s)),
            )
        )
    return gates


def _gate_to_yaml(gate: SectionGate, geometry: RasterGeometry) -> Dict[str, Any]:
    p0 = geometry.pixel_to_world(gate.points[0])
    p1 = geometry.pixel_to_world(gate.points[1])
    return {
        "id": gate.gate_id,
        "lane_id": gate.lane_id,
        "s_m": _fmt_float(gate.s_m),
        "line": [
            [_fmt_float(p0[0]), _fmt_float(p0[1]), 0.0],
            [_fmt_float(p1[0]), _fmt_float(p1[1]), 0.0],
        ],
    }


def _build_sections(data: Dict[str, Any], gates: Sequence[SectionGate], lanes: Sequence[LaneView], geometry: RasterGeometry) -> List[Dict[str, Any]]:
    previous_by_key: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    raw_sections = data.get("sections", [])
    if not isinstance(raw_sections, list):
        raw_sections = []
    for section in raw_sections:
        if not isinstance(section, dict):
            continue
        key = (
            str(section.get("lane_id", "")),
            str(section.get("start_gate_id", "")),
            str(section.get("end_gate_id", "")),
        )
        previous_by_key[key] = section

    lane_by_id = {lane.lane_id: lane for lane in lanes}
    gates_by_lane: Dict[str, List[SectionGate]] = {}
    for gate in gates:
        gates_by_lane.setdefault(gate.lane_id, []).append(gate)

    sections: List[Dict[str, Any]] = []
    section_index = 1
    for lane_id, lane_gates in sorted(gates_by_lane.items()):
        lane = lane_by_id.get(lane_id)
        if lane is None or len(lane_gates) < 2:
            continue
        sorted_gates = sorted(lane_gates, key=lambda gate: gate.s_m)
        _s_values, lane_length = _polyline_s(lane.centerline, geometry, lane.closed_loop)
        pair_count = len(sorted_gates) if lane.closed_loop else len(sorted_gates) - 1
        for i in range(pair_count):
            start_gate = sorted_gates[i]
            end_gate = sorted_gates[(i + 1) % len(sorted_gates)]
            if not lane.closed_loop and i + 1 >= len(sorted_gates):
                continue
            key = (lane_id, start_gate.gate_id, end_gate.gate_id)
            previous = previous_by_key.get(key, {})
            section: Dict[str, Any] = {
                "id": str(previous.get("id", f"section_{section_index:03d}")),
                "lane_id": lane_id,
                "start_gate_id": start_gate.gate_id,
                "end_gate_id": end_gate.gate_id,
                "start_s_m": _fmt_float(start_gate.s_m),
                "end_s_m": _fmt_float(end_gate.s_m),
            }
            if lane.closed_loop:
                section["wrap"] = bool(start_gate.s_m > end_gate.s_m)
                section["lane_length_m"] = _fmt_float(lane_length)
            for key_name in (
                "speed_override_mps",
                "speed_scale",
                "class",
                "policy",
                "allow_overtake",
                "note",
            ):
                section[key_name] = previous.get(key_name, None)
            sections.append(section)
            section_index += 1
    return sections


class SectionGateEditor:
    def __init__(
        self,
        hd_map_path: Path,
        data: Dict[str, Any],
        geometry: RasterGeometry,
        background: np.ndarray,
        lanes: Sequence[LaneView],
        gates: Sequence[SectionGate],
        window_width: int,
        window_height: int,
        scale: float,
    ) -> None:
        self.hd_map_path = hd_map_path
        self.data = data
        self.geometry = geometry
        self.background = background.copy()
        self.lanes = list(lanes)
        self.gates = list(gates)
        self.active_lane_index = 0
        self.pending_gate_start: Optional[PointPx] = None
        self.window_name = "HD Map Section Gate Editor"
        self.window_width = max(480, int(window_width))
        self.window_height = max(320, int(window_height))
        self.min_scale = 0.1
        self.max_scale = 24.0
        self.scale = max(self.min_scale, min(self.max_scale, float(scale))) if scale > 0.0 else 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_offset = (0, 0)
        self.show_help = True
        self.has_unsaved_changes = False
        self.undo_stack: List[tuple[List[SectionGate], Optional[PointPx], bool]] = []
        if scale > 0.0:
            self._center_view()
        else:
            self._reset_view()

    @property
    def active_lane(self) -> LaneView:
        return self.lanes[self.active_lane_index]

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

    def _center_view(self) -> None:
        scaled_width, scaled_height = self._scaled_size()
        self.pan_x = max(0, int(round((scaled_width - self.window_width) / 2.0)))
        self.pan_y = max(0, int(round((scaled_height - self.window_height) / 2.0)))
        self._clamp_pan()

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

    def _draw_text(self, frame: np.ndarray, text: str, origin: PointPx, scale: float, color: Tuple[int, int, int]) -> None:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

    def _draw_polyline(self, canvas: np.ndarray, points: Sequence[PointPx], color: Tuple[int, int, int], closed: bool, thickness: int) -> None:
        if len(points) < 2:
            return
        pts = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts], bool(closed and len(points) >= 3), color, thickness, cv2.LINE_AA)

    def _draw_map(self) -> np.ndarray:
        canvas = self.background.copy()
        shade = np.full_like(canvas, 235)
        canvas = cv2.addWeighted(canvas, 0.82, shade, 0.18, 0.0)

        for lane_index, lane in enumerate(self.lanes):
            active = lane_index == self.active_lane_index
            scale = 1.0 if active else 0.48
            self._draw_polyline(canvas, lane.left_bound, (int(80 * scale), int(220 * scale), int(80 * scale)), lane.closed_loop, 2 if active else 1)
            self._draw_polyline(canvas, lane.right_bound, (int(230 * scale), int(100 * scale), int(230 * scale)), lane.closed_loop, 2 if active else 1)
            self._draw_polyline(canvas, lane.centerline, (0, int(255 * scale), int(255 * scale)), lane.closed_loop, 3 if active else 1)

        for gate in self.gates:
            active = gate.lane_id == self.active_lane.lane_id
            color = (30, 140, 255) if active else (40, 90, 150)
            cv2.line(canvas, gate.points[0], gate.points[1], color, 3 if active else 1, cv2.LINE_AA)
            midpoint = _gate_midpoint(gate)
            cv2.circle(canvas, midpoint, 5 if active else 3, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, gate.gate_id, (midpoint[0] + 5, midpoint[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, gate.gate_id, (midpoint[0] + 5, midpoint[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        if self.pending_gate_start is not None:
            cursor = self._to_map_pixel(self.last_mouse_x, self.last_mouse_y)
            cv2.line(canvas, self.pending_gate_start, cursor, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(canvas, self.pending_gate_start, 5, (255, 255, 255), -1, cv2.LINE_AA)

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
        overlay = frame.copy()
        panel_h = 178 if self.show_help else 72
        cv2.rectangle(overlay, (8, 8), (min(self.window_width - 8, 1120), min(self.window_height - 8, panel_h)), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.74, frame, 0.26, 0.0, dst=frame)
        lane_gates = [gate for gate in self.gates if gate.lane_id == self.active_lane.lane_id]
        dirty = "unsaved" if self.has_unsaved_changes else "saved"
        pending = " first endpoint set" if self.pending_gate_start is not None else ""
        self._draw_text(
            frame,
            f"Section gates: {self.active_lane.lane_id} ({self.active_lane_index + 1}/{len(self.lanes)}) gates={len(lane_gates)} {dirty}{pending}",
            (22, 46),
            0.72,
            (255, 255, 255),
        )
        if not self.show_help:
            return
        cursor_px = self._to_map_pixel(self.last_mouse_x, self.last_mouse_y)
        cursor_m = self.geometry.pixel_to_world(cursor_px)
        s_m = _project_point_to_lane_s(cursor_px, self.active_lane, self.geometry)
        self._draw_text(
            frame,
            f"L-click two points: add gate across active lane   cursor ({cursor_m[0]:.3f}, {cursor_m[1]:.3f}) s={s_m:.3f} m",
            (22, 80),
            0.60,
            (230, 235, 245),
        )
        self._draw_text(
            frame,
            "[/]:switch lane  d:delete nearest gate  u:undo  Esc:cancel pending gate  s:save  i:help  q:quit",
            (22, 112),
            0.60,
            (230, 235, 245),
        )
        self._draw_text(
            frame,
            "Saved sections are generated between sorted gates. Edit speed_override_mps in YAML when needed.",
            (22, 144),
            0.58,
            (130, 235, 130),
        )

    def _nearest_gate_index(self, point: PointPx) -> Optional[int]:
        threshold = max(3.0, 16.0 / max(self.scale, 1.0e-6))
        best_index = None
        best_distance = float("inf")
        for index, gate in enumerate(self.gates):
            if gate.lane_id != self.active_lane.lane_id:
                continue
            distance = _distance(point, _gate_midpoint(gate))
            if distance <= threshold and distance < best_distance:
                best_index = index
                best_distance = distance
        return best_index

    def _clone_gates(self) -> List[SectionGate]:
        return [
            SectionGate(
                gate_id=gate.gate_id,
                lane_id=gate.lane_id,
                points=(gate.points[0], gate.points[1]),
                s_m=gate.s_m,
            )
            for gate in self.gates
        ]

    def _push_undo_state(self) -> None:
        self.undo_stack.append((self._clone_gates(), self.pending_gate_start, self.has_unsaved_changes))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _push_gate_edit_undo_state(self) -> None:
        self.undo_stack.append((self._clone_gates(), None, self.has_unsaved_changes))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if not self.undo_stack:
            print("[INFO] Nothing to undo.")
            return
        gates, pending_gate_start, has_unsaved_changes = self.undo_stack.pop()
        self.gates = gates
        self.pending_gate_start = pending_gate_start
        self.has_unsaved_changes = has_unsaved_changes
        print("[INFO] Undid last section gate edit.")

    def _add_gate_endpoint(self, point: PointPx) -> None:
        if self.pending_gate_start is None:
            self.pending_gate_start = point
            return
        if _distance(self.pending_gate_start, point) < 2.0:
            print("[WARN] Gate endpoints are too close.")
            self.pending_gate_start = None
            return
        self._push_gate_edit_undo_state()
        midpoint = (
            int(round((self.pending_gate_start[0] + point[0]) * 0.5)),
            int(round((self.pending_gate_start[1] + point[1]) * 0.5)),
        )
        gate = SectionGate(
            gate_id=_next_gate_id(self.gates),
            lane_id=self.active_lane.lane_id,
            points=(self.pending_gate_start, point),
            s_m=_project_point_to_lane_s(midpoint, self.active_lane, self.geometry),
        )
        self.gates.append(gate)
        self.pending_gate_start = None
        self.has_unsaved_changes = True
        print(f"[INFO] Added {gate.gate_id} on {gate.lane_id} at s={gate.s_m:.3f} m.")

    def _delete_gate(self) -> None:
        cursor = self._to_map_pixel(self.last_mouse_x, self.last_mouse_y)
        index = self._nearest_gate_index(cursor)
        had_pending = self.pending_gate_start is not None
        self.pending_gate_start = None
        if index is None:
            if had_pending:
                print("[INFO] Canceled pending gate endpoint.")
            else:
                print("[INFO] No nearby gate on active lane.")
            return
        self._push_undo_state()
        gate = self.gates.pop(index)
        self.has_unsaved_changes = True
        print(f"[INFO] Removed {gate.gate_id}.")

    def _switch_lane(self, delta: int) -> None:
        self.active_lane_index = (self.active_lane_index + delta) % len(self.lanes)
        self.pending_gate_start = None
        print(f"[INFO] Active lane: {self.active_lane.lane_id}.")

    def _save(self) -> None:
        for gate in self.gates:
            lane = next((candidate for candidate in self.lanes if candidate.lane_id == gate.lane_id), None)
            if lane is not None:
                gate.s_m = _project_point_to_lane_s(_gate_midpoint(gate), lane, self.geometry)
        self.data["section_gates"] = [_gate_to_yaml(gate, self.geometry) for gate in sorted(self.gates, key=lambda item: (item.lane_id, item.s_m, item.gate_id))]
        self.data["sections"] = _build_sections(self.data, self.gates, self.lanes, self.geometry)
        self.hd_map_path.write_text(
            yaml.safe_dump(self.data, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        self.has_unsaved_changes = False
        self.undo_stack.clear()
        print(f"[INFO] Saved HD map section gates: {self.hd_map_path}")
        print(f"[INFO] Generated sections: {len(self.data['sections'])}")

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
            self._add_gate_endpoint(self._to_map_pixel(x, y))

    def _handle_key(self, key: int) -> bool:
        low = key & 0xFF
        if low == ord("q"):
            if self.has_unsaved_changes:
                print("[WARN] Quit with unsaved section gate edits.")
            return False
        if low == 27:
            self.pending_gate_start = None
        elif low in (ord("["), ord(",")):
            self._switch_lane(-1)
        elif low in (ord("]"), ord(".")):
            self._switch_lane(1)
        elif low == ord("s"):
            self._save()
        elif low in (ord("d"), 8, 127):
            self._delete_gate()
        elif low == ord("u"):
            self._undo()
        elif low == ord("i"):
            self.show_help = not self.show_help
        elif low == ord("0"):
            self._reset_view()
        elif low in (ord("+"), ord("=")):
            self._zoom_at(1.15, self.window_width // 2, self.window_height // 2)
        elif low in (ord("-"), ord("_")):
            self._zoom_at(1.0 / 1.15, self.window_width // 2, self.window_height // 2)
        elif low in (ord("h"), 81):
            self.pan_x -= 60
            self._clamp_pan()
        elif low in (ord("l"), 83):
            self.pan_x += 60
            self._clamp_pan()
        elif low in (ord("k"), 82):
            self.pan_y -= 60
            self._clamp_pan()
        elif low in (ord("j"), 84):
            self.pan_y += 60
            self._clamp_pan()
        return True

    def run(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        print("[INFO] HD map section gate editor started. Press i for help and s to save.")
        keep_running = True
        while keep_running:
            cv2.imshow(self.window_name, self._draw_map())
            key = cv2.waitKeyEx(20)
            if key >= 0:
                keep_running = self._handle_key(key)
        cv2.destroyWindow(self.window_name)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Draw section gates on an editable HD map YAML.")
    parser.add_argument("--hd-map-yaml", required=True, help="Editable HD map YAML to update in place.")
    parser.add_argument("--window-width", type=int, default=1600)
    parser.add_argument("--window-height", type=int, default=1000)
    parser.add_argument("--scale", type=float, default=1.0, help="Initial image scale. 0 fits the raster.")
    return parser


def main() -> int:
    if _IMPORT_ERROR is not None:
        raise SystemExit(f"hd_map_section_gate_editor.py requires numpy, opencv-python, and PyYAML: {_IMPORT_ERROR}")

    args = build_arg_parser().parse_args()
    hd_map_path = Path(args.hd_map_yaml).expanduser().resolve()
    data = yaml.safe_load(hd_map_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"HD map YAML root must be a mapping: {hd_map_path}")
    geometry, background = _source_raster_geometry(data, hd_map_path)
    lanes = _load_lanes(data, geometry)
    if not lanes:
        raise RuntimeError(f"HD map has no lanes: {hd_map_path}")
    gates = _load_gates(data, lanes, geometry)
    editor = SectionGateEditor(
        hd_map_path=hd_map_path,
        data=data,
        geometry=geometry,
        background=background,
        lanes=lanes,
        gates=gates,
        window_width=args.window_width,
        window_height=args.window_height,
        scale=args.scale,
    )
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

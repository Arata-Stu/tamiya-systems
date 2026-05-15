#!/usr/bin/env python3
"""
Interactive map section editor.

Defines polygon sections on map image pixels and saves them in CSV format.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@dataclass
class Section:
    name: str
    pixels: List[Tuple[int, int]]


@dataclass
class GateCandidate:
    name: str
    from_section: str
    to_section: str
    p0: Tuple[int, int]
    p1: Tuple[int, int]


def _sanitize_section_name(name: str, fallback: str) -> str:
    cleaned = name.strip().replace(",", "_").replace(" ", "_")
    allowed = []
    for c in cleaned:
        if c.isalnum() or c in ("_", "-", "."):
            allowed.append(c)
    result = "".join(allowed)
    return result if result else fallback


def _parse_map_yaml_minimal(map_yaml_path: Path) -> dict:
    data = {}
    with map_yaml_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()
    return data


def load_map_yaml(map_yaml_path: Path) -> dict:
    # Prefer yaml.safe_load when available; fallback to minimal parser.
    try:
        import yaml  # type: ignore

        with map_yaml_path.open("r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)
            if not isinstance(obj, dict):
                raise ValueError("map yaml root must be a mapping")
            return obj
    except Exception:
        return _parse_map_yaml_minimal(map_yaml_path)


def resolve_map_image_path(map_yaml_path: Path, map_obj: dict) -> Path:
    image_value = str(map_obj.get("image", "")).strip()
    if not image_value:
        raise ValueError("map yaml does not contain 'image' key")
    image_path = Path(image_value)
    if image_path.is_absolute():
        return image_path
    return (map_yaml_path.parent / image_path).resolve()


def save_sections_csv(
    output_path: Path,
    map_yaml_path: Path,
    image_width: int,
    image_height: int,
    sections: List[Section],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# map_section_definition_v1"])
        writer.writerow(["map_yaml", str(map_yaml_path)])
        writer.writerow(["image_width", str(image_width)])
        writer.writerow(["image_height", str(image_height)])
        for section in sections:
            row = ["section", section.name]
            for u, v in section.pixels:
                row.extend([str(int(u)), str(int(v))])
            writer.writerow(row)


def load_sections_csv(input_path: Path) -> tuple[Optional[int], Optional[int], List[Section]]:
    width: Optional[int] = None
    height: Optional[int] = None
    sections: List[Section] = []

    with input_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key or key.startswith("#"):
                continue

            if key == "image_width" and len(row) >= 2:
                width = int(row[1].strip())
                continue
            if key == "image_height" and len(row) >= 2:
                height = int(row[1].strip())
                continue
            if key != "section":
                continue
            if len(row) < 8:
                continue

            name = row[1].strip()
            values = [token.strip() for token in row[2:] if token.strip() != ""]
            if len(values) % 2 != 0:
                continue

            points: List[Tuple[int, int]] = []
            for i in range(0, len(values), 2):
                points.append((int(values[i]), int(values[i + 1])))
            if len(points) >= 3:
                sections.append(Section(name=name, pixels=points))

    return width, height, sections


def polygon_to_mask(
    points: List[Tuple[int, int]], width: int, height: int
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(points) < 3:
        return mask
    poly = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [poly], 255, lineType=cv2.LINE_8)
    return mask


def mask_to_polygons(mask: np.ndarray, epsilon_px: float = 1.2) -> List[List[Tuple[int, int]]]:
    if mask.dtype != np.uint8:
        mask_u8 = mask.astype(np.uint8)
    else:
        mask_u8 = mask
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons: List[List[Tuple[int, int]]] = []

    for contour in contours:
        if contour.shape[0] < 3:
            continue
        approx = cv2.approxPolyDP(contour, epsilon_px, True)
        if approx.shape[0] < 3:
            continue
        points = [(int(p[0][0]), int(p[0][1])) for p in approx]
        area = cv2.contourArea(approx)
        if area < 10.0:
            continue
        polygons.append(points)
    return polygons


def save_gates_csv(output_path: Path, gates: List[GateCandidate]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["# map_section_gate_definition_v1"])
        writer.writerow(["# gate,name,from_section,to_section,u0,v0,u1,v1"])
        for gate in gates:
            writer.writerow(
                [
                    "gate",
                    gate.name,
                    gate.from_section,
                    gate.to_section,
                    str(int(gate.p0[0])),
                    str(int(gate.p0[1])),
                    str(int(gate.p1[0])),
                    str(int(gate.p1[1])),
                ]
            )


class SectionEditor:
    def __init__(
        self,
        image: np.ndarray,
        map_yaml_path: Path,
        output_path: Path,
        scale: float,
        window_width: int,
        window_height: int,
        overlap_mode: str,
    ) -> None:
        self.image = image
        self.map_yaml_path = map_yaml_path
        self.output_path = output_path
        self.window_width = max(320, int(window_width))
        self.window_height = max(240, int(window_height))
        self.min_scale = 0.1
        self.max_scale = 20.0
        self.scale = 1.0
        self.window_name = "Map Section Editor"

        self.section_order: List[int] = []
        self.section_names: Dict[int, str] = {}
        self.section_masks: Dict[int, np.ndarray] = {}
        self.next_section_id = 1
        self.overlap_mode = overlap_mode if overlap_mode in ("overwrite", "keep_old") else "overwrite"
        self.current_points: List[Tuple[int, int]] = []
        self.section_counter = 1

        self.height, self.width = self.image.shape[:2]
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_offset = (0, 0)
        self.show_hud = True
        self.gate_output_path = self.output_path.with_name(
            f"{self.output_path.stem}_gates.csv"
        )
        self.cached_gate_count = 0
        self.needs_gate_recalc = True
        self.display_polygons_cache: Dict[int, List[List[Tuple[int, int]]]] = {}
        self.needs_polygon_cache_recalc = True
        self.section_palette: List[Tuple[int, int, int]] = [
            (255, 80, 80),   # blue-ish (BGR)
            (80, 80, 255),   # red-ish
            (255, 180, 70),  # orange
            (90, 210, 255),  # light blue
            (220, 120, 255), # pink
            (120, 220, 120), # green
        ]

        if scale > 0.0:
            self.scale = max(self.min_scale, min(self.max_scale, float(scale)))
        else:
            fit_scale = min(
                self.window_width / float(max(1, self.width)),
                self.window_height / float(max(1, self.height)),
            )
            self.scale = max(self.min_scale, min(self.max_scale, fit_scale))

    def _scaled_size(self) -> Tuple[int, int]:
        scaled_w = max(1, int(round(self.width * self.scale)))
        scaled_h = max(1, int(round(self.height * self.scale)))
        return scaled_w, scaled_h

    def _clamp_pan(self) -> None:
        scaled_w, scaled_h = self._scaled_size()
        max_pan_x = max(0, scaled_w - self.window_width)
        max_pan_y = max(0, scaled_h - self.window_height)
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

    def _zoom_at(self, factor: float, view_x: int, view_y: int) -> None:
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, old_scale * factor))
        if abs(new_scale - old_scale) < 1e-9:
            return

        sx = self.pan_x + view_x
        sy = self.pan_y + view_y
        u = sx / old_scale
        v = sy / old_scale

        self.scale = new_scale
        self.pan_x = int(round(u * new_scale - view_x))
        self.pan_y = int(round(v * new_scale - view_y))
        self._clamp_pan()

    def _reset_view(self) -> None:
        fit_scale = min(
            self.window_width / float(max(1, self.width)),
            self.window_height / float(max(1, self.height)),
        )
        self.scale = max(self.min_scale, min(self.max_scale, fit_scale))
        self.pan_x = 0
        self.pan_y = 0

    def _unique_name(self, base_name: str) -> str:
        existing = set(self.section_names.values())
        if base_name not in existing:
            return base_name
        idx = 1
        while True:
            candidate = f"{base_name}_{idx:02d}"
            if candidate not in existing:
                return candidate
            idx += 1

    def _section_count(self) -> int:
        return len(self.section_order)

    def _mark_sections_dirty(self) -> None:
        self.needs_gate_recalc = True
        self.needs_polygon_cache_recalc = True

    def _rebuild_polygon_cache(self) -> None:
        self.display_polygons_cache = {}
        for section_id in self.section_order:
            mask = self.section_masks.get(section_id)
            if mask is None:
                continue
            self.display_polygons_cache[section_id] = mask_to_polygons(mask, epsilon_px=1.4)
        self.needs_polygon_cache_recalc = False

    def _build_label_map(self) -> np.ndarray:
        label_map = np.zeros((self.height, self.width), dtype=np.int32)
        for section_id in self.section_order:
            mask = self.section_masks.get(section_id)
            if mask is None:
                continue
            label_map[mask > 0] = int(section_id)
        return label_map

    def _extract_gate_candidates(self) -> List[GateCandidate]:
        label_map = self._build_label_map()
        boundary_points: Dict[Tuple[int, int], List[Tuple[float, float]]] = {}

        left = label_map[:, :-1]
        right = label_map[:, 1:]
        h_mask = (left > 0) & (right > 0) & (left != right)
        hv, hu = np.where(h_mask)
        if hv.size > 0:
            a = left[hv, hu]
            b = right[hv, hu]
            low = np.minimum(a, b)
            high = np.maximum(a, b)
            for idx in range(hv.size):
                key = (int(low[idx]), int(high[idx]))
                boundary_points.setdefault(key, []).append(
                    (float(hu[idx]) + 0.5, float(hv[idx]))
                )

        top = label_map[:-1, :]
        bottom = label_map[1:, :]
        v_mask = (top > 0) & (bottom > 0) & (top != bottom)
        vv, vu = np.where(v_mask)
        if vv.size > 0:
            a = top[vv, vu]
            b = bottom[vv, vu]
            low = np.minimum(a, b)
            high = np.maximum(a, b)
            for idx in range(vv.size):
                key = (int(low[idx]), int(high[idx]))
                boundary_points.setdefault(key, []).append(
                    (float(vu[idx]), float(vv[idx]) + 0.5)
                )

        gates: List[GateCandidate] = []
        gate_idx = 1
        for (id_a, id_b), pts in boundary_points.items():
            if len(pts) < 12:
                continue
            section_a = self.section_names.get(id_a, f"section_{id_a}")
            section_b = self.section_names.get(id_b, f"section_{id_b}")

            points = np.array(pts, dtype=np.float64)
            mean = points.mean(axis=0)
            centered = points - mean
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, int(np.argmax(eigvals))]

            if np.linalg.norm(direction) < 1e-12:
                continue
            direction = direction / np.linalg.norm(direction)
            proj = centered @ direction
            p0 = mean + direction * proj.min()
            p1 = mean + direction * proj.max()

            if np.linalg.norm(p1 - p0) < 8.0:
                continue

            centroid_a = np.argwhere(label_map == id_a)
            centroid_b = np.argwhere(label_map == id_b)
            if centroid_a.size == 0 or centroid_b.size == 0:
                continue
            ca = np.array([centroid_a[:, 1].mean(), centroid_a[:, 0].mean()], dtype=np.float64)
            cb = np.array([centroid_b[:, 1].mean(), centroid_b[:, 0].mean()], dtype=np.float64)

            # Orient line so section_a is on the right side and section_b on the left side.
            line_x = float(p1[0] - p0[0])
            line_y = float(p1[1] - p0[1])
            side_a = line_x * float(ca[1] - p0[1]) - line_y * float(ca[0] - p0[0])
            side_b = line_x * float(cb[1] - p0[1]) - line_y * float(cb[0] - p0[0])
            if side_a > side_b:
                p0, p1 = p1, p0

            u0 = int(round(float(p0[0])))
            v0 = int(round(float(p0[1])))
            u1 = int(round(float(p1[0])))
            v1 = int(round(float(p1[1])))
            u0 = max(0, min(self.width - 1, u0))
            v0 = max(0, min(self.height - 1, v0))
            u1 = max(0, min(self.width - 1, u1))
            v1 = max(0, min(self.height - 1, v1))

            gates.append(
                GateCandidate(
                    name=f"gate_{gate_idx:02d}",
                    from_section=section_a,
                    to_section=section_b,
                    p0=(u0, v0),
                    p1=(u1, v1),
                )
            )
            gate_idx += 1

        return gates

    def _to_original_pixel(self, x: int, y: int) -> Tuple[int, int]:
        sx = self.pan_x + x
        sy = self.pan_y + y
        u = int(round(sx / self.scale))
        v = int(round(sy / self.scale))
        u = max(0, min(self.width - 1, u))
        v = max(0, min(self.height - 1, v))
        return (u, v)

    def _is_inside_map(self, x: int, y: int) -> bool:
        sx = self.pan_x + x
        sy = self.pan_y + y
        scaled_w, scaled_h = self._scaled_size()
        return 0 <= sx < scaled_w and 0 <= sy < scaled_h

    def _draw_polyline(
        self,
        canvas: np.ndarray,
        points: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        closed: bool,
        thickness: int = 2,
        point_radius: int = 4,
        fill_alpha: float = 0.0,
    ) -> None:
        if not points:
            return
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        if closed and len(points) >= 3 and fill_alpha > 0.0:
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color, lineType=cv2.LINE_AA)
            cv2.addWeighted(
                overlay,
                max(0.0, min(1.0, fill_alpha)),
                canvas,
                1.0 - max(0.0, min(1.0, fill_alpha)),
                0.0,
                dst=canvas,
            )
        cv2.polylines(canvas, [pts], closed, color, thickness, lineType=cv2.LINE_AA)
        for p in points:
            cv2.circle(canvas, p, point_radius, color, -1, lineType=cv2.LINE_AA)

    def _draw_text_with_outline(
        self,
        canvas: np.ndarray,
        text: str,
        org: Tuple[int, int],
        scale: float,
        color: Tuple[int, int, int],
        thickness: int,
    ) -> None:
        cv2.putText(
            canvas,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            text,
            org,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def _draw_panel(
        self, frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72
    ) -> None:
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(frame.shape[1], x + w)
        y1 = min(frame.shape[0], y + h)
        if x0 >= x1 or y0 >= y1:
            return
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0.0, dst=frame)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (230, 230, 230), 1, cv2.LINE_AA)

    def _point_in_rect(self, x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _hud_toggle_rect(self) -> Tuple[int, int, int, int]:
        width = 116
        height = 34
        margin = 10
        return (self.window_width - width - margin, margin, width, height)

    def _draw_button(
        self,
        frame: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        active: bool,
    ) -> None:
        x, y, w, h = rect
        fill = (55, 130, 215) if active else (42, 42, 42)
        border = (235, 235, 235) if active else (185, 185, 185)
        text_color = (255, 255, 255)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), fill, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = x + max(8, (w - text_w) // 2)
        text_y = y + max(text_h + 6, (h + text_h) // 2) - baseline
        self._draw_text_with_outline(frame, label, (text_x, text_y), 0.55, text_color, 2)

    def _draw_controls(self, frame: np.ndarray) -> None:
        self._draw_button(
            frame,
            self._hud_toggle_rect(),
            f"Help {'ON' if self.show_hud else 'OFF'}",
            self.show_hud,
        )

    def _draw_hud(self, frame: np.ndarray) -> None:
        if not self.show_hud:
            return

        mouse_u, mouse_v = self._to_original_pixel(self.last_mouse_x, self.last_mouse_y)
        section_count = self._section_count()
        recent_sections = [self.section_names[sid] for sid in self.section_order[-6:]]
        gate_candidates_text = (
            "stale" if self.needs_gate_recalc else str(self.cached_gate_count)
        )

        self._draw_panel(frame, 10, 10, min(900, self.window_width - 20), 210, alpha=0.72)
        self._draw_text_with_outline(
            frame,
            f"Sections: {section_count}",
            (24, 52),
            1.05,
            (80, 80, 255),
            3,
        )
        self._draw_text_with_outline(
            frame,
            (
                f"Editing Points: {len(self.current_points)}   Zoom: {self.scale:.2f}x"
                f"   Cursor: ({mouse_u}, {mouse_v})"
            ),
            (24, 88),
            0.72,
            (255, 255, 255),
            2,
        )
        self._draw_text_with_outline(
            frame,
            (
                "L-click:add point  n:finish section  d:delete last"
                f"  s:save+gates({gate_candidates_text})"
            ),
            (24, 120),
            0.72,
            (255, 255, 255),
            2,
        )
        self._draw_text_with_outline(
            frame,
            (
                "Wheel/+/-:zoom  Right-drag or H/J/K/L/Arrow:pan  0:reset view"
                f"  o:overlap={self.overlap_mode}  i:help  q:quit"
            ),
            (24, 150),
            0.68,
            (220, 220, 220),
            2,
        )

        if recent_sections:
            self._draw_text_with_outline(
                frame,
                "Recent Sections:",
                (24, 184),
                0.64,
                (160, 220, 255),
                2,
            )
            x = 220
            for idx, name in enumerate(recent_sections):
                color = self.section_palette[(section_count - len(recent_sections) + idx) % len(self.section_palette)]
                self._draw_text_with_outline(
                    frame,
                    name,
                    (x, 184),
                    0.62,
                    color,
                    2,
                )
                x += max(120, len(name) * 11)

    def _draw(self) -> np.ndarray:
        canvas = self.image.copy()
        shade = np.full_like(canvas, 30)
        canvas = cv2.addWeighted(canvas, 0.9, shade, 0.1, 0.0)

        if self.needs_polygon_cache_recalc:
            self._rebuild_polygon_cache()

        for idx, section_id in enumerate(self.section_order):
            mask = self.section_masks.get(section_id)
            if mask is None:
                continue
            polygons = self.display_polygons_cache.get(section_id, [])
            color = self.section_palette[idx % len(self.section_palette)]
            section_name = self.section_names.get(section_id, f"section_{section_id:02d}")

            centroid_xy: Optional[Tuple[float, float]] = None
            ys, xs = np.where(mask > 0)
            if xs.size > 0:
                centroid_xy = (float(xs.mean()), float(ys.mean()))

            for poly in polygons:
                self._draw_polyline(
                    canvas,
                    poly,
                    color,
                    closed=True,
                    thickness=4,
                    point_radius=5,
                    fill_alpha=0.14,
                )

            if centroid_xy is not None:
                text_pt = (int(centroid_xy[0]), int(centroid_xy[1]))
                self._draw_text_with_outline(canvas, section_name, text_pt, 0.78, color, 2)

        self._draw_polyline(
            canvas,
            self.current_points,
            (0, 0, 255),
            closed=False,
            thickness=4,
            point_radius=6,
        )
        if self.current_points:
            cv2.circle(canvas, self.current_points[0], 9, (255, 255, 255), 2, cv2.LINE_AA)

        scaled_w, scaled_h = self._scaled_size()
        interp = cv2.INTER_NEAREST if self.scale >= 1.0 else cv2.INTER_AREA
        scaled = cv2.resize(canvas, (scaled_w, scaled_h), interpolation=interp)

        self._clamp_pan()
        x0 = self.pan_x
        y0 = self.pan_y
        x1 = min(scaled_w, x0 + self.window_width)
        y1 = min(scaled_h, y0 + self.window_height)
        cropped = scaled[y0:y1, x0:x1]

        frame = np.zeros((self.window_height, self.window_width, 3), dtype=np.uint8)
        h, w = cropped.shape[:2]
        frame[0:h, 0:w] = cropped
        self._draw_hud(frame)
        self._draw_controls(frame)
        return frame

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
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

        if event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.pan_start_mouse[0]
            dy = y - self.pan_start_mouse[1]
            self.pan_x = self.pan_start_offset[0] - dx
            self.pan_y = self.pan_start_offset[1] - dy
            self._clamp_pan()
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            if _flags > 0:
                self._zoom_at(1.15, x, y)
            elif _flags < 0:
                self._zoom_at(1.0 / 1.15, x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._point_in_rect(x, y, self._hud_toggle_rect()):
                self.show_hud = not self.show_hud
                return
            if not self._is_inside_map(x, y):
                return
            u, v = self._to_original_pixel(x, y)
            self.current_points.append((u, v))

    def _finalize_current_section(self) -> None:
        if len(self.current_points) < 3:
            print("[WARN] Section requires at least 3 points.")
            return

        new_mask = polygon_to_mask(self.current_points, self.width, self.height)
        if np.count_nonzero(new_mask) == 0:
            print("[WARN] Empty polygon. Skip.")
            self.current_points.clear()
            return

        default_name = f"section_{self.section_counter:02d}"
        # Avoid blocking terminal input here; blocking input can freeze the GUI loop.
        name = _sanitize_section_name(default_name, default_name)
        name = self._unique_name(name)

        if self.overlap_mode == "overwrite":
            for section_id in list(self.section_order):
                mask = self.section_masks[section_id]
                mask[new_mask > 0] = 0
                if np.count_nonzero(mask) == 0:
                    self.section_order.remove(section_id)
                    self.section_masks.pop(section_id, None)
                    self.section_names.pop(section_id, None)
        else:
            occupied = np.zeros_like(new_mask, dtype=np.uint8)
            for section_id in self.section_order:
                occupied = cv2.bitwise_or(occupied, self.section_masks[section_id])
            new_mask[occupied > 0] = 0
            if np.count_nonzero(new_mask) == 0:
                print("[WARN] New polygon fully overlapped by existing sections.")
                self.current_points.clear()
                return

        section_id = self.next_section_id
        self.next_section_id += 1
        self.section_order.append(section_id)
        self.section_names[section_id] = name
        self.section_masks[section_id] = new_mask

        self.current_points.clear()
        self.section_counter += 1
        self._mark_sections_dirty()
        print(f"[INFO] Added section '{name}'")

    def _save(self) -> None:
        sections_to_save: List[Section] = []
        for section_id in self.section_order:
            section_name = self.section_names.get(section_id, f"section_{section_id:02d}")
            mask = self.section_masks.get(section_id)
            if mask is None:
                continue
            polygons = mask_to_polygons(mask, epsilon_px=1.3)
            for poly in polygons:
                if len(poly) >= 3:
                    sections_to_save.append(Section(name=section_name, pixels=poly))

        save_sections_csv(
            output_path=self.output_path,
            map_yaml_path=self.map_yaml_path,
            image_width=self.width,
            image_height=self.height,
            sections=sections_to_save,
        )

        gates = self._extract_gate_candidates()
        save_gates_csv(self.gate_output_path, gates)
        self.cached_gate_count = len(gates)
        self.needs_gate_recalc = False

        print(
            f"[INFO] Saved {len(sections_to_save)} polygons to {self.output_path} "
            f"and {len(gates)} gates to {self.gate_output_path}"
        )

    def load_existing(self, csv_path: Path) -> None:
        width, height, sections = load_sections_csv(csv_path)
        if width is not None and width != self.width:
            print(f"[WARN] CSV image_width ({width}) != map image width ({self.width})")
        if height is not None and height != self.height:
            print(f"[WARN] CSV image_height ({height}) != map image height ({self.height})")
        self.section_order.clear()
        self.section_masks.clear()
        self.section_names.clear()
        self.next_section_id = 1

        name_to_id: Dict[str, int] = {}
        for section in sections:
            if section.name in name_to_id:
                section_id = name_to_id[section.name]
            else:
                section_id = self.next_section_id
                self.next_section_id += 1
                self.section_order.append(section_id)
                self.section_names[section_id] = section.name
                self.section_masks[section_id] = np.zeros((self.height, self.width), dtype=np.uint8)
                name_to_id[section.name] = section_id

            poly_mask = polygon_to_mask(section.pixels, self.width, self.height)
            self.section_masks[section_id] = cv2.bitwise_or(self.section_masks[section_id], poly_mask)

        self.section_counter = len(self.section_order) + 1
        self._mark_sections_dirty()
        print(f"[INFO] Loaded {len(self.section_order)} sections from {csv_path}")

    def run(self) -> None:
        print("[INFO] Press 'i' or click the Help button to toggle the instruction panel.")

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.window_width, self.window_height)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        while True:
            frame = self._draw()
            cv2.imshow(self.window_name, frame)
            key = cv2.waitKeyEx(20)
            if key < 0:
                continue

            key_ascii = key & 0xFF
            if key in (27,) or key_ascii == ord("q"):
                break
            if key_ascii == ord("i"):
                self.show_hud = not self.show_hud
            elif key_ascii == ord("u") and self.current_points:
                self.current_points.pop()
            elif key_ascii == ord("c"):
                self.current_points.clear()
            elif key_ascii == ord("d") and self.section_order:
                removed_id = self.section_order.pop()
                removed_name = self.section_names.pop(removed_id, f"section_{removed_id:02d}")
                self.section_masks.pop(removed_id, None)
                self._mark_sections_dirty()
                print(f"[INFO] Removed section '{removed_name}'")
            elif key_ascii == ord("n"):
                self._finalize_current_section()
            elif key_ascii == ord("s"):
                self._save()
            elif key_ascii == ord("g"):
                gates = self._extract_gate_candidates()
                save_gates_csv(self.gate_output_path, gates)
                self.cached_gate_count = len(gates)
                self.needs_gate_recalc = False
                print(f"[INFO] Saved {len(gates)} gates to {self.gate_output_path}")
            elif key_ascii == ord("o"):
                self.overlap_mode = "keep_old" if self.overlap_mode == "overwrite" else "overwrite"
                print(f"[INFO] overlap_mode = {self.overlap_mode}")
            elif key_ascii in (ord("+"), ord("=")):
                self._zoom_at(1.15, self.window_width // 2, self.window_height // 2)
            elif key_ascii in (ord("-"), ord("_")):
                self._zoom_at(1.0 / 1.15, self.window_width // 2, self.window_height // 2)
            elif key_ascii == ord("0"):
                self._reset_view()
            elif key_ascii == ord("h"):
                self.pan_x -= 80
                self._clamp_pan()
            elif key_ascii == ord("l"):
                self.pan_x += 80
                self._clamp_pan()
            elif key_ascii == ord("k"):
                self.pan_y -= 80
                self._clamp_pan()
            elif key_ascii == ord("j"):
                self.pan_y += 80
                self._clamp_pan()
            elif key in (2424832, 65361):  # Left arrow
                self.pan_x -= 80
                self._clamp_pan()
            elif key in (2555904, 65363):  # Right arrow
                self.pan_x += 80
                self._clamp_pan()
            elif key in (2490368, 65362):  # Up arrow
                self.pan_y -= 80
                self._clamp_pan()
            elif key in (2621440, 65364):  # Down arrow
                self.pan_y += 80
                self._clamp_pan()

        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map section editor GUI")
    parser.add_argument("--map-yaml", required=True, help="Path to map YAML")
    parser.add_argument("--output", default="", help="Output CSV path")
    parser.add_argument(
        "--load",
        default="",
        help="Optional existing CSV to load before editing (default: --output if exists)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="Initial display scale. <=0 means auto-fit to window.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1400,
        help="Editor window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=900,
        help="Editor window height in pixels.",
    )
    parser.add_argument(
        "--overlap-mode",
        choices=["overwrite", "keep_old"],
        default="overwrite",
        help="How to resolve overlaps when adding a new section.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency. Please install python_ws/requirements.txt "
            f"(root cause: {_IMPORT_ERROR})"
        ) from _IMPORT_ERROR

    map_yaml_path = Path(args.map_yaml).expanduser().resolve()
    if not map_yaml_path.exists():
        raise FileNotFoundError(f"map yaml not found: {map_yaml_path}")

    map_obj = load_map_yaml(map_yaml_path)
    map_image_path = resolve_map_image_path(map_yaml_path, map_obj)
    if not map_image_path.exists():
        raise FileNotFoundError(f"map image not found: {map_image_path}")

    image = cv2.imread(str(map_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to load map image: {map_image_path}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = (map_yaml_path.parent / "sections_pixels.csv").resolve()

    editor = SectionEditor(
        image=image,
        map_yaml_path=map_yaml_path,
        output_path=output_path,
        scale=args.scale,
        window_width=args.window_width,
        window_height=args.window_height,
        overlap_mode=args.overlap_mode,
    )

    load_path = Path(args.load).expanduser().resolve() if args.load else output_path
    if load_path.exists():
        try:
            editor.load_existing(load_path)
        except Exception as exc:
            print(f"[WARN] Failed to load existing CSV ({load_path}): {exc}")

    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

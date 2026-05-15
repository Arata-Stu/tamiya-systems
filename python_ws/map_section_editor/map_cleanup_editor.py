#!/usr/bin/env python3
"""
Interactive occupancy-map cleanup editor for centerline preprocessing.

This tool is intended for cases where automatic centerline extraction fails on
branching or noisy maps. It lets an operator paint black/white directly on a
PNG/PGM map, then save a cleaned PNG for later centerline/raceline generation.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

try:
    import cv2
    import numpy as np

    _IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    cv2 = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc


@dataclass
class EditorImages:
    raw_input: np.ndarray
    session_base: np.ndarray


class MapCleanupEditor:
    def __init__(
        self,
        images: EditorImages,
        output_path: Path,
        scale: float,
        window_width: int,
        window_height: int,
        brush_radius: int,
        undo_depth: int,
        loaded_saved_output: bool,
    ) -> None:
        self.raw_input = images.raw_input.copy()
        self.session_base = images.session_base.copy()
        self.image = images.session_base.copy()
        self.output_path = output_path
        self.loaded_saved_output = loaded_saved_output

        self.window_name = "Map Cleanup Editor"
        self.window_width = max(320, int(window_width))
        self.window_height = max(240, int(window_height))
        self.min_scale = 0.1
        self.max_scale = 20.0
        self.scale = 1.0

        self.height, self.width = self.image.shape[:2]
        self.pan_x = 0
        self.pan_y = 0
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.is_panning = False
        self.pan_start_mouse = (0, 0)
        self.pan_start_offset = (0, 0)
        self.is_drawing = False
        self.last_draw_uv: Optional[Tuple[int, int]] = None

        self.mode = "paint_black"
        self.brush_radius = max(1, int(brush_radius))
        self.show_hud = True
        self.undo_stack: Deque[np.ndarray] = deque(maxlen=max(1, int(undo_depth)))
        self.has_unsaved_changes = False

        if scale > 0.0:
            self.scale = max(self.min_scale, min(self.max_scale, float(scale)))
        else:
            self._reset_view()

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

    def _to_original_pixel(self, x: int, y: int) -> Tuple[int, int]:
        sx = self.pan_x + x
        sy = self.pan_y + y
        u = int(round(sx / self.scale))
        v = int(round(sy / self.scale))
        u = max(0, min(self.width - 1, u))
        v = max(0, min(self.height - 1, v))
        return (u, v)

    def _to_view_pixel(self, u: int, v: int) -> Tuple[int, int]:
        x = int(round(u * self.scale - self.pan_x))
        y = int(round(v * self.scale - self.pan_y))
        return (x, y)

    def _is_inside_map(self, x: int, y: int) -> bool:
        sx = self.pan_x + x
        sy = self.pan_y + y
        scaled_w, scaled_h = self._scaled_size()
        return 0 <= sx < scaled_w and 0 <= sy < scaled_h

    def _sync_dirty_state(self) -> None:
        self.has_unsaved_changes = not np.array_equal(self.image, self.session_base)

    def _push_undo(self) -> None:
        self.undo_stack.append(self.image.copy())

    def _point_in_rect(self, x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
        rx, ry, rw, rh = rect
        return rx <= x < rx + rw and ry <= y < ry + rh

    def _brush_diameter_px(self) -> int:
        return self.brush_radius * 2 + 1

    def _change_brush_radius(self, delta: int) -> None:
        self.brush_radius = max(1, min(512, self.brush_radius + delta))

    def _hud_toggle_rect(self) -> Tuple[int, int, int, int]:
        width = 116
        height = 34
        margin = 10
        return (self.window_width - width - margin, margin, width, height)

    def _brush_minus_rect(self) -> Tuple[int, int, int, int]:
        width = 34
        height = 34
        margin = 10
        x = self.window_width - 116 - 34 - 146 - margin - 8
        return (x, margin, width, height)

    def _brush_value_rect(self) -> Tuple[int, int, int, int]:
        width = 104
        height = 34
        margin = 10
        x = self.window_width - 116 - 104 - 34 - margin - 4
        return (x, margin, width, height)

    def _brush_plus_rect(self) -> Tuple[int, int, int, int]:
        width = 34
        height = 34
        margin = 10
        x = self.window_width - 116 - 34 - margin
        return (x, margin, width, height)

    def _brush_value(self) -> int:
        return 0 if self.mode == "paint_black" else 255

    def _paint_segment(self, start_uv: Tuple[int, int], end_uv: Tuple[int, int]) -> None:
        color = self._brush_value()
        thickness = max(1, self.brush_radius * 2 + 1)
        cv2.line(self.image, start_uv, end_uv, color, thickness=thickness, lineType=cv2.LINE_8)
        cv2.circle(self.image, end_uv, self.brush_radius, color, -1, lineType=cv2.LINE_8)

    def _save(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(self.output_path), self.image):
            raise RuntimeError(f"Failed to save image: {self.output_path}")
        self.session_base = self.image.copy()
        self.loaded_saved_output = True
        self.has_unsaved_changes = False
        print(f"[INFO] Saved cleaned map: {self.output_path}")

    def _reset_to_session_base(self) -> None:
        self._push_undo()
        self.image = self.session_base.copy()
        self._sync_dirty_state()
        print("[INFO] Reverted unsaved edits.")

    def _reset_to_raw_input(self) -> None:
        self._push_undo()
        self.image = self.raw_input.copy()
        self._sync_dirty_state()
        print("[INFO] Reset to raw input map.")

    def _undo(self) -> None:
        if not self.undo_stack:
            print("[INFO] Nothing to undo.")
            return
        self.image = self.undo_stack.pop()
        self._sync_dirty_state()
        print("[INFO] Undo applied.")

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
        self, frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.74
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

    def _draw_button(
        self,
        frame: np.ndarray,
        rect: Tuple[int, int, int, int],
        label: str,
        active: bool = False,
    ) -> None:
        x, y, w, h = rect
        fill = (55, 130, 215) if active else (42, 42, 42)
        border = (235, 235, 235) if active else (185, 185, 185)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), fill, -1)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0.0, dst=frame)
        cv2.rectangle(frame, (x, y), (x + w, y + h), border, 1, cv2.LINE_AA)

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        text_x = x + max(8, (w - text_w) // 2)
        text_y = y + max(text_h + 6, (h + text_h) // 2) - baseline
        self._draw_text_with_outline(frame, label, (text_x, text_y), 0.55, (255, 255, 255), 2)

    def _draw_controls(self, frame: np.ndarray) -> None:
        self._draw_button(frame, self._brush_minus_rect(), "-", False)
        self._draw_button(frame, self._brush_value_rect(), f"{self._brush_diameter_px()} px", False)
        self._draw_button(frame, self._brush_plus_rect(), "+", False)
        self._draw_button(
            frame,
            self._hud_toggle_rect(),
            f"Help {'ON' if self.show_hud else 'OFF'}",
            self.show_hud,
        )

    def _draw_hud(self, frame: np.ndarray) -> None:
        if not self.show_hud:
            return

        mode_text = "black" if self.mode == "paint_black" else "white"
        source_text = "saved output" if self.loaded_saved_output else "raw input"
        dirty_text = "yes" if self.has_unsaved_changes else "no"
        cursor_u, cursor_v = self._to_original_pixel(self.last_mouse_x, self.last_mouse_y)

        panel_width = min(1240, self.window_width - 20)
        self._draw_panel(frame, 10, 10, panel_width, 190, alpha=0.76)
        self._draw_text_with_outline(
            frame,
            "Map Cleanup Editor",
            (24, 44),
            0.95,
            (80, 210, 255),
            3,
        )
        self._draw_text_with_outline(
            frame,
            (
                f"Mode: {mode_text}   Brush: {self._brush_diameter_px()}px   Zoom: {self.scale:.2f}x"
                f"   Cursor: ({cursor_u}, {cursor_v})   Unsaved: {dirty_text}"
            ),
            (24, 78),
            0.66,
            (255, 255, 255),
            2,
        )
        self._draw_text_with_outline(
            frame,
            f"Session base: {source_text}   Output: {self.output_path}",
            (24, 108),
            0.58,
            (220, 220, 220),
            2,
        )
        self._draw_text_with_outline(
            frame,
            "Left-drag:paint  Right-drag or H/J/K/L/Arrow:pan  Wheel/+/-:zoom  [/] or ,/.:brush",
            (24, 138),
            0.58,
            (220, 220, 220),
            2,
        )
        self._draw_text_with_outline(
            frame,
            "b:black  e:white  u:undo  r:revert unsaved  R:reset raw map  i:help  s:save  0:reset view  q/Esc:quit",
            (24, 166),
            0.58,
            (220, 220, 220),
            2,
        )

    def _draw(self) -> np.ndarray:
        canvas = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

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

        if self._is_inside_map(self.last_mouse_x, self.last_mouse_y):
            cursor_u, cursor_v = self._to_original_pixel(self.last_mouse_x, self.last_mouse_y)
            cursor_x, cursor_y = self._to_view_pixel(cursor_u, cursor_v)
            radius = max(2, int(round(self.brush_radius * self.scale)))
            color = (30, 30, 230) if self.mode == "paint_black" else (40, 200, 90)
            cv2.circle(frame, (cursor_x, cursor_y), radius, color, 1, cv2.LINE_AA)

        self._draw_hud(frame)
        self._draw_controls(frame)
        return frame

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

        if event == cv2.EVENT_MOUSEMOVE and self.is_panning:
            dx = x - self.pan_start_mouse[0]
            dy = y - self.pan_start_mouse[1]
            self.pan_x = self.pan_start_offset[0] - dx
            self.pan_y = self.pan_start_offset[1] - dy
            self._clamp_pan()
            return

        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self._zoom_at(1.15, x, y)
            elif flags < 0:
                self._zoom_at(1.0 / 1.15, x, y)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            if self._point_in_rect(x, y, self._hud_toggle_rect()):
                self.show_hud = not self.show_hud
                return
            if self._point_in_rect(x, y, self._brush_minus_rect()):
                self._change_brush_radius(-1)
                return
            if self._point_in_rect(x, y, self._brush_plus_rect()):
                self._change_brush_radius(1)
                return
            if not self._is_inside_map(x, y):
                return
            self._push_undo()
            self.is_drawing = True
            uv = self._to_original_pixel(x, y)
            self.last_draw_uv = uv
            self._paint_segment(uv, uv)
            self.has_unsaved_changes = True
            return

        if event == cv2.EVENT_MOUSEMOVE and self.is_drawing and (flags & cv2.EVENT_FLAG_LBUTTON):
            if not self._is_inside_map(x, y):
                return
            uv = self._to_original_pixel(x, y)
            if self.last_draw_uv is None:
                self.last_draw_uv = uv
            self._paint_segment(self.last_draw_uv, uv)
            self.last_draw_uv = uv
            self.has_unsaved_changes = True
            return

        if event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.last_draw_uv = None

    def run(self) -> None:
        print("[INFO] Launching map cleanup editor.")
        print("[INFO] Paint black to remove branches/noise, or white to reopen track areas.")
        print("[INFO] Press 's' to save the cleaned PNG before closing.")
        print("[INFO] Press 'i' or click the Help button to toggle the instruction panel.")
        print("[INFO] Use '[' / ']' or ',' / '.' or the +/- buttons to change brush size.")

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
            elif key_ascii == ord("s"):
                self._save()
            elif key_ascii == ord("u"):
                self._undo()
            elif key_ascii == ord("b"):
                self.mode = "paint_black"
            elif key_ascii == ord("e"):
                self.mode = "paint_white"
            elif key_ascii == ord("r"):
                self._reset_to_session_base()
            elif key_ascii == ord("R"):
                self._reset_to_raw_input()
            elif key_ascii in (ord("+"), ord("=")):
                self._zoom_at(1.15, self.window_width // 2, self.window_height // 2)
            elif key_ascii in (ord("-"), ord("_")):
                self._zoom_at(1.0 / 1.15, self.window_width // 2, self.window_height // 2)
            elif key_ascii == ord("0"):
                self._reset_view()
            elif key_ascii in (ord("["), ord("{"), ord(","), ord("<")):
                self._change_brush_radius(-1)
            elif key_ascii in (ord("]"), ord("}"), ord("."), ord(">")):
                self._change_brush_radius(1)
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
            elif key in (2424832, 65361):
                self.pan_x -= 80
                self._clamp_pan()
            elif key in (2555904, 65363):
                self.pan_x += 80
                self._clamp_pan()
            elif key in (2490368, 65362):
                self.pan_y -= 80
                self._clamp_pan()
            elif key in (2621440, 65364):
                self.pan_y += 80
                self._clamp_pan()

        cv2.destroyAllWindows()
        if self.has_unsaved_changes:
            print("[INFO] Closed editor with unsaved changes. Saved output was not updated.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map cleanup editor GUI")
    parser.add_argument("--input", required=True, help="Path to source PNG/PGM map image")
    parser.add_argument("--output", required=True, help="Path to cleaned PNG output")
    parser.add_argument(
        "--scale",
        type=float,
        default=0.0,
        help="Initial display scale. <=0 means auto-fit to window.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1500,
        help="Editor window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=960,
        help="Editor window height in pixels.",
    )
    parser.add_argument(
        "--brush-radius",
        type=int,
        default=8,
        help="Initial paint brush radius in pixels.",
    )
    parser.add_argument(
        "--undo-depth",
        type=int,
        default=32,
        help="Maximum number of undo snapshots.",
    )
    return parser.parse_args()


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"failed to load image: {path}")
    return image


def main() -> int:
    args = parse_args()

    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "Missing dependency. Please install python_ws/requirements.txt "
            f"(root cause: {_IMPORT_ERROR})"
        ) from _IMPORT_ERROR

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"input map not found: {input_path}")

    raw_input = load_grayscale(input_path)
    loaded_saved_output = output_path.exists()
    if loaded_saved_output:
        session_base = load_grayscale(output_path)
        if session_base.shape != raw_input.shape:
            raise RuntimeError(
                f"saved output size {session_base.shape} does not match input size {raw_input.shape}: {output_path}"
            )
        print(f"[INFO] Loaded existing cleaned map as session base: {output_path}")
    else:
        session_base = raw_input.copy()

    editor = MapCleanupEditor(
        images=EditorImages(raw_input=raw_input, session_base=session_base),
        output_path=output_path,
        scale=args.scale,
        window_width=args.window_width,
        window_height=args.window_height,
        brush_radius=args.brush_radius,
        undo_depth=args.undo_depth,
        loaded_saved_output=loaded_saved_output,
    )
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

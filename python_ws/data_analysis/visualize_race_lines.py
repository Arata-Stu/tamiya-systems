#!/usr/bin/env python3
"""Render centerline/raceline CSV overlays on a map image."""

from __future__ import annotations

import argparse
import math
import os
from ast import literal_eval
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf


def load_map_yaml(yaml_path: Path) -> dict:
    cfg = OmegaConf.load(yaml_path)
    return OmegaConf.to_container(cfg, resolve=True)


def parse_origin(origin_value) -> Tuple[float, float, float]:
    if isinstance(origin_value, str):
        origin_value = literal_eval(origin_value)
    if isinstance(origin_value, (list, tuple)) and len(origin_value) >= 2:
        yaw = float(origin_value[2]) if len(origin_value) >= 3 else 0.0
        return float(origin_value[0]), float(origin_value[1]), yaw
    raise ValueError("map yaml 'origin' must be a list like [x, y, yaw]")


def resolve_map_image(yaml_path: Optional[Path], map_path: Optional[Path]) -> Optional[Path]:
    if map_path is not None:
        return map_path
    if yaml_path is None:
        return None

    cfg = load_map_yaml(yaml_path)
    image = cfg.get("image")
    if not image:
        return None
    return (yaml_path.parent / str(image)).resolve()


def load_centerline(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 2:
        raise RuntimeError(f"centerline CSV needs at least x,y columns: {path}")
    return data[:, :2]


def load_raceline(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=";", comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 3:
        raise RuntimeError(f"raceline CSV needs at least s,x,y columns: {path}")
    return data[:, 1:3]


def world_to_image_pixels(points: np.ndarray, yaml_path: Path, image_shape: Tuple[int, int]) -> np.ndarray:
    cfg = load_map_yaml(yaml_path)
    resolution = float(cfg["resolution"])
    origin_x, origin_y, origin_yaw = parse_origin(cfg["origin"])
    h, _ = image_shape

    dx = points[:, 0] - origin_x
    dy = points[:, 1] - origin_y
    cos_t = math.cos(origin_yaw)
    sin_t = math.sin(origin_yaw)
    grid_x = (cos_t * dx + sin_t * dy) / resolution
    grid_y = (-sin_t * dx + cos_t * dy) / resolution

    img_x = grid_x
    img_y = (h - 1) - grid_y
    return np.column_stack([img_x, img_y])


def fit_points_to_canvas(points: np.ndarray, size: int, padding: int) -> np.ndarray:
    min_xy = points.min(axis=0)
    max_xy = points.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-9)
    scale = (size - 2 * padding) / float(np.max(span))
    out = (points - min_xy) * scale + padding
    out[:, 1] = size - out[:, 1]
    return out


def draw_polyline(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    thickness: int,
    closed: bool = True,
) -> None:
    if points is None or len(points) < 2:
        return
    pts = np.round(points).astype(np.int32)
    cv2.polylines(image, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


def draw_start_marker(image: np.ndarray, points: np.ndarray, color: Tuple[int, int, int]) -> None:
    if points is None or len(points) == 0:
        return
    p = tuple(np.round(points[0]).astype(np.int32))
    cv2.circle(image, p, 5, color, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, p, 8, (255, 255, 255), 1, lineType=cv2.LINE_AA)


def draw_legend(image: np.ndarray, has_centerline: bool, has_raceline: bool) -> None:
    y = 24
    if has_centerline:
        cv2.line(image, (16, y), (52, y), (255, 90, 40), 3, lineType=cv2.LINE_AA)
        cv2.putText(image, "centerline", (60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2)
        y += 26
    if has_raceline:
        cv2.line(image, (16, y), (52, y), (40, 40, 255), 3, lineType=cv2.LINE_AA)
        cv2.putText(image, "raceline", (60, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 2)


def build_canvas(
    map_image_path: Optional[Path],
    all_points: np.ndarray,
    canvas_size: int,
    padding: int,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]]]:
    if map_image_path is None:
        canvas = np.full((canvas_size, canvas_size, 3), 245, dtype=np.uint8)
        return canvas, None

    gray = cv2.imread(str(map_image_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read map image: {map_image_path}")

    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # Soften the map slightly so colored lines remain readable.
    canvas = cv2.addWeighted(canvas, 0.72, np.full_like(canvas, 255), 0.28, 0.0)
    return canvas, gray.shape[:2]


def run(args: argparse.Namespace) -> None:
    yaml_path = Path(args.yaml).expanduser().resolve() if args.yaml else None
    map_path = Path(args.map).expanduser().resolve() if args.map else None
    map_image_path = resolve_map_image(yaml_path, map_path)

    centerline = load_centerline(Path(args.centerline).expanduser().resolve()) if args.centerline else None
    raceline = load_raceline(Path(args.raceline).expanduser().resolve()) if args.raceline else None
    if centerline is None and raceline is None:
        raise RuntimeError("At least one of --centerline or --raceline is required.")

    all_points = np.vstack([p for p in (centerline, raceline) if p is not None])
    canvas, map_shape = build_canvas(
        map_image_path=map_image_path,
        all_points=all_points,
        canvas_size=args.canvas_size,
        padding=args.padding_px,
    )

    if yaml_path is not None and map_shape is not None:
        center_px = world_to_image_pixels(centerline, yaml_path, map_shape) if centerline is not None else None
        race_px = world_to_image_pixels(raceline, yaml_path, map_shape) if raceline is not None else None
    elif map_shape is not None:
        h, _ = map_shape
        center_px = np.column_stack([centerline[:, 0], (h - 1) - centerline[:, 1]]) if centerline is not None else None
        race_px = np.column_stack([raceline[:, 0], (h - 1) - raceline[:, 1]]) if raceline is not None else None
    else:
        fitted = fit_points_to_canvas(all_points, size=args.canvas_size, padding=args.padding_px)
        n_center = 0 if centerline is None else len(centerline)
        center_px = fitted[:n_center] if centerline is not None else None
        race_px = fitted[n_center:] if raceline is not None else None

    draw_polyline(canvas, center_px, color=(255, 90, 40), thickness=args.centerline_thickness)
    draw_polyline(canvas, race_px, color=(40, 40, 255), thickness=args.raceline_thickness)
    draw_start_marker(canvas, center_px, color=(255, 90, 40))
    draw_start_marker(canvas, race_px, color=(40, 40, 255))
    if not args.no_legend:
        draw_legend(canvas, has_centerline=centerline is not None, has_raceline=raceline is not None)

    out_path = Path(args.output).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), canvas):
        raise RuntimeError(f"Failed to write output image: {out_path}")

    print(f"Wrote line preview image: {out_path}")
    if map_image_path is not None:
        print(f"Map image: {map_image_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render centerline/raceline overlays.")
    p.add_argument("--map", default=None, help="Optional map image path")
    p.add_argument("--yaml", default=None, help="Optional map yaml path; image is inferred when --map is omitted")
    p.add_argument("--centerline", default=None, help="Optional centerline CSV")
    p.add_argument("--raceline", default=None, help="Optional raceline CSV")
    p.add_argument("--output", required=True, help="Output preview image path")
    p.add_argument("--canvas-size", type=int, default=1200, help="Canvas size when no map image is provided")
    p.add_argument("--padding-px", type=int, default=32)
    p.add_argument("--centerline-thickness", type=int, default=1)
    p.add_argument("--raceline-thickness", type=int, default=1)
    p.add_argument("--no-legend", action="store_true")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

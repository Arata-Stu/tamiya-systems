#!/usr/bin/env python3
"""Render global-localization quality overlays from evaluation CSV and map yaml."""

from __future__ import annotations

import argparse
import csv
import math
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class MapMeta:
    image_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    image_width: int
    image_height: int


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


def _parse_origin(origin_value) -> tuple[float, float, float]:
    if isinstance(origin_value, (list, tuple)) and len(origin_value) >= 3:
        return float(origin_value[0]), float(origin_value[1]), float(origin_value[2])
    if isinstance(origin_value, str):
        parsed = literal_eval(origin_value)
        if isinstance(parsed, (list, tuple)) and len(parsed) >= 3:
            return float(parsed[0]), float(parsed[1]), float(parsed[2])
    raise ValueError("map yaml 'origin' must be a 3-element list")


def _load_map_image_rgb(image_path: Path):
    try:
        import cv2  # type: ignore
        import numpy as np

        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read map image: {image_path}")
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 4:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb.dtype.kind in ("u", "i"):
            rgb = rgb.astype("float32") / 255.0
        else:
            rgb = rgb.astype("float32")
            vmax = float(rgb.max()) if rgb.size > 0 else 1.0
            if vmax > 1.0:
                rgb = rgb / 255.0
        return rgb
    except Exception:
        import matplotlib.image as mpimg
        import numpy as np

        img = mpimg.imread(str(image_path))
        if img.ndim == 2:
            rgb = np.stack([img, img, img], axis=-1)
        else:
            rgb = img[..., :3]
        rgb = rgb.astype("float32")
        vmax = float(rgb.max()) if rgb.size > 0 else 1.0
        if vmax > 1.0:
            rgb = rgb / 255.0
        return rgb


def _load_map_meta(map_yaml_path: Path) -> MapMeta:
    map_obj = load_map_yaml(map_yaml_path)
    image_path = resolve_map_image_path(map_yaml_path, map_obj)
    resolution = float(map_obj["resolution"])
    origin_x, origin_y, origin_yaw = _parse_origin(map_obj["origin"])
    image = _load_map_image_rgb(image_path)
    h, w = image.shape[:2]
    return MapMeta(
        image_path=image_path,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        origin_yaw=origin_yaw,
        image_width=int(w),
        image_height=int(h),
    )


def _world_to_pixel(x: float, y: float, map_meta: MapMeta) -> tuple[float, float]:
    dx = x - map_meta.origin_x
    dy = y - map_meta.origin_y
    cos_t = math.cos(map_meta.origin_yaw)
    sin_t = math.sin(map_meta.origin_yaw)
    gx = (cos_t * dx + sin_t * dy) / map_meta.resolution
    gy = (-sin_t * dx + cos_t * dy) / map_meta.resolution
    u = gx - 0.5
    v = float(map_meta.image_height) - gy - 0.5
    return u, v


def _parse_opt_float(value: str) -> Optional[float]:
    text = (value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _default_quality_output_paths(csv_path: Path) -> tuple[Path, Path]:
    stem = csv_path.with_suffix("")
    return (
        stem.parent / f"{stem.name}_success_rate.png",
        stem.parent / f"{stem.name}_points.png",
    )


def render_map_quality_images(
    csv_path: Path,
    map_yaml_path: Path,
    good_pos_error_threshold_m: float,
    grid_size_m: float,
    min_samples_per_cell: int,
    rate_output_path: Optional[Path],
    points_output_path: Optional[Path],
) -> tuple[Path, Path]:
    import matplotlib.pyplot as plt
    import numpy as np

    map_meta = _load_map_meta(map_yaml_path)
    map_img = _load_map_image_rgb(map_meta.image_path)

    if rate_output_path is None or points_output_path is None:
        default_rate, default_points = _default_quality_output_paths(csv_path)
        if rate_output_path is None:
            rate_output_path = default_rate
        if points_output_path is None:
            points_output_path = default_points

    rate_output_path.parent.mkdir(parents=True, exist_ok=True)
    points_output_path.parent.mkdir(parents=True, exist_ok=True)

    anchor_u = []
    anchor_v = []
    is_good = []
    is_bad = []
    is_fail = []
    is_ok_no_reference = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = (row.get("status") or "").strip()
            ref_x = _parse_opt_float(row.get("reference_x", ""))
            ref_y = _parse_opt_float(row.get("reference_y", ""))
            loc_x = _parse_opt_float(row.get("localization_x", ""))
            loc_y = _parse_opt_float(row.get("localization_y", ""))
            pos_error = _parse_opt_float(row.get("position_error_m", ""))

            if ref_x is not None and ref_y is not None:
                wx, wy = ref_x, ref_y
            elif loc_x is not None and loc_y is not None:
                wx, wy = loc_x, loc_y
            else:
                continue

            u, v = _world_to_pixel(wx, wy, map_meta)
            anchor_u.append(u)
            anchor_v.append(v)

            ok = (status == "ok") and (pos_error is not None)
            good = ok and (pos_error <= good_pos_error_threshold_m)
            bad = ok and (pos_error > good_pos_error_threshold_m)
            fail = status in ("localization_timeout", "trigger_failed")
            ok_no_reference = status == "ok_no_reference"

            is_good.append(good)
            is_bad.append(bad)
            is_fail.append(fail)
            is_ok_no_reference.append(ok_no_reference)

    if not anchor_u:
        raise RuntimeError("No plottable points found in evaluation CSV.")

    u_arr = np.asarray(anchor_u, dtype=np.float64)
    v_arr = np.asarray(anchor_v, dtype=np.float64)
    good_arr = np.asarray(is_good, dtype=bool)
    bad_arr = np.asarray(is_bad, dtype=bool)
    fail_arr = np.asarray(is_fail, dtype=bool)
    ok_no_ref_arr = np.asarray(is_ok_no_reference, dtype=bool)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img, origin="upper")
    if np.any(good_arr):
        ax.scatter(
            u_arr[good_arr],
            v_arr[good_arr],
            c="#0066ff",
            s=36,
            alpha=0.9,
            label=f"good (<= {good_pos_error_threshold_m:.2f} m)",
            edgecolors="white",
            linewidths=0.4,
        )
    if np.any(bad_arr):
        ax.scatter(
            u_arr[bad_arr],
            v_arr[bad_arr],
            c="#ff2d2d",
            s=36,
            alpha=0.9,
            label=f"bad (> {good_pos_error_threshold_m:.2f} m)",
            edgecolors="white",
            linewidths=0.4,
        )
    if np.any(fail_arr):
        ax.scatter(
            u_arr[fail_arr],
            v_arr[fail_arr],
            c="black",
            marker="x",
            s=42,
            alpha=0.9,
            label="timeout/trigger_failed",
        )
    if np.any(ok_no_ref_arr):
        ax.scatter(
            u_arr[ok_no_ref_arr],
            v_arr[ok_no_ref_arr],
            c="#ff9800",
            s=36,
            alpha=0.9,
            label="localized (no reference)",
            edgecolors="white",
            linewidths=0.4,
        )
    ax.set_title("Global localization quality points")
    ax.set_xlim(0, map_meta.image_width - 1)
    ax.set_ylim(map_meta.image_height - 1, 0)
    ax.set_xlabel("u [pixel]")
    ax.set_ylabel("v [pixel]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(points_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    bin_px = max(1, int(round(grid_size_m / map_meta.resolution)))
    bins_x = int(math.ceil(map_meta.image_width / bin_px))
    bins_y = int(math.ceil(map_meta.image_height / bin_px))

    total = np.zeros((bins_y, bins_x), dtype=np.int32)
    good = np.zeros((bins_y, bins_x), dtype=np.int32)
    localized = np.zeros((bins_y, bins_x), dtype=np.int32)

    for u, v, g, b, ok_no_ref in zip(u_arr, v_arr, good_arr, bad_arr, ok_no_ref_arr):
        if not np.isfinite(u) or not np.isfinite(v):
            continue
        if u < 0.0 or v < 0.0 or u >= map_meta.image_width or v >= map_meta.image_height:
            continue
        ix = min(bins_x - 1, max(0, int(u // bin_px)))
        iy = min(bins_y - 1, max(0, int(v // bin_px)))
        if g or b:
            total[iy, ix] += 1
        if g:
            good[iy, ix] += 1
        if g or b or ok_no_ref:
            localized[iy, ix] += 1

    rate = np.full((bins_y, bins_x), np.nan, dtype=np.float32)
    has_reference_quality = bool(np.any(good_arr) or np.any(bad_arr))
    if has_reference_quality:
        valid = total >= max(1, min_samples_per_cell)
        rate[valid] = good[valid].astype(np.float32) / total[valid].astype(np.float32)
    else:
        valid = localized >= max(1, min_samples_per_cell)
        if np.any(valid):
            max_count = int(localized[valid].max())
            denom = float(max(1, max_count))
            rate[valid] = localized[valid].astype(np.float32) / denom

    heatmap = np.full(
        (map_meta.image_height, map_meta.image_width), np.nan, dtype=np.float32
    )
    for iy in range(bins_y):
        y0 = iy * bin_px
        y1 = min(map_meta.image_height, (iy + 1) * bin_px)
        for ix in range(bins_x):
            x0 = ix * bin_px
            x1 = min(map_meta.image_width, (ix + 1) * bin_px)
            heatmap[y0:y1, x0:x1] = rate[iy, ix]

    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad((0.0, 0.0, 0.0, 0.0))
    masked_heatmap = np.ma.masked_invalid(heatmap)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(map_img, origin="upper")
    im = ax.imshow(
        masked_heatmap,
        origin="upper",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        alpha=0.68,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if has_reference_quality:
        cbar.set_label("good match rate (0=bad, 1=good)")
        ax.set_title(
            f"Global localization success-rate heatmap (grid={grid_size_m:.2f} m, min_n={max(1, min_samples_per_cell)})"
        )
    else:
        cbar.set_label("relative localized sample density")
        ax.set_title(
            f"Global localization localized-density heatmap (no reference, grid={grid_size_m:.2f} m, min_n={max(1, min_samples_per_cell)})"
        )
    ax.set_xlim(0, map_meta.image_width - 1)
    ax.set_ylim(map_meta.image_height - 1, 0)
    ax.set_xlabel("u [pixel]")
    ax.set_ylabel("v [pixel]")
    fig.tight_layout()
    fig.savefig(rate_output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rate_output_path, points_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render localization quality overlays from evaluation CSV."
    )
    parser.add_argument("--eval-csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--map-yaml", required=True, help="Path to 2D map yaml")
    parser.add_argument("--good-pos-error-threshold-m", type=float, default=0.5)
    parser.add_argument("--quality-grid-size-m", type=float, default=1.0)
    parser.add_argument("--quality-min-samples-per-cell", type=int, default=1)
    parser.add_argument("--quality-rate-output", default="")
    parser.add_argument("--quality-points-output", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    eval_csv = Path(args.eval_csv).expanduser().resolve()
    map_yaml = Path(args.map_yaml).expanduser().resolve()
    if not eval_csv.exists():
        raise FileNotFoundError(f"evaluation csv not found: {eval_csv}")
    if not map_yaml.exists():
        raise FileNotFoundError(f"map yaml not found: {map_yaml}")

    rate_path, points_path = render_map_quality_images(
        csv_path=eval_csv,
        map_yaml_path=map_yaml,
        good_pos_error_threshold_m=args.good_pos_error_threshold_m,
        grid_size_m=args.quality_grid_size_m,
        min_samples_per_cell=args.quality_min_samples_per_cell,
        rate_output_path=(
            Path(args.quality_rate_output).expanduser().resolve()
            if args.quality_rate_output
            else None
        ),
        points_output_path=(
            Path(args.quality_points_output).expanduser().resolve()
            if args.quality_points_output
            else None
        ),
    )
    print(f"[INFO] Saved quality points plot: {points_path}")
    print(f"[INFO] Saved success-rate heatmap: {rate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

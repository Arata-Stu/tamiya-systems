#!/usr/bin/env python3
"""
Robust centerline extraction from PNG/PGM occupancy maps.

Dependency target:
- numpy
- opencv-python
- omegaconf

This script avoids scipy/skimage/Pillow/PyYAML so it can run in leaner ML
environments while keeping robust behavior for noisy maps (e.g. Cartographer).
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf


DIR8: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def parse_quantiles(text: str) -> List[float]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        q = float(token)
        if q <= 0.0 or q >= 1.0:
            raise ValueError(f"Quantile must be in (0,1), got {q}")
        values.append(q)
    if not values:
        raise ValueError("At least one quantile is required.")
    return values


def read_map_grayscale(map_path: str) -> np.ndarray:
    gray = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read map image: {map_path}")
    # Keep compatibility with map_converter.ipynb coordinate convention.
    return np.flipud(gray.astype(np.float32))


def gray_to_black(gray: np.ndarray, white_threshold: float) -> np.ndarray:
    if white_threshold <= 0.0:
        return gray
    out = gray.copy()
    out[out < white_threshold] = 0.0
    return out


def ellipse_kernel(radius: int) -> np.ndarray:
    if radius <= 0:
        return np.ones((1, 1), dtype=np.uint8)
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def remove_small_objects(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 1:
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= min_area:
            out[labels == i] = 1
    return out


def remove_small_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    if max_hole_area <= 0:
        return mask

    h, w = mask.shape
    inv = (1 - mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    out = mask.copy().astype(np.uint8)

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)
        if (not touches_border) and area <= max_hole_area:
            out[labels == i] = 1

    return out


def choose_free_space_mask(
    gray: np.ndarray,
    min_free_intensity: float,
    gaussian_sigma: float,
) -> Tuple[np.ndarray, float]:
    if gaussian_sigma > 0.0:
        blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    else:
        blurred = gray.copy()

    blurred_u8 = np.clip(blurred, 0, 255).astype(np.uint8)
    otsu_thr, _ = cv2.threshold(blurred_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    p85 = float(np.percentile(blurred, 85.0))

    candidates = [
        max(float(min_free_intensity), float(otsu_thr)),
        float(min_free_intensity),
        float(otsu_thr),
        p85,
    ]

    # Preserve order while removing duplicates.
    seen = set()
    uniq = []
    for c in candidates:
        key = round(c, 3)
        if key not in seen:
            uniq.append(c)
            seen.add(key)

    target_ratio = 0.18
    best_score = float("inf")
    best_mask = None
    best_thr = uniq[0]

    for thr in uniq:
        mask = (blurred >= thr).astype(np.uint8)
        ratio = float(mask.mean())
        score = abs(ratio - target_ratio)
        if ratio < 0.005 or ratio > 0.85:
            score += 2.0
        if ratio < 0.02 or ratio > 0.65:
            score += 0.5

        if score < best_score:
            best_score = score
            best_mask = mask
            best_thr = thr

    if best_mask is None:
        raise RuntimeError("Failed to generate free-space mask.")
    return best_mask, best_thr


def clean_mask(
    mask: np.ndarray,
    close_radius: int,
    open_radius: int,
    min_object_area: int,
    max_small_hole_area: int,
) -> np.ndarray:
    out = mask.astype(np.uint8)

    if close_radius > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ellipse_kernel(close_radius))
    if open_radius > 0:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ellipse_kernel(open_radius))

    out = remove_small_objects(out, min_object_area)
    out = remove_small_holes(out, max_small_hole_area)
    return out.astype(np.uint8)


def select_track_component(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        raise RuntimeError("No connected component found in free-space mask.")

    h, w = mask.shape
    best_non_border = -1
    best_non_border_area = -1
    best_any = -1
    best_any_area = -1

    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        touches_border = (x == 0) or (y == 0) or (x + ww >= w) or (y + hh >= h)

        if area > best_any_area:
            best_any_area = area
            best_any = i

        if (not touches_border) and area > best_non_border_area:
            best_non_border_area = area
            best_non_border = i

    chosen = best_non_border if best_non_border >= 0 else best_any
    if chosen < 0:
        raise RuntimeError("Unable to select track component.")

    return (labels == chosen).astype(np.uint8)


def filter_binary_track_mask(
    gray: np.ndarray,
    min_free_intensity: float,
    gaussian_sigma: float,
    morph_close_radius: int,
    morph_open_radius: int,
    min_track_area: int,
    max_small_hole_area: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if gaussian_sigma > 0.0:
        gray_work = cv2.GaussianBlur(gray, (0, 0), sigmaX=gaussian_sigma, sigmaY=gaussian_sigma)
    else:
        gray_work = gray

    free_mask = (gray_work >= float(min_free_intensity)).astype(np.uint8)

    if morph_close_radius > 0:
        free_mask = cv2.morphologyEx(
            free_mask,
            cv2.MORPH_CLOSE,
            ellipse_kernel(morph_close_radius),
            iterations=1,
        )

    if morph_open_radius > 0:
        free_mask = cv2.morphologyEx(
            free_mask,
            cv2.MORPH_OPEN,
            ellipse_kernel(morph_open_radius),
            iterations=2,
        )

    free_mask = remove_small_objects(free_mask, min_track_area)
    free_mask = remove_small_holes(free_mask, max_small_hole_area)
    track_mask = select_track_component(free_mask)
    return free_mask, track_mask


def distance_transform(mask: np.ndarray) -> np.ndarray:
    # OpenCV expects foreground as non-zero.
    src = (mask > 0).astype(np.uint8) * 255
    dist = cv2.distanceTransform(src, distanceType=cv2.DIST_L2, maskSize=5)
    return dist.astype(np.float32)


def neighbors_and_counts(img01: np.ndarray):
    padded = np.pad(img01, ((1, 1), (1, 1)), mode="constant", constant_values=0)

    p2 = padded[0:-2, 1:-1]
    p3 = padded[0:-2, 2:]
    p4 = padded[1:-1, 2:]
    p5 = padded[2:, 2:]
    p6 = padded[2:, 1:-1]
    p7 = padded[2:, 0:-2]
    p8 = padded[1:-1, 0:-2]
    p9 = padded[0:-2, 0:-2]

    neighbors = (p2, p3, p4, p5, p6, p7, p8, p9)
    n = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9

    # Number of 0->1 transitions in ordered loop P2...P9,P2.
    s = (
        ((p2 == 0) & (p3 == 1)).astype(np.uint8)
        + ((p3 == 0) & (p4 == 1)).astype(np.uint8)
        + ((p4 == 0) & (p5 == 1)).astype(np.uint8)
        + ((p5 == 0) & (p6 == 1)).astype(np.uint8)
        + ((p6 == 0) & (p7 == 1)).astype(np.uint8)
        + ((p7 == 0) & (p8 == 1)).astype(np.uint8)
        + ((p8 == 0) & (p9 == 1)).astype(np.uint8)
        + ((p9 == 0) & (p2 == 1)).astype(np.uint8)
    )
    return neighbors, n, s


def zhang_suen_thinning(mask: np.ndarray, max_iters: int = 200) -> np.ndarray:
    img = (mask > 0).astype(np.uint8)
    if img.size == 0:
        return img

    for _ in range(max_iters):
        changed = False

        neighbors, n, s = neighbors_and_counts(img)
        p2, _, p4, _, p6, _, p8, _ = neighbors

        m1 = (
            (img == 1)
            & (n >= 2)
            & (n <= 6)
            & (s == 1)
            & ((p2 * p4 * p6) == 0)
            & ((p4 * p6 * p8) == 0)
        )
        if np.any(m1):
            img[m1] = 0
            changed = True

        neighbors, n, s = neighbors_and_counts(img)
        p2, _, p4, _, p6, _, p8, _ = neighbors

        m2 = (
            (img == 1)
            & (n >= 2)
            & (n <= 6)
            & (s == 1)
            & ((p2 * p4 * p8) == 0)
            & ((p2 * p6 * p8) == 0)
        )
        if np.any(m2):
            img[m2] = 0
            changed = True

        if not changed:
            break

    return img.astype(np.uint8)


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    count = cv2.filter2D(binary.astype(np.uint8), ddepth=cv2.CV_16U, kernel=k, borderType=cv2.BORDER_CONSTANT)
    return count.astype(np.int32) - binary.astype(np.int32)


def prune_spurs(skel: np.ndarray, iterations: int) -> np.ndarray:
    out = (skel > 0).astype(np.uint8)
    for _ in range(max(0, iterations)):
        deg = neighbor_count(out)
        endpoints = (out == 1) & (deg <= 1)
        if not np.any(endpoints):
            break
        out[endpoints] = 0
    return out.astype(np.uint8)


def extract_centerline_contour(
    skeleton: np.ndarray,
    resolution: float,
    expected_length_m: float,
    allow_any_length: bool,
) -> np.ndarray:
    contours, hierarchy = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None or len(contours) == 0:
        raise RuntimeError("No contours found in skeleton.")

    closed_contours = []
    for i, cont in enumerate(contours):
        opened = hierarchy[0][i][2] < 0 and hierarchy[0][i][3] < 0
        if not opened:
            closed_contours.append(cont)

    if not closed_contours:
        raise RuntimeError("No closed contours found in skeleton.")

    line_lengths = [float("inf")] * len(closed_contours)
    for i, cont in enumerate(closed_contours):
        pts = cont.reshape(-1, 2).astype(np.float64)
        if len(pts) < 4:
            continue

        diffs = pts - np.roll(pts, 1, axis=0)
        line_length = float(np.linalg.norm(diffs, axis=1).sum()) * resolution

        if allow_any_length or expected_length_m <= 0.0:
            line_lengths[i] = line_length
        elif abs(expected_length_m / line_length - 1.0) < 0.15:
            line_lengths[i] = line_length

    min_length = min(line_lengths)
    if not np.isfinite(min_length):
        raise RuntimeError("Closed contours found, but none matched expected centerline length.")

    return closed_contours[line_lengths.index(min_length)].reshape(-1, 2).astype(np.float64)


def adjacency_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    points = [(int(x), int(y)) for y, x in zip(ys, xs)]
    point_set = set(points)
    adj = {}
    for p in points:
        px, py = p
        nbs = []
        for dx, dy in DIR8:
            q = (px + dx, py + dy)
            if q in point_set:
                nbs.append(q)
        adj[p] = nbs
    return points, adj


def choose_next_neighbor(
    current: Tuple[int, int],
    prev: Optional[Tuple[int, int]],
    neighbors: Sequence[Tuple[int, int]],
    dist: np.ndarray,
) -> Tuple[int, int]:
    if len(neighbors) == 1:
        return neighbors[0]

    best_nb = neighbors[0]
    best_score = -1e18
    for nb in neighbors:
        score = 0.05 * float(dist[nb[1], nb[0]])
        if prev is not None:
            vx, vy = current[0] - prev[0], current[1] - prev[1]
            wx, wy = nb[0] - current[0], nb[1] - current[1]
            score += float(vx * wx + vy * wy)
        if score > best_score:
            best_score = score
            best_nb = nb
    return best_nb


def order_closed_centerline(points: List[Tuple[int, int]], adj: dict, dist: np.ndarray) -> np.ndarray:
    start = max(points, key=lambda p: float(dist[p[1], p[0]]))
    first_neighbors = adj[start]
    if not first_neighbors:
        raise RuntimeError("Closed centerline candidate has no neighbors.")

    first = choose_next_neighbor(start, None, first_neighbors, dist)
    ordered = [start, first]
    prev = start
    current = first

    for _ in range(len(points) + 5):
        candidates = [nb for nb in adj[current] if nb != prev]
        if not candidates:
            break
        next_p = choose_next_neighbor(current, prev, candidates, dist)
        if next_p == start:
            break
        ordered.append(next_p)
        prev, current = current, next_p

    if len(ordered) < max(8, len(points) // 3):
        raise RuntimeError("Failed to trace a stable closed centerline loop.")
    return np.asarray(ordered, dtype=np.float64)


def order_open_centerline(points: List[Tuple[int, int]], adj: dict, dist: np.ndarray) -> np.ndarray:
    endpoints = [p for p in points if len(adj[p]) == 1]
    start = endpoints[0] if endpoints else max(points, key=lambda p: float(dist[p[1], p[0]]))

    visited_edges = set()
    ordered = [start]
    prev = None
    current = start

    max_steps = len(points) * 2 + 10
    for _ in range(max_steps):
        candidates = []
        for nb in adj[current]:
            edge = tuple(sorted((current, nb)))
            if edge in visited_edges:
                continue
            candidates.append((choose_next_neighbor(current, prev, [nb], dist), nb, edge))

        if not candidates:
            break

        scored = []
        for _, nb, edge in candidates:
            score = 0.05 * float(dist[nb[1], nb[0]])
            if prev is not None:
                vx, vy = current[0] - prev[0], current[1] - prev[1]
                wx, wy = nb[0] - current[0], nb[1] - current[1]
                score += float(vx * wx + vy * wy)
            scored.append((score, nb, edge))

        scored.sort(key=lambda x: x[0], reverse=True)
        _, next_p, edge = scored[0]
        visited_edges.add(edge)
        prev, current = current, next_p
        ordered.append(current)

    return np.asarray(ordered, dtype=np.float64)


def order_centerline_points(mask: np.ndarray, dist: np.ndarray) -> np.ndarray:
    points, adj = adjacency_from_mask(mask)
    if not points:
        raise RuntimeError("Centerline mask has no points.")

    endpoints = [p for p in points if len(adj[p]) == 1]
    branchpoints = [p for p in points if len(adj[p]) > 2]
    if not endpoints and not branchpoints:
        return order_closed_centerline(points, adj, dist)
    return order_open_centerline(points, adj, dist)


def circular_gaussian_smooth(values: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    if sigma <= 0.0 or len(values) < 5:
        return values

    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    if closed:
        padded = np.concatenate([values[-radius:], values, values[:radius]])
        smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
    else:
        padded = np.pad(values, (radius, radius), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="same")[radius:-radius]
    return smoothed


def smooth_points(points: np.ndarray, sigma: float, closed: bool) -> np.ndarray:
    if sigma <= 0.0 or len(points) < 5:
        return points
    xs = circular_gaussian_smooth(points[:, 0], sigma=sigma, closed=closed)
    ys = circular_gaussian_smooth(points[:, 1], sigma=sigma, closed=closed)
    return np.stack([xs, ys], axis=1)


def resample_polyline(points: np.ndarray, spacing: float, closed: bool) -> np.ndarray:
    if len(points) < 2 or spacing <= 0.0:
        return points

    pts = points.copy()
    if closed and np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[0]])

    seg = np.diff(pts, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    keep = seg_len > 1e-9
    if not np.any(keep):
        return points

    filtered = [pts[0]]
    cumulative = [0.0]
    total = 0.0
    for i, length in enumerate(seg_len):
        if length <= 1e-9:
            continue
        total += float(length)
        filtered.append(pts[i + 1])
        cumulative.append(total)

    if total < spacing:
        return np.asarray(filtered, dtype=np.float64)

    filtered = np.asarray(filtered, dtype=np.float64)
    cumulative = np.asarray(cumulative, dtype=np.float64)

    n_samples = max(int(total / spacing), 16)
    targets = np.linspace(0.0, total, n_samples, endpoint=not closed)

    xs = np.interp(targets, cumulative, filtered[:, 0])
    ys = np.interp(targets, cumulative, filtered[:, 1])
    return np.stack([xs, ys], axis=1)


def infer_yaml_path(image_path: str) -> Optional[str]:
    root, _ = os.path.splitext(image_path)
    candidate = root + ".yaml"
    return candidate if os.path.exists(candidate) else None


def read_yaml_metadata(yaml_path: str) -> Tuple[float, float, float]:
    cfg = OmegaConf.load(yaml_path)
    if "resolution" not in cfg or "origin" not in cfg:
        raise RuntimeError(f"YAML missing required keys: {yaml_path}")
    resolution = float(cfg["resolution"])
    origin = cfg["origin"]
    if len(origin) < 2:
        raise RuntimeError(f"YAML origin must have at least [x, y]: {yaml_path}")
    return resolution, float(origin[0]), float(origin[1])


def save_debug_images(
    debug_dir: str,
    gray: np.ndarray,
    free_mask: np.ndarray,
    track_mask: np.ndarray,
    dist: np.ndarray,
    centerline_mask: np.ndarray,
) -> None:
    os.makedirs(debug_dir, exist_ok=True)

    cv2.imwrite(os.path.join(debug_dir, "01_gray.png"), np.clip(gray, 0, 255).astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, "02_free_mask.png"), (free_mask * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(debug_dir, "03_track_mask.png"), (track_mask * 255).astype(np.uint8))

    if float(dist.max()) > 0.0:
        dist_img = np.clip((dist / dist.max()) * 255.0, 0, 255).astype(np.uint8)
    else:
        dist_img = np.zeros_like(dist, dtype=np.uint8)
    cv2.imwrite(os.path.join(debug_dir, "04_dist.png"), dist_img)

    cv2.imwrite(
        os.path.join(debug_dir, "05_centerline_mask.png"),
        (centerline_mask.astype(np.uint8) * 255),
    )


def run(args: argparse.Namespace) -> None:
    gray = read_map_grayscale(args.map)
    gray = gray_to_black(gray, args.gray_to_black_white_threshold)

    yaml_path = args.yaml
    if yaml_path == "auto":
        yaml_path = infer_yaml_path(args.map)

    if yaml_path is not None and os.path.exists(yaml_path):
        resolution, origin_x, origin_y = read_yaml_metadata(yaml_path)
        yaml_used = True
    else:
        resolution, origin_x, origin_y = 1.0, 0.0, 0.0
        yaml_used = False

    free_mask, track_mask = filter_binary_track_mask(
        gray=gray,
        min_free_intensity=args.min_free_intensity,
        gaussian_sigma=args.gaussian_sigma,
        morph_close_radius=args.close_radius,
        morph_open_radius=args.open_radius,
        min_track_area=args.min_track_area,
        max_small_hole_area=args.max_small_hole_area,
    )
    dist = distance_transform(track_mask)

    skeleton = zhang_suen_thinning(track_mask)
    skeleton = prune_spurs(skeleton, args.prune_iters)

    points_px = extract_centerline_contour(
        skeleton=skeleton,
        resolution=resolution,
        expected_length_m=args.expected_centerline_length_m,
        allow_any_length=args.allow_any_length or (not yaml_used),
    )
    centerline_mask = np.zeros_like(track_mask, dtype=np.uint8)
    if len(points_px) > 0:
        cv2.drawContours(centerline_mask, [points_px.astype(np.int32)], 0, 1, 1, cv2.LINE_8)

    is_closed = True
    points_px = smooth_points(points_px, sigma=args.smooth_sigma, closed=is_closed)
    points_px = resample_polyline(points_px, spacing=args.spacing_px, closed=is_closed)

    # Bilinear sampling via OpenCV remap.
    x_map = points_px[:, 0].astype(np.float32).reshape(-1, 1)
    y_map = points_px[:, 1].astype(np.float32).reshape(-1, 1)
    sampled = cv2.remap(dist, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    widths_px = sampled.reshape(-1).astype(np.float64)

    if yaml_used:
        points = points_px * resolution + np.array([origin_x, origin_y], dtype=np.float64)
        margin_px = args.track_width_margin_m / resolution if resolution > 0 else 0.0
        widths = np.maximum(widths_px - margin_px, 0.0) * resolution
        header = "x_m,y_m,w_tr_right_m,w_tr_left_m"
    else:
        points = points_px
        widths = np.maximum(widths_px - args.track_width_margin_px, 0.0)
        header = "x_px,y_px,w_tr_right_px,w_tr_left_px"

    out = np.column_stack([points[:, 0], points[:, 1], widths, widths])

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as fh:
        np.savetxt(fh, out, fmt="%.4f", delimiter=",", header=header)

    if args.debug_dir:
        save_debug_images(
            debug_dir=args.debug_dir,
            gray=gray,
            free_mask=free_mask,
            track_mask=track_mask,
            dist=dist,
            centerline_mask=centerline_mask,
        )

    print(f"Input map: {args.map}")
    print(f"Track pixels: {int(track_mask.sum())}")
    print(f"Skeleton pixels: {int(skeleton.sum())}")
    print(f"Centerline pixels: {int(centerline_mask.sum())}")
    print(f"Output points: {len(out)}")
    print(f"Wrote centerline CSV: {out_path}")
    if yaml_used:
        print(f"Used YAML metadata: {yaml_path}")
    else:
        print("No YAML metadata used. Output is in pixel coordinates.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Robust centerline extraction from PNG/PGM map.")
    p.add_argument("--map", required=True, help="Path to input .png or .pgm map image")
    p.add_argument("--output", required=True, help="Path to output centerline CSV")
    p.add_argument(
        "--yaml",
        default="auto",
        help="Map YAML path. Use 'auto' to infer from map path, or non-existing path to disable.",
    )

    p.add_argument("--min-free-intensity", type=float, default=210.0)
    p.add_argument(
        "--gray-to-black-white-threshold",
        type=float,
        default=250.0,
        help=(
            "Before centerline extraction, convert pixels below this intensity to black. "
            "Set <=0 to disable."
        ),
    )
    p.add_argument("--gaussian-sigma", type=float, default=1.0)
    p.add_argument("--close-radius", type=int, default=2)
    p.add_argument("--open-radius", type=int, default=1)
    p.add_argument("--min-track-area", type=int, default=800)
    p.add_argument("--max-small-hole-area", type=int, default=128)
    p.add_argument("--prune-iters", type=int, default=30)
    p.add_argument("--smooth-sigma", type=float, default=1.2)
    p.add_argument("--spacing-px", type=float, default=1.5)
    p.add_argument(
        "--expected-centerline-length-m",
        type=float,
        default=0.0,
        help="Optional expected lap length in meters. When >0, closed contours are filtered by +/-15%% like race_stack.",
    )
    p.add_argument(
        "--allow-any-length",
        action="store_true",
        help="Ignore expected length filtering and choose the shortest closed contour.",
    )

    p.add_argument("--track-width-margin-m", type=float, default=0.0)
    p.add_argument("--track-width-margin-px", type=float, default=0.0)
    p.add_argument("--debug-dir", default=None, help="Optional directory to save debug images")

    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

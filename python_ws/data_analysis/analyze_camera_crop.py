#!/usr/bin/env python3
"""rosbag内の画像特徴分布から camera crop の許容量を見積もる。"""

from __future__ import annotations

import argparse
import csv
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    from rosbags.highlevel import AnyReader
except ImportError:
    AnyReader = object

try:
    import cv2
except ImportError:
    cv2 = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze image feature distributions in rosbag and estimate safe crop ratios "
            "for VSLAM camera_crop / image masking decisions."
        )
    )
    parser.add_argument("--bag", required=True, help="Path to rosbag2 directory, metadata.yaml, or rosbag1 .bag")
    parser.add_argument(
        "--topics",
        nargs="+",
        required=True,
        help="Image topics to analyze. Multiple topics can be specified for stereo/multi-camera bags.",
    )
    parser.add_argument(
        "--max_frames_per_topic",
        type=int,
        default=300,
        help="Maximum number of frames to analyze per topic. Use 0 for all frames.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=5,
        help="Analyze one frame every N frames per topic.",
    )
    parser.add_argument(
        "--feature_method",
        choices=["gftt", "orb"],
        default="gftt",
        help="Feature detector to use. gftt is stable for border-density analysis.",
    )
    parser.add_argument("--max_features", type=int, default=600, help="Maximum features per frame")
    parser.add_argument("--quality_level", type=float, default=0.01, help="goodFeaturesToTrack qualityLevel")
    parser.add_argument("--min_distance", type=float, default=8.0, help="goodFeaturesToTrack minDistance in pixels")
    parser.add_argument(
        "--retained_feature_ratio",
        type=float,
        default=0.95,
        help="Required retained-feature ratio for recommendation (e.g. 0.95 = keep at least 95%%).",
    )
    parser.add_argument(
        "--frame_quantile",
        type=float,
        default=0.10,
        help=(
            "Per-frame retained ratio quantile used for recommendation. "
            "0.10 means the 10th percentile frame must still satisfy retained_feature_ratio."
        ),
    )
    parser.add_argument(
        "--ratio_step",
        type=float,
        default=0.01,
        help="Crop-ratio sweep step for recommendation and CSV export.",
    )
    parser.add_argument(
        "--heatmap_bins",
        type=int,
        default=48,
        help="2D histogram bins per axis for feature heatmap.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to save CSV/PNG outputs. Defaults to /tmp/camera_crop_analysis_<bagname>.",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip PNG plot generation")
    return parser.parse_args()


def normalize_bag_path(path_str: str) -> Path:
    bag_path = Path(path_str).expanduser().resolve()
    if bag_path.is_file() and bag_path.name == "metadata.yaml":
        return bag_path.parent
    return bag_path


def require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "opencv-python is required for camera crop analysis. "
            "Install with: pip install opencv-python"
        )
    return cv2


def require_numpy():
    if np is None:
        raise RuntimeError(
            "numpy is required for camera crop analysis. "
            "Install with: pip install numpy"
        )
    return np


def require_rosbags():
    if AnyReader is object:
        raise RuntimeError(
            "rosbags is required for camera crop analysis. "
            "Install with: pip install rosbags"
        )
    return AnyReader


def _build_default_ros2_typestore():
    try:
        from rosbags.typesys import Stores, get_typestore
    except Exception:
        return None

    preferred_store_names = (
        "ROS2_JAZZY",
        "ROS2_IRON",
        "ROS2_HUMBLE",
        "ROS2_GALACTIC",
        "ROS2_FOXY",
    )
    for store_name in preferred_store_names:
        if hasattr(Stores, store_name):
            return get_typestore(getattr(Stores, store_name))

    for store in Stores:
        if store.name.startswith("ROS2_"):
            return get_typestore(store)

    return None


def _open_reader(bag_path: Path) -> AnyReader:
    any_reader_cls = require_rosbags()
    reader_kwargs = {}
    default_typestore = _build_default_ros2_typestore()

    try:
        supports_default_typestore = "default_typestore" in inspect.signature(any_reader_cls).parameters
    except (TypeError, ValueError):
        supports_default_typestore = False

    if supports_default_typestore and default_typestore is not None:
        reader_kwargs["default_typestore"] = default_typestore

    return any_reader_cls([bag_path], **reader_kwargs)


def _decode_raw_image_to_gray(msg) -> Optional[np.ndarray]:
    cv = require_cv2()
    np_mod = require_numpy()
    height = int(msg.height)
    width = int(msg.width)
    if height <= 0 or width <= 0:
        return None

    encoding = msg.encoding.lower()
    step = int(msg.step)
    raw = np_mod.frombuffer(msg.data, dtype=np_mod.uint8)
    if step <= 0 or raw.size < height * step:
        return None

    raw = raw[: height * step].reshape(height, step)

    if encoding in ("mono8", "8uc1"):
        return raw[:, :width].copy()

    if encoding in ("rgb8",):
        rgb = raw[:, : width * 3].reshape(height, width, 3)
        return cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)

    if encoding in ("bgr8", "8uc3"):
        bgr = raw[:, : width * 3].reshape(height, width, 3)
        return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    if encoding in ("rgba8",):
        rgba = raw[:, : width * 4].reshape(height, width, 4)
        return cv.cvtColor(rgba, cv.COLOR_RGBA2GRAY)

    if encoding in ("bgra8",):
        bgra = raw[:, : width * 4].reshape(height, width, 4)
        return cv.cvtColor(bgra, cv.COLOR_BGRA2GRAY)

    if encoding in ("mono16", "16uc1", "16sc1"):
        raw16 = np_mod.frombuffer(msg.data, dtype=np_mod.uint16)
        if raw16.size < height * width:
            return None
        img16 = raw16[: height * width].reshape(height, width)
        return (img16 / 256.0).astype(np_mod.uint8)

    if encoding in ("yuyv", "yuyv422", "yuv422"):
        yuv = raw[:, : width * 2].reshape(height, width, 2)
        return cv.cvtColor(yuv, cv.COLOR_YUV2GRAY_YUY2)

    if encoding == "uyvy":
        yuv = raw[:, : width * 2].reshape(height, width, 2)
        return cv.cvtColor(yuv, cv.COLOR_YUV2GRAY_UYVY)

    return None


def _decode_compressed_image_to_gray(msg) -> Optional[np.ndarray]:
    cv = require_cv2()
    np_mod = require_numpy()
    buf = np_mod.frombuffer(msg.data, dtype=np_mod.uint8)
    return cv.imdecode(buf, cv.IMREAD_GRAYSCALE)


def _iter_topic_images(bag_path: Path, topic: str):
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag path does not exist: {bag_path}")

    try:
        with _open_reader(bag_path) as reader:
            connections = [
                c
                for c in reader.connections
                if c.topic == topic and c.msgtype in ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")
            ]
            if not connections:
                raise RuntimeError(f"Image topic not found: topic={topic} path={bag_path}")

            for frame_id, (conn, timestamp, raw) in enumerate(reader.messages(connections=connections)):
                msg = reader.deserialize(raw, conn.msgtype)
                image = None
                if conn.msgtype == "sensor_msgs/msg/Image":
                    image = _decode_raw_image_to_gray(msg)
                elif conn.msgtype == "sensor_msgs/msg/CompressedImage":
                    image = _decode_compressed_image_to_gray(msg)

                if image is not None:
                    yield frame_id, timestamp, image
    except Exception as exc:
        msg = str(exc)
        if "default_typestore" in msg and "no type definitions" in msg.lower():
            raise RuntimeError(
                "Bag contains no type definitions and could not load a default ROS2 typestore. "
                "Please update rosbags and ensure rosbags.typesys Stores/get_typestore are available."
            ) from exc
        raise


@dataclass
class TopicAnalysis:
    topic: str
    width: int
    height: int
    analyzed_frames: int
    feature_count_total: int
    per_frame_counts: np.ndarray
    xs_norm: np.ndarray
    ys_norm: np.ndarray
    preview_image: np.ndarray


def detect_feature_points(gray: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cv = require_cv2()
    np_mod = require_numpy()
    if args.feature_method == "gftt":
        corners = cv.goodFeaturesToTrack(
            gray,
            maxCorners=args.max_features,
            qualityLevel=args.quality_level,
            minDistance=args.min_distance,
            blockSize=7,
            useHarrisDetector=False,
        )
        if corners is None:
            return np_mod.empty((0, 2), dtype=np_mod.float32)
        return corners.reshape(-1, 2).astype(np_mod.float32)

    orb = cv.ORB_create(nfeatures=args.max_features)
    keypoints = orb.detect(gray, None)
    if not keypoints:
        return np_mod.empty((0, 2), dtype=np_mod.float32)
    return np_mod.array([kp.pt for kp in keypoints], dtype=np_mod.float32)


def collect_topic_analysis(bag_path: Path, topic: str, args: argparse.Namespace) -> TopicAnalysis:
    np_mod = require_numpy()
    xs_norm_list: list[np.ndarray] = []
    ys_norm_list: list[np.ndarray] = []
    per_frame_counts: list[int] = []
    preview_image = None
    width = 0
    height = 0
    analyzed = 0

    for frame_id, _timestamp, gray in _iter_topic_images(bag_path, topic):
        if args.frame_stride > 1 and frame_id % args.frame_stride != 0:
            continue
        if args.max_frames_per_topic > 0 and analyzed >= args.max_frames_per_topic:
            break

        if preview_image is None:
            preview_image = gray

        h, w = gray.shape[:2]
        if width == 0 and height == 0:
            width = w
            height = h
        elif width != w or height != h:
            raise RuntimeError(
                f"Image size changed within topic {topic}: expected {width}x{height}, got {w}x{h}"
            )

        points = detect_feature_points(gray, args)
        per_frame_counts.append(int(points.shape[0]))
        analyzed += 1

        if points.size == 0:
            continue

        xs_norm_list.append(points[:, 0] / max(1.0, float(w - 1)))
        ys_norm_list.append(points[:, 1] / max(1.0, float(h - 1)))

    if analyzed == 0:
        raise RuntimeError(f"No frames analyzed for topic {topic}. Check topic name or frame selection.")

    if preview_image is None:
        preview_image = np_mod.zeros((height, width), dtype=np_mod.uint8)

    xs_norm = np_mod.concatenate(xs_norm_list) if xs_norm_list else np_mod.empty((0,), dtype=np_mod.float32)
    ys_norm = np_mod.concatenate(ys_norm_list) if ys_norm_list else np_mod.empty((0,), dtype=np_mod.float32)

    return TopicAnalysis(
        topic=topic,
        width=width,
        height=height,
        analyzed_frames=analyzed,
        feature_count_total=int(xs_norm.shape[0]),
        per_frame_counts=np_mod.asarray(per_frame_counts, dtype=np_mod.int32),
        xs_norm=xs_norm.astype(np_mod.float32),
        ys_norm=ys_norm.astype(np_mod.float32),
        preview_image=preview_image,
    )


def _axis_mask(axis_values: np.ndarray, side: str, ratio: float) -> np.ndarray:
    if side == "top":
        return axis_values >= ratio
    if side == "bottom":
        return axis_values <= (1.0 - ratio)
    if side == "left":
        return axis_values >= ratio
    if side == "right":
        return axis_values <= (1.0 - ratio)
    raise ValueError(f"Unsupported side: {side}")


def compute_side_curve(points_norm: np.ndarray, ratios: np.ndarray, side: str) -> np.ndarray:
    np_mod = require_numpy()
    if points_norm.size == 0:
        return np_mod.zeros_like(ratios, dtype=np_mod.float64)
    values = []
    for ratio in ratios:
        kept = _axis_mask(points_norm, side, float(ratio))
        values.append(float(np.mean(kept)))
    return np_mod.asarray(values, dtype=np_mod.float64)


def compute_per_frame_ratios(axis_values_per_frame: list[np.ndarray], ratios: np.ndarray, side: str) -> np.ndarray:
    np_mod = require_numpy()
    result = np_mod.zeros((len(axis_values_per_frame), len(ratios)), dtype=np_mod.float64)
    for frame_idx, axis_values in enumerate(axis_values_per_frame):
        if axis_values.size == 0:
            continue
        for ratio_idx, ratio in enumerate(ratios):
            kept = _axis_mask(axis_values, side, float(ratio))
            result[frame_idx, ratio_idx] = float(np.mean(kept))
    return result


def find_recommended_ratio(
    per_frame_retained: np.ndarray,
    ratios: np.ndarray,
    required_retained_ratio: float,
    frame_quantile: float,
) -> float:
    np_mod = require_numpy()
    if per_frame_retained.size == 0:
        return 0.0

    quantiles = np_mod.quantile(per_frame_retained, frame_quantile, axis=0)
    ok = np_mod.where(quantiles >= required_retained_ratio)[0]
    if ok.size == 0:
        return 0.0
    return float(ratios[int(ok[-1])])


def compute_combined_retained_ratio(points_xy_norm: np.ndarray, crop_rect: dict[str, float]) -> float:
    if points_xy_norm.size == 0:
        return 0.0
    x = points_xy_norm[:, 0]
    y = points_xy_norm[:, 1]
    kept = (y >= crop_rect["top"]) & (y <= 1.0 - crop_rect["bottom"])
    kept &= (x >= crop_rect["left"]) & (x <= 1.0 - crop_rect["right"])
    return float(np.mean(kept))


def compute_combined_per_frame_ratios(
    points_per_frame: list[np.ndarray],
    crop_rect: dict[str, float],
) -> np.ndarray:
    values = []
    for points_xy_norm in points_per_frame:
        if points_xy_norm.size == 0:
            values.append(0.0)
            continue
        values.append(compute_combined_retained_ratio(points_xy_norm, crop_rect))
    return np.asarray(values, dtype=np.float64)


def export_csv(
    out_path: Path,
    ratios: np.ndarray,
    aggregate: np.ndarray,
    per_frame_retained: np.ndarray,
) -> None:
    np_mod = require_numpy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    quantile_10 = np_mod.quantile(per_frame_retained, 0.10, axis=0) if per_frame_retained.size else np_mod.zeros_like(ratios)
    quantile_50 = np_mod.quantile(per_frame_retained, 0.50, axis=0) if per_frame_retained.size else np_mod.zeros_like(ratios)
    quantile_90 = np_mod.quantile(per_frame_retained, 0.90, axis=0) if per_frame_retained.size else np_mod.zeros_like(ratios)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "crop_ratio",
                "aggregate_retained_ratio",
                "frame_retained_ratio_p10",
                "frame_retained_ratio_p50",
                "frame_retained_ratio_p90",
            ]
        )
        for idx, ratio in enumerate(ratios):
            writer.writerow(
                [
                    f"{ratio:.4f}",
                    f"{aggregate[idx]:.6f}",
                    f"{quantile_10[idx]:.6f}",
                    f"{quantile_50[idx]:.6f}",
                    f"{quantile_90[idx]:.6f}",
                ]
            )


def save_topic_plot(
    out_path: Path,
    analysis: TopicAnalysis,
    ratios: np.ndarray,
    side_curves: dict[str, np.ndarray],
    per_frame_curves: dict[str, np.ndarray],
    recommendations: dict[str, float],
    required_retained_ratio: float,
    frame_quantile: float,
    heatmap_bins: int,
) -> None:
    np_mod = require_numpy()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plot output. Install with: pip install matplotlib") from exc

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    ax_preview = axes[0, 0]
    ax_heatmap = axes[0, 1]
    ax_counts = axes[0, 2]
    ax_tb = axes[1, 0]
    ax_lr = axes[1, 1]
    ax_text = axes[1, 2]

    ax_preview.imshow(analysis.preview_image, cmap="gray")
    colors = {"top": "tab:red", "bottom": "tab:orange", "left": "tab:blue", "right": "tab:green"}
    if analysis.height > 0:
        top_px = recommendations["top"] * analysis.height
        bottom_px = analysis.height - recommendations["bottom"] * analysis.height
        ax_preview.axhline(top_px, color=colors["top"], linestyle="--", linewidth=2)
        ax_preview.axhline(bottom_px, color=colors["bottom"], linestyle="--", linewidth=2)
    if analysis.width > 0:
        left_px = recommendations["left"] * analysis.width
        right_px = analysis.width - recommendations["right"] * analysis.width
        ax_preview.axvline(left_px, color=colors["left"], linestyle="--", linewidth=2)
        ax_preview.axvline(right_px, color=colors["right"], linestyle="--", linewidth=2)
    ax_preview.set_title("Preview + recommended crop lines")
    ax_preview.set_axis_off()

    if analysis.feature_count_total > 0:
        heatmap, _, _ = np_mod.histogram2d(
            analysis.ys_norm,
            analysis.xs_norm,
            bins=heatmap_bins,
            range=[[0.0, 1.0], [0.0, 1.0]],
        )
        im = ax_heatmap.imshow(heatmap, cmap="hot", origin="upper")
        fig.colorbar(im, ax=ax_heatmap, fraction=0.046, pad=0.04)
    ax_heatmap.set_title("Feature density heatmap")
    ax_heatmap.set_xlabel("x / width")
    ax_heatmap.set_ylabel("y / height")

    ax_counts.plot(analysis.per_frame_counts, color="tab:purple", linewidth=1.0)
    ax_counts.set_title("Detected features per frame")
    ax_counts.set_xlabel("Analyzed frame index")
    ax_counts.set_ylabel("Feature count")
    ax_counts.grid(alpha=0.3)

    for side in ("top", "bottom"):
        ax_tb.plot(ratios, side_curves[side], label=f"{side} aggregate", color=colors[side], linewidth=2)
        quantile_curve = np_mod.quantile(per_frame_curves[side], frame_quantile, axis=0)
        ax_tb.plot(ratios, quantile_curve, linestyle=":", color=colors[side], linewidth=2, label=f"{side} q{frame_quantile:.2f}")
    ax_tb.axhline(required_retained_ratio, color="black", linestyle="--", linewidth=1)
    ax_tb.set_title("Top/Bottom crop retained-feature ratio")
    ax_tb.set_xlabel("crop ratio")
    ax_tb.set_ylabel("retained ratio")
    ax_tb.set_ylim(0.0, 1.02)
    ax_tb.grid(alpha=0.3)
    ax_tb.legend()

    for side in ("left", "right"):
        ax_lr.plot(ratios, side_curves[side], label=f"{side} aggregate", color=colors[side], linewidth=2)
        quantile_curve = np_mod.quantile(per_frame_curves[side], frame_quantile, axis=0)
        ax_lr.plot(ratios, quantile_curve, linestyle=":", color=colors[side], linewidth=2, label=f"{side} q{frame_quantile:.2f}")
    ax_lr.axhline(required_retained_ratio, color="black", linestyle="--", linewidth=1)
    ax_lr.set_title("Left/Right crop retained-feature ratio")
    ax_lr.set_xlabel("crop ratio")
    ax_lr.set_ylabel("retained ratio")
    ax_lr.set_ylim(0.0, 1.02)
    ax_lr.grid(alpha=0.3)
    ax_lr.legend()

    ax_text.axis("off")
    summary_lines = [
        f"topic: {analysis.topic}",
        f"frames: {analysis.analyzed_frames}",
        f"resolution: {analysis.width} x {analysis.height}",
        f"total features: {analysis.feature_count_total}",
        f"recommended top: {recommendations['top']:.3f} ({recommendations['top'] * analysis.height:.1f} px)",
        f"recommended bottom: {recommendations['bottom']:.3f} ({recommendations['bottom'] * analysis.height:.1f} px)",
        f"recommended left: {recommendations['left']:.3f} ({recommendations['left'] * analysis.width:.1f} px)",
        f"recommended right: {recommendations['right']:.3f} ({recommendations['right'] * analysis.width:.1f} px)",
        f"constraint: q{frame_quantile:.2f} >= retained {required_retained_ratio:.2f}",
    ]
    ax_text.text(0.0, 1.0, "\n".join(summary_lines), va="top", ha="left", family="monospace")

    fig.suptitle(f"Camera Crop Analysis: {analysis.topic}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def analyze_single_topic(
    bag_path: Path,
    topic: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, object]:
    np_mod = require_numpy()
    analysis = collect_topic_analysis(bag_path, topic, args)
    ratios = np_mod.arange(0.0, 0.5001, args.ratio_step, dtype=np_mod.float64)

    axis_values_per_frame = {"top": [], "bottom": [], "left": [], "right": []}
    points_per_frame: list[np.ndarray] = []
    for frame_id, _timestamp, gray in _iter_topic_images(bag_path, topic):
        if args.frame_stride > 1 and frame_id % args.frame_stride != 0:
            continue
        if len(axis_values_per_frame["top"]) >= analysis.analyzed_frames:
            break
        points = detect_feature_points(gray, args)
        if points.size == 0:
            empty = np_mod.empty((0,), dtype=np_mod.float32)
            axis_values_per_frame["top"].append(empty)
            axis_values_per_frame["bottom"].append(empty)
            axis_values_per_frame["left"].append(empty)
            axis_values_per_frame["right"].append(empty)
            points_per_frame.append(np_mod.empty((0, 2), dtype=np_mod.float32))
            continue
        h, w = gray.shape[:2]
        xs_norm = points[:, 0] / max(1.0, float(w - 1))
        ys_norm = points[:, 1] / max(1.0, float(h - 1))
        axis_values_per_frame["top"].append(ys_norm)
        axis_values_per_frame["bottom"].append(ys_norm)
        axis_values_per_frame["left"].append(xs_norm)
        axis_values_per_frame["right"].append(xs_norm)
        points_per_frame.append(np_mod.stack([xs_norm, ys_norm], axis=1))

    side_curves = {
        "top": compute_side_curve(analysis.ys_norm, ratios, "top"),
        "bottom": compute_side_curve(analysis.ys_norm, ratios, "bottom"),
        "left": compute_side_curve(analysis.xs_norm, ratios, "left"),
        "right": compute_side_curve(analysis.xs_norm, ratios, "right"),
    }
    per_frame_curves = {
        "top": compute_per_frame_ratios(axis_values_per_frame["top"], ratios, "top"),
        "bottom": compute_per_frame_ratios(axis_values_per_frame["bottom"], ratios, "bottom"),
        "left": compute_per_frame_ratios(axis_values_per_frame["left"], ratios, "left"),
        "right": compute_per_frame_ratios(axis_values_per_frame["right"], ratios, "right"),
    }
    recommendations = {
        side: find_recommended_ratio(
            per_frame_retained=per_frame_curves[side],
            ratios=ratios,
            required_retained_ratio=args.retained_feature_ratio,
            frame_quantile=args.frame_quantile,
        )
        for side in ("top", "bottom", "left", "right")
    }
    combined_retained_per_frame = compute_combined_per_frame_ratios(points_per_frame, recommendations)
    combined_retained_aggregate = compute_combined_retained_ratio(
        np_mod.stack([analysis.xs_norm, analysis.ys_norm], axis=1)
        if analysis.feature_count_total > 0
        else np_mod.empty((0, 2), dtype=np_mod.float32),
        recommendations,
    )

    slug = topic.strip("/").replace("/", "__") or "root"
    for side in ("top", "bottom", "left", "right"):
        export_csv(
            output_dir / f"{slug}_{side}_retained.csv",
            ratios=ratios,
            aggregate=side_curves[side],
            per_frame_retained=per_frame_curves[side],
        )

    if not args.no_plots:
        save_topic_plot(
            output_dir / f"{slug}_summary.png",
            analysis=analysis,
            ratios=ratios,
            side_curves=side_curves,
            per_frame_curves=per_frame_curves,
            recommendations=recommendations,
            required_retained_ratio=args.retained_feature_ratio,
            frame_quantile=args.frame_quantile,
            heatmap_bins=args.heatmap_bins,
        )

    return {
        "topic": topic,
        "analysis": analysis,
        "recommendations": recommendations,
        "combined_retained_aggregate": combined_retained_aggregate,
        "combined_retained_frame_quantile": float(np_mod.quantile(combined_retained_per_frame, args.frame_quantile)),
    }


def print_summary(result: dict[str, object]) -> None:
    topic = result["topic"]
    analysis: TopicAnalysis = result["analysis"]  # type: ignore[assignment]
    recommendations: dict[str, float] = result["recommendations"]  # type: ignore[assignment]
    combined_retained_aggregate = float(result["combined_retained_aggregate"])
    combined_retained_frame_quantile = float(result["combined_retained_frame_quantile"])

    print(
        "[RESULT] "
        f"topic={topic} frames={analysis.analyzed_frames} size={analysis.width}x{analysis.height} "
        f"features={analysis.feature_count_total}"
    )
    for side in ("top", "bottom", "left", "right"):
        pixels = recommendations[side] * (analysis.height if side in ("top", "bottom") else analysis.width)
        print(f"  - recommended_{side}_ratio={recommendations[side]:.4f} ({pixels:.1f} px)")
    print(
        "  - combined_selected_retained_ratio="
        f"{combined_retained_aggregate:.4f} "
        f"(frame_q={combined_retained_frame_quantile:.4f})"
    )


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame_stride must be >= 1")
    if not (0.0 < args.retained_feature_ratio <= 1.0):
        raise ValueError("--retained_feature_ratio must be within (0, 1]")
    if not (0.0 <= args.frame_quantile <= 1.0):
        raise ValueError("--frame_quantile must be within [0, 1]")
    if args.ratio_step <= 0.0:
        raise ValueError("--ratio_step must be > 0")

    bag_path = normalize_bag_path(args.bag)
    bag_name = bag_path.name if bag_path.name else "bag"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (Path("/tmp") / f"camera_crop_analysis_{bag_name}").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] bag={bag_path}")
    print(f"[INFO] output_dir={output_dir}")
    print(f"[INFO] topics={args.topics}")

    results = []
    for topic in args.topics:
        print(f"[INFO] analyzing topic={topic}")
        result = analyze_single_topic(bag_path=bag_path, topic=topic, args=args, output_dir=output_dir)
        results.append(result)
        print_summary(result)

    summary_csv = output_dir / "recommended_crop_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "topic",
                "frames",
                "width_px",
                "height_px",
                "feature_count_total",
                "recommended_top_ratio",
                "recommended_bottom_ratio",
                "recommended_left_ratio",
                "recommended_right_ratio",
                "recommended_top_px",
                "recommended_bottom_px",
                "recommended_left_px",
                "recommended_right_px",
                "combined_selected_retained_ratio",
                "combined_selected_retained_ratio_frame_quantile",
            ]
        )
        for result in results:
            analysis: TopicAnalysis = result["analysis"]  # type: ignore[assignment]
            recommendations: dict[str, float] = result["recommendations"]  # type: ignore[assignment]
            writer.writerow(
                [
                    result["topic"],
                    analysis.analyzed_frames,
                    analysis.width,
                    analysis.height,
                    analysis.feature_count_total,
                    f"{recommendations['top']:.6f}",
                    f"{recommendations['bottom']:.6f}",
                    f"{recommendations['left']:.6f}",
                    f"{recommendations['right']:.6f}",
                    f"{recommendations['top'] * analysis.height:.2f}",
                    f"{recommendations['bottom'] * analysis.height:.2f}",
                    f"{recommendations['left'] * analysis.width:.2f}",
                    f"{recommendations['right'] * analysis.width:.2f}",
                    f"{float(result['combined_retained_aggregate']):.6f}",
                    f"{float(result['combined_retained_frame_quantile']):.6f}",
                ]
            )

    print(f"[INFO] saved summary csv: {summary_csv}")


if __name__ == "__main__":
    main()

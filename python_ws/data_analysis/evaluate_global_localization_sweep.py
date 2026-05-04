#!/usr/bin/env python3
"""
Automated global-localization evaluation with rosbag2 step playback.

Workflow:
1) Keep rosbag2 player paused.
2) Advance playback message-by-message until scan count reaches stride.
3) Trigger global localization service.
4) Wait localization result and compare against reference pose topic.
5) Write residuals to CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosbag2_interfaces.srv import Pause, PlayNext
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


CSV_FIELDS = [
    "idx",
    "status",
    "scan_count",
    "trigger_scan_stamp_ns",
    "reference_stamp_ns",
    "localization_stamp_ns",
    "reference_x",
    "reference_y",
    "reference_yaw_rad",
    "localization_x",
    "localization_y",
    "localization_yaw_rad",
    "position_error_m",
    "yaw_error_rad",
    "localization_latency_sec",
]


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


@dataclass
class Pose2DStamped:
    stamp_ns: int
    x: float
    y: float
    yaw: float


@dataclass
class MapMeta:
    image_path: Path
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    image_width: int
    image_height: int


class LocalizationSweepEvaluator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("global_localization_sweep_evaluator")
        self.set_parameters(
            [Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)]
        )

        self.args = args
        self.scan_count = 0
        self.latest_scan_stamp_ns = 0

        self.localization_seq = 0
        self.latest_localization: Optional[Pose2DStamped] = None
        self.latest_localization_latency_sec = math.nan
        self.last_trigger_wall_time = 0.0

        self.latest_reference: Optional[Pose2DStamped] = None

        self.scan_sub = self.create_subscription(
            LaserScan, args.scan_topic, self._scan_callback, 20
        )
        self.localization_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            args.localization_topic,
            self._localization_callback,
            20,
        )

        self.reference_type = args.reference_type
        if self.reference_type == "pose_stamped":
            self.reference_sub = self.create_subscription(
                PoseStamped, args.reference_topic, self._reference_pose_stamped_cb, 20
            )
        elif self.reference_type == "pose_cov":
            self.reference_sub = self.create_subscription(
                PoseWithCovarianceStamped,
                args.reference_topic,
                self._reference_pose_cov_cb,
                20,
            )
        else:
            self.reference_sub = self.create_subscription(
                Odometry, args.reference_topic, self._reference_odom_cb, 20
            )

        player_prefix = args.player_prefix.rstrip("/")
        self.pause_client = self.create_client(Pause, f"{player_prefix}/pause")
        self.play_next_client = self.create_client(
            PlayNext, f"{player_prefix}/play_next"
        )
        self.trigger_client = self.create_client(
            Empty, args.localization_trigger_service
        )

    def _scan_callback(self, msg: LaserScan) -> None:
        self.scan_count += 1
        self.latest_scan_stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(
            msg.header.stamp.nanosec
        )

    def _localization_callback(self, msg: PoseWithCovarianceStamped) -> None:
        q = msg.pose.pose.orientation
        self.latest_localization = Pose2DStamped(
            stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )
        self.localization_seq += 1
        if self.last_trigger_wall_time > 0.0:
            self.latest_localization_latency_sec = (
                time.monotonic() - self.last_trigger_wall_time
            )

    def _reference_pose_stamped_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self.latest_reference = Pose2DStamped(
            stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec),
            x=float(msg.pose.position.x),
            y=float(msg.pose.position.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def _reference_pose_cov_cb(self, msg: PoseWithCovarianceStamped) -> None:
        q = msg.pose.pose.orientation
        self.latest_reference = Pose2DStamped(
            stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def _reference_odom_cb(self, msg: Odometry) -> None:
        q = msg.pose.pose.orientation
        self.latest_reference = Pose2DStamped(
            stamp_ns=int(msg.header.stamp.sec) * 1_000_000_000
            + int(msg.header.stamp.nanosec),
            x=float(msg.pose.pose.position.x),
            y=float(msg.pose.pose.position.y),
            yaw=yaw_from_quaternion(q.x, q.y, q.z, q.w),
        )

    def _spin_until(self, predicate, timeout_sec: float) -> bool:
        end = time.monotonic() + max(0.0, timeout_sec)
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(self, timeout_sec=0.05)
            if predicate():
                return True
        return predicate()

    def _call_service(self, client, request, timeout_sec: float):
        future = client.call_async(request)
        ok = self._spin_until(lambda: future.done(), timeout_sec)
        if not ok:
            return None
        try:
            return future.result()
        except Exception as exc:  # pragma: no cover
            self.get_logger().error(f"Service call failed: {exc}")
            return None

    def wait_for_services(self) -> bool:
        clients = [
            (self.pause_client, "pause"),
            (self.play_next_client, "play_next"),
            (self.trigger_client, "localization_trigger"),
        ]
        for client, name in clients:
            if not client.wait_for_service(timeout_sec=self.args.service_timeout_sec):
                self.get_logger().error(
                    "Service unavailable: "
                    f"{name} (expected service: {client.srv_name})"
                )
                return False
        return True

    def ensure_paused(self) -> bool:
        res = self._call_service(
            self.pause_client, Pause.Request(), self.args.service_timeout_sec
        )
        if res is None:
            return False
        return True

    def step_until_scan_delta(self, scan_delta: int) -> bool:
        if scan_delta <= 0:
            return True

        start_scan_count = self.scan_count
        max_calls = self.args.max_play_next_calls_per_trigger
        calls = 0
        while self.scan_count - start_scan_count < scan_delta:
            if max_calls > 0 and calls >= max_calls:
                self.get_logger().warn(
                    "Exceeded max play_next calls before reaching requested scan delta. "
                    f"scan_delta={self.scan_count - start_scan_count}, "
                    f"requested_scan_delta={scan_delta}, "
                    f"play_next_calls={calls}. "
                    "Increase --max-play-next-calls-per-trigger or set it to 0 for no limit."
                )
                return False
            calls += 1

            prev_scan_count = self.scan_count
            res = self._call_service(
                self.play_next_client, PlayNext.Request(), self.args.service_timeout_sec
            )
            if res is None:
                return False
            if not getattr(res, "success", True):
                self.get_logger().info("play_next returned success=false (likely EOF).")
                return False

            # Wait a short time for callbacks after each played message.
            self._spin_until(
                lambda: self.scan_count > prev_scan_count,
                self.args.spin_wait_after_play_next_sec,
            )

        return True

    def step_until_scan_stride(self) -> bool:
        return self.step_until_scan_delta(self.args.scan_stride)

    def run_sweep(self) -> None:
        output_path = Path(self.args.output_csv).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)

            if not self.wait_for_services():
                return
            if not self.ensure_paused():
                return

            self.get_logger().info("Start automated sweep...")
            idx = 0
            while rclpy.ok():
                if self.args.max_triggers > 0 and idx >= self.args.max_triggers:
                    self.get_logger().info("Reached max_triggers.")
                    break

                # Trigger works on "next flatscan". To evaluate every scan_stride scans,
                # pre-advance (scan_stride - 1), then trigger, then advance one scan.
                pre_trigger_scan_delta = max(0, self.args.scan_stride - 1)
                step_ok = self.step_until_scan_delta(pre_trigger_scan_delta)
                if not step_ok:
                    self.get_logger().info("Sweep finished (no more messages or step failed).")
                    break

                idx += 1
                before_seq = self.localization_seq
                trigger_scan_stamp_ns = self.latest_scan_stamp_ns
                trigger_ref = self.latest_reference

                self.last_trigger_wall_time = time.monotonic()
                trigger_res = self._call_service(
                    self.trigger_client,
                    Empty.Request(),
                    self.args.service_timeout_sec,
                )
                if trigger_res is None:
                    self.get_logger().warn(f"[{idx}] Trigger call failed.")
                    writer.writerow(
                        self._build_csv_row(
                            idx=idx,
                            status="trigger_failed",
                            trigger_scan_stamp_ns=trigger_scan_stamp_ns,
                            trigger_ref=trigger_ref,
                        )
                    )
                    continue

                # Supply the "next scan" after trigger so OccupancyGridLocalizer can run.
                post_trigger_step_ok = self.step_until_scan_delta(1)
                if not post_trigger_step_ok:
                    self.get_logger().warn(f"[{idx}] No scan available after trigger.")
                    writer.writerow(
                        self._build_csv_row(
                            idx=idx,
                            status="no_scan_after_trigger",
                            trigger_scan_stamp_ns=trigger_scan_stamp_ns,
                            trigger_ref=trigger_ref,
                        )
                    )
                    continue

                # Re-bind trigger-aligned scan/reference after supplying next scan.
                trigger_scan_stamp_ns = self.latest_scan_stamp_ns
                trigger_ref = self.latest_reference

                got_result = self._spin_until(
                    lambda: self.localization_seq > before_seq,
                    self.args.localization_timeout_sec,
                )
                loc = self.latest_localization

                if not got_result or loc is None:
                    self.get_logger().warn(f"[{idx}] Localization timeout.")
                    writer.writerow(
                        self._build_csv_row(
                            idx=idx,
                            status="localization_timeout",
                            trigger_scan_stamp_ns=trigger_scan_stamp_ns,
                            trigger_ref=trigger_ref,
                        )
                    )
                    continue

                if trigger_ref is None:
                    row = self._build_csv_row(
                        idx=idx,
                        status="ok_no_reference",
                        trigger_scan_stamp_ns=trigger_scan_stamp_ns,
                        loc=loc,
                    )
                else:
                    pos_error = math.hypot(loc.x - trigger_ref.x, loc.y - trigger_ref.y)
                    yaw_error = wrap_to_pi(loc.yaw - trigger_ref.yaw)
                    row = self._build_csv_row(
                        idx=idx,
                        status="ok",
                        trigger_scan_stamp_ns=trigger_scan_stamp_ns,
                        trigger_ref=trigger_ref,
                        loc=loc,
                        pos_error=pos_error,
                        yaw_error=yaw_error,
                    )

                    self.get_logger().info(
                        f"[{idx}] pos_err={pos_error:.3f} m yaw_err={yaw_error:.3f} rad "
                        f"latency={self.latest_localization_latency_sec:.3f} s"
                    )

                writer.writerow(row)
                f.flush()

            self.get_logger().info(f"Saved evaluation result: {output_path}")

        if self.args.map_yaml:
            self._render_map_quality_images(output_path)

    def _build_csv_row(
        self,
        idx: int,
        status: str,
        trigger_scan_stamp_ns: int,
        trigger_ref: Optional[Pose2DStamped] = None,
        loc: Optional[Pose2DStamped] = None,
        pos_error: Optional[float] = None,
        yaw_error: Optional[float] = None,
    ) -> list:
        return [
            idx,
            status,
            self.scan_count,
            trigger_scan_stamp_ns,
            trigger_ref.stamp_ns if trigger_ref is not None else "",
            loc.stamp_ns if loc is not None else "",
            trigger_ref.x if trigger_ref is not None else "",
            trigger_ref.y if trigger_ref is not None else "",
            trigger_ref.yaw if trigger_ref is not None else "",
            loc.x if loc is not None else "",
            loc.y if loc is not None else "",
            loc.yaw if loc is not None else "",
            pos_error if pos_error is not None else "",
            yaw_error if yaw_error is not None else "",
            self.latest_localization_latency_sec if loc is not None else "",
        ]

    def _render_map_quality_images(self, csv_path: Path) -> None:
        try:
            rate_path, points_path = render_map_quality_images(
                csv_path=csv_path,
                map_yaml_path=Path(self.args.map_yaml).expanduser().resolve(),
                good_pos_error_threshold_m=self.args.good_pos_error_threshold_m,
                grid_size_m=self.args.quality_grid_size_m,
                min_samples_per_cell=self.args.quality_min_samples_per_cell,
                rate_output_path=(
                    Path(self.args.quality_rate_output).expanduser().resolve()
                    if self.args.quality_rate_output
                    else None
                ),
                points_output_path=(
                    Path(self.args.quality_points_output).expanduser().resolve()
                    if self.args.quality_points_output
                    else None
                ),
            )
            self.get_logger().info(f"Saved quality points plot: {points_path}")
            self.get_logger().info(f"Saved success-rate heatmap: {rate_path}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to render quality images: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Automated global localization sweep evaluator"
    )
    parser.add_argument("--scan-topic", default="/scan")
    parser.add_argument("--scan-stride", type=int, default=50)
    parser.add_argument("--max-triggers", type=int, default=0)
    parser.add_argument(
        "--max-play-next-calls-per-trigger",
        type=int,
        default=0,
        help=(
            "Maximum rosbag2 play_next calls used to reach one trigger point. "
            "Set to 0 to disable the limit. Default: 0."
        ),
    )
    parser.add_argument("--spin-wait-after-play-next-sec", type=float, default=0.05)

    parser.add_argument("--localization-trigger-service", default="/trigger_grid_search_localization")
    parser.add_argument("--localization-topic", default="/localization_result")
    parser.add_argument("--localization-timeout-sec", type=float, default=8.0)

    parser.add_argument("--reference-topic", default="/visual_slam/tracking/vo_pose")
    parser.add_argument(
        "--reference-type",
        choices=["pose_stamped", "pose_cov", "odom"],
        default="pose_stamped",
    )

    parser.add_argument("--player-prefix", default="/rosbag2_player")
    parser.add_argument("--service-timeout-sec", type=float, default=5.0)
    sim_time_group = parser.add_mutually_exclusive_group()
    sim_time_group.add_argument(
        "--use-sim-time",
        dest="use_sim_time",
        action="store_true",
        default=True,
        help="Use simulation time (default: enabled).",
    )
    sim_time_group.add_argument(
        "--no-use-sim-time",
        dest="use_sim_time",
        action="store_false",
        help="Disable simulation time.",
    )
    parser.add_argument("--output-csv", default="./localization_sweep_eval.csv")
    parser.add_argument(
        "--map-yaml",
        default="",
        help="Optional 2D map yaml path. If set, quality overlay images are generated.",
    )
    parser.add_argument(
        "--good-pos-error-threshold-m",
        type=float,
        default=0.5,
        help="Threshold for classifying good localization points.",
    )
    parser.add_argument(
        "--quality-grid-size-m",
        type=float,
        default=1.0,
        help="Grid size [m] for success-rate heatmap aggregation.",
    )
    parser.add_argument(
        "--quality-min-samples-per-cell",
        type=int,
        default=1,
        help="Minimum samples per cell to visualize heatmap value.",
    )
    parser.add_argument(
        "--quality-rate-output",
        default="",
        help="Optional output image path for success-rate heatmap.",
    )
    parser.add_argument(
        "--quality-points-output",
        default="",
        help="Optional output image path for good/bad points plot.",
    )
    return parser.parse_args()


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
    import matplotlib
    matplotlib.use("Agg", force=True)
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

    # 1) Good/Bad/Fail/No-reference points overlay
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

    # 2) Success-rate heatmap (good / total), or successful-localization density when
    # no reference pose is available.
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

    has_reference_quality = bool(np.any(good_arr) or np.any(bad_arr))

    rate = np.full((bins_y, bins_x), np.nan, dtype=np.float32)
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


def main() -> int:
    args = parse_args()
    rclpy.init()
    node = LocalizationSweepEvaluator(args)
    try:
        node.run_sweep()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

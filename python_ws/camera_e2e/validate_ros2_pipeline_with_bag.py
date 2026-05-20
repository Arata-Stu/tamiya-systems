#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosbag2_interfaces.srv import Pause, PlayNext
from sensor_msgs.msg import CompressedImage, Image

import compare_pytorch_triton as compare


CSV_FIELDS = [
    "sample_idx",
    "image_seq",
    "image_stamp_ns",
    "cmd_seq",
    "cmd_stamp_ns",
    "play_next_calls_for_image",
    "play_next_calls_for_cmd",
    "image_encoding",
    "image_height",
    "image_width",
    "pytorch_train_steer",
    "pytorch_train_speed",
    "pytorch_ros2_steer",
    "pytorch_ros2_speed",
    "onnx_ros2_steer",
    "onnx_ros2_speed",
    "live_cmd_steer",
    "live_cmd_speed",
    "abs_live_vs_pytorch_ros2_steer",
    "abs_live_vs_pytorch_ros2_speed",
    "abs_live_vs_onnx_ros2_steer",
    "abs_live_vs_onnx_ros2_speed",
]


@dataclass
class CapturedImage:
    seq: int
    stamp_ns: int
    encoding: str
    image: np.ndarray


@dataclass
class CapturedCmd:
    seq: int
    stamp_ns: int
    steer: float
    speed: float


class Ros2PipelineValidator(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("camera_e2e_pipeline_validator")
        self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, args.use_sim_time)])

        self.args = args
        self.image_seq = 0
        self.cmd_seq = 0
        self.images: Deque[CapturedImage] = deque()
        self.cmds: Deque[CapturedCmd] = deque()

        image_topic = args.image_topic
        compressed = image_topic.endswith("/compressed")
        if compressed:
            self.image_sub = self.create_subscription(CompressedImage, image_topic, self._compressed_image_cb, 20)
        else:
            self.image_sub = self.create_subscription(Image, image_topic, self._image_cb, 20)

        self.cmd_sub = self.create_subscription(AckermannDriveStamped, args.cmd_topic, self._cmd_cb, 20)

        player_prefix = args.player_prefix.rstrip("/")
        self.pause_client = self.create_client(Pause, f"{player_prefix}/pause")
        self.play_next_client = self.create_client(PlayNext, f"{player_prefix}/play_next")

        self.cfg = compare.OmegaConf.load(compare.resolve_path(args.config))
        self.checkpoint_path = compare.resolve_checkpoint_path(args.checkpoint)
        model_repo_root = Path(args.model_repository_root).expanduser().resolve()
        self.onnx_path = compare.resolve_onnx_path(
            self.checkpoint_path,
            args.onnx,
            model_repo_root,
            args.model_name,
        )
        self.model = compare.load_model_from_checkpoint(self.checkpoint_path, self.cfg)
        self.train_preview_transform = compare.build_train_preview_transform(self.cfg)
        self.ros2_preview_transform = compare.build_ros2_preview_transform(self.cfg)

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
        res = self._call_service(self.pause_client, Pause.Request(), self.args.service_timeout_sec)
        return res is not None

    def _image_cb(self, msg: Image) -> None:
        try:
            image, encoding = compare.decode_raw_image_message(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode image message: {exc}")
            return

        self.image_seq += 1
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.images.append(
            CapturedImage(
                seq=self.image_seq,
                stamp_ns=stamp_ns,
                encoding=encoding,
                image=image,
            )
        )

    def _compressed_image_cb(self, msg: CompressedImage) -> None:
        try:
            image, encoding = compare.decode_compressed_image_message(msg)
        except Exception as exc:
            self.get_logger().warn(f"Failed to decode compressed image: {exc}")
            return

        self.image_seq += 1
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.images.append(
            CapturedImage(
                seq=self.image_seq,
                stamp_ns=stamp_ns,
                encoding=encoding,
                image=image,
            )
        )

    def _cmd_cb(self, msg: AckermannDriveStamped) -> None:
        self.cmd_seq += 1
        stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
        self.cmds.append(
            CapturedCmd(
                seq=self.cmd_seq,
                stamp_ns=stamp_ns,
                steer=float(msg.drive.steering_angle),
                speed=float(msg.drive.speed),
            )
        )

    def _play_next(self) -> bool:
        res = self._call_service(self.play_next_client, PlayNext.Request(), self.args.service_timeout_sec)
        if res is None:
            return False
        return getattr(res, "success", True)

    def step_until_image(self, after_image_seq: int) -> tuple[Optional[CapturedImage], int]:
        calls = 0
        while rclpy.ok():
            if self.images and self.images[-1].seq > after_image_seq:
                while self.images and self.images[0].seq <= after_image_seq:
                    self.images.popleft()
                return self.images.popleft(), calls

            if self.args.max_play_next_calls_per_image > 0 and calls >= self.args.max_play_next_calls_per_image:
                return None, calls

            calls += 1
            ok = self._play_next()
            if not ok:
                return None, calls
            self._spin_until(lambda: self.image_seq > after_image_seq, self.args.spin_wait_after_play_next_sec)

        return None, calls

    def wait_for_cmd_after_image(self, after_cmd_seq: int, image_stamp_ns: int) -> tuple[Optional[CapturedCmd], int]:
        calls = 0
        start = time.monotonic()
        while rclpy.ok():
            while self.cmds:
                cmd = self.cmds.popleft()
                if cmd.seq <= after_cmd_seq:
                    continue
                if self.args.require_matching_stamp and cmd.stamp_ns != image_stamp_ns:
                    continue
                if (not self.args.require_matching_stamp) and cmd.stamp_ns < image_stamp_ns:
                    continue
                return cmd, calls

            if time.monotonic() - start < self.args.processing_timeout_sec:
                self._spin_until(lambda: len(self.cmds) > 0, self.args.processing_timeout_sec * 0.1)
                continue

            if self.args.max_play_next_calls_per_cmd > 0 and calls >= self.args.max_play_next_calls_per_cmd:
                return None, calls

            calls += 1
            ok = self._play_next()
            if not ok:
                return None, calls
            self._spin_until(lambda: len(self.cmds) > 0, self.args.spin_wait_after_play_next_sec)

        return None, calls

    def evaluate_image(self, image: np.ndarray) -> dict:
        extractor_image = compare.emulate_dataset_extractor_output(image)
        ros2_encoder_image = compare.emulate_ros2_encoder_input(
            image,
            force_grayscale_3ch=self.args.force_grayscale_3ch,
            use_rgb_format_converter=self.args.use_rgb_format_converter,
        )

        train_preview = compare.apply_preview_transform(extractor_image, self.train_preview_transform)
        ros2_preview = compare.apply_preview_transform(ros2_encoder_image, self.ros2_preview_transform)

        train_tensor = compare.normalize_preview_image(
            train_preview, self.cfg.dataset.pixel_mean, self.cfg.dataset.pixel_std
        )
        ros2_tensor = compare.normalize_preview_image(
            ros2_preview, self.cfg.dataset.pixel_mean, self.cfg.dataset.pixel_std
        )

        pytorch_train = compare.run_pytorch(self.model, train_tensor)
        pytorch_ros2 = compare.run_pytorch(self.model, ros2_tensor)
        onnx_ros2 = compare.run_onnx(self.onnx_path, ros2_tensor) if self.onnx_path is not None else None

        return {
            "pytorch_train": pytorch_train,
            "pytorch_ros2": pytorch_ros2,
            "onnx_ros2": onnx_ros2,
        }

    def run(self) -> None:
        if not self.wait_for_services():
            return
        if not self.ensure_paused():
            self.get_logger().error("Failed to pause rosbag2 player.")
            return

        output_csv = compare.resolve_path(self.args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_FIELDS)

            image_marker = self.image_seq
            cmd_marker = self.cmd_seq
            sample_idx = 0

            while rclpy.ok():
                if self.args.max_samples > 0 and sample_idx >= self.args.max_samples:
                    self.get_logger().info("Reached max_samples.")
                    break

                image_event, play_calls_for_image = self.step_until_image(image_marker)
                if image_event is None:
                    self.get_logger().info("No more images available or step_until_image failed.")
                    break
                image_marker = image_event.seq

                outputs = self.evaluate_image(image_event.image)
                cmd_event, play_calls_for_cmd = self.wait_for_cmd_after_image(cmd_marker, image_event.stamp_ns)
                if cmd_event is None:
                    self.get_logger().warn(
                        f"[{sample_idx}] No command received for image stamp {image_event.stamp_ns}."
                    )
                    break
                cmd_marker = cmd_event.seq

                pytorch_train = outputs["pytorch_train"]
                pytorch_ros2 = outputs["pytorch_ros2"]
                onnx_ros2 = outputs["onnx_ros2"]

                live_vs_pt = np.abs(
                    np.array([cmd_event.steer, cmd_event.speed], dtype=np.float32) - pytorch_ros2
                )
                live_vs_onnx = (
                    np.abs(np.array([cmd_event.steer, cmd_event.speed], dtype=np.float32) - onnx_ros2)
                    if onnx_ros2 is not None
                    else np.array([np.nan, np.nan], dtype=np.float32)
                )

                writer.writerow(
                    [
                        sample_idx,
                        image_event.seq,
                        image_event.stamp_ns,
                        cmd_event.seq,
                        cmd_event.stamp_ns,
                        play_calls_for_image,
                        play_calls_for_cmd,
                        image_event.encoding,
                        int(image_event.image.shape[0]),
                        int(image_event.image.shape[1]),
                        float(pytorch_train[0]),
                        float(pytorch_train[1]),
                        float(pytorch_ros2[0]),
                        float(pytorch_ros2[1]),
                        float(onnx_ros2[0]) if onnx_ros2 is not None else "",
                        float(onnx_ros2[1]) if onnx_ros2 is not None else "",
                        float(cmd_event.steer),
                        float(cmd_event.speed),
                        float(live_vs_pt[0]),
                        float(live_vs_pt[1]),
                        float(live_vs_onnx[0]) if onnx_ros2 is not None else "",
                        float(live_vs_onnx[1]) if onnx_ros2 is not None else "",
                    ]
                )
                f.flush()

                self.get_logger().info(
                    f"[{sample_idx}] image_stamp={image_event.stamp_ns} "
                    f"live=({cmd_event.steer:+.6f}, {cmd_event.speed:+.6f}) "
                    f"pt_ros2=({pytorch_ros2[0]:+.6f}, {pytorch_ros2[1]:+.6f}) "
                    f"abs=({live_vs_pt[0]:.8f}, {live_vs_pt[1]:.8f})"
                )
                sample_idx += 1

        self.get_logger().info(f"Saved evaluation CSV: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the launched ROS2 camera_e2e pipeline by stepping a paused rosbag2 player "
            "and comparing live autonomous command outputs against offline PyTorch/ONNX results."
        )
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Training config path")
    parser.add_argument("--onnx", type=str, default=None, help="Optional model.onnx path")
    parser.add_argument("--model-repository-root", type=str, default="/workspaces/isaac_ros_assets/models")
    parser.add_argument("--model-name", type=str, default="pilotnet")

    parser.add_argument("--image-topic", default="/camera/left/image_raw")
    parser.add_argument(
        "--cmd-topic",
        default="/autonomous/cmd_drive",
        help=(
            "Live command topic to compare against. "
            "Use /autonomous/cmd_drive for the default launch, or /autonomous/cmd_drive_raw "
            "when isaac_ros_camera_e2e.launch.xml is started with control_filter:=true."
        ),
    )
    parser.add_argument("--player-prefix", default="/rosbag2_player")
    parser.add_argument("--output-csv", default="./outputs/ros2_pipeline_eval.csv")

    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-play-next-calls-per-image", type=int, default=200)
    parser.add_argument("--max-play-next-calls-per-cmd", type=int, default=200)
    parser.add_argument("--spin-wait-after-play-next-sec", type=float, default=0.05)
    parser.add_argument("--processing-timeout-sec", type=float, default=0.5)
    parser.add_argument("--service-timeout-sec", type=float, default=5.0)
    parser.add_argument("--require-matching-stamp", action="store_true", default=False)

    parser.add_argument("--force-grayscale-3ch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-rgb-format-converter", action=argparse.BooleanOptionalAction, default=True)

    sim_time_group = parser.add_mutually_exclusive_group()
    sim_time_group.add_argument(
        "--use-sim-time",
        dest="use_sim_time",
        action="store_true",
        default=True,
    )
    sim_time_group.add_argument(
        "--no-use-sim-time",
        dest="use_sim_time",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = Ros2PipelineValidator(args)
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

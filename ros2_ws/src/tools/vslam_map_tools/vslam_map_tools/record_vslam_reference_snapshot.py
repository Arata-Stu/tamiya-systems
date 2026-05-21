#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy


def pose_to_dict(pose) -> dict[str, object]:
    return {
        "position": {
            "x": float(pose.position.x),
            "y": float(pose.position.y),
            "z": float(pose.position.z),
        },
        "orientation": {
            "x": float(pose.orientation.x),
            "y": float(pose.orientation.y),
            "z": float(pose.orientation.z),
            "w": float(pose.orientation.w),
        },
    }


def twist_to_dict(twist) -> dict[str, object]:
    return {
        "linear": {
            "x": float(twist.linear.x),
            "y": float(twist.linear.y),
            "z": float(twist.linear.z),
        },
        "angular": {
            "x": float(twist.angular.x),
            "y": float(twist.angular.y),
            "z": float(twist.angular.z),
        },
    }


class VslamReferenceSnapshotRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("vslam_reference_snapshot_recorder")
        self.args = args
        self.output_path = Path(args.output).expanduser().resolve()
        self.latest_path: PathMsg | None = None
        self.latest_odom: Odometry | None = None
        self.path_seen = False
        self.odom_seen = False
        self.dirty = False

        # VSLAM topics may be published as BEST_EFFORT, so request a permissive QoS.
        best_effort_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(PathMsg, args.path_topic, self.on_path, best_effort_qos)
        if args.odom_topic:
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, best_effort_qos)
        self.create_timer(1.0, self.flush_if_dirty)

    def on_path(self, msg: PathMsg) -> None:
        self.latest_path = msg
        if not self.path_seen:
            self.path_seen = True
            self.get_logger().info(
                f"Received first path on {self.args.path_topic} with {len(msg.poses)} poses."
            )
        self.dirty = True

    def on_odom(self, msg: Odometry) -> None:
        self.latest_odom = msg
        if not self.odom_seen:
            self.odom_seen = True
            self.get_logger().info(f"Received first odometry on {self.args.odom_topic}.")
        self.dirty = True

    def flush_if_dirty(self) -> None:
        if not self.dirty:
            return
        self.write_snapshot()
        self.dirty = False

    def snapshot_data(self) -> dict[str, object]:
        data: dict[str, object] = {"path": None, "odometry": None}

        if self.latest_path is not None:
            data["path"] = {
                "frame_id": self.latest_path.header.frame_id,
                "poses": [
                    pose_to_dict(pose_stamped.pose)
                    for pose_stamped in self.latest_path.poses
                ],
            }

        if self.latest_odom is not None:
            data["odometry"] = {
                "frame_id": self.latest_odom.header.frame_id,
                "child_frame_id": self.latest_odom.child_frame_id,
                "pose": pose_to_dict(self.latest_odom.pose.pose),
                "twist": twist_to_dict(self.latest_odom.twist.twist),
            }

        return data

    def write_snapshot(self) -> None:
        if self.latest_path is None and self.latest_odom is None:
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self.snapshot_data(), ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record the latest VSLAM path/odometry snapshot to disk.")
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--odom-topic", default="/visual_slam/tracking/odometry")
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=rclpy.utilities.remove_ros_args()[1:])

    rclpy.init()
    node = VslamReferenceSnapshotRecorder(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write_snapshot()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

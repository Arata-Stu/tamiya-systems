#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node


def fill_pose(target_pose, pose_dict: dict[str, object]) -> None:
    position = pose_dict["position"]
    orientation = pose_dict["orientation"]
    target_pose.position.x = float(position["x"])
    target_pose.position.y = float(position["y"])
    target_pose.position.z = float(position["z"])
    target_pose.orientation.x = float(orientation["x"])
    target_pose.orientation.y = float(orientation["y"])
    target_pose.orientation.z = float(orientation["z"])
    target_pose.orientation.w = float(orientation["w"])


def fill_twist(target_twist, twist_dict: dict[str, object]) -> None:
    linear = twist_dict["linear"]
    angular = twist_dict["angular"]
    target_twist.linear.x = float(linear["x"])
    target_twist.linear.y = float(linear["y"])
    target_twist.linear.z = float(linear["z"])
    target_twist.angular.x = float(angular["x"])
    target_twist.angular.y = float(angular["y"])
    target_twist.angular.z = float(angular["z"])


class SavedVslamReferencePublisher(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("saved_vslam_reference_publisher")
        snapshot_path = Path(args.input).expanduser().resolve()
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        self.path_data = snapshot.get("path")
        self.odom_data = snapshot.get("odometry")

        self.path_pub = None
        self.odom_pub = None
        if self.path_data is not None:
            self.path_pub = self.create_publisher(PathMsg, args.path_topic, 10)
        if self.odom_data is not None and args.odom_topic:
            self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)

        self.timer = self.create_timer(max(0.05, 1.0 / max(args.publish_rate_hz, 1.0)), self.publish_messages)
        self.get_logger().info(f"Publishing saved VSLAM reference from {snapshot_path}")

    def publish_messages(self) -> None:
        stamp = self.get_clock().now().to_msg()

        if self.path_pub is not None and self.path_data is not None:
            path_msg = PathMsg()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = str(self.path_data.get("frame_id", "vslam_map"))
            poses = self.path_data.get("poses", [])
            for pose_dict in poses:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = stamp
                pose_msg.header.frame_id = path_msg.header.frame_id
                fill_pose(pose_msg.pose, pose_dict)
                path_msg.poses.append(pose_msg)
            self.path_pub.publish(path_msg)

        if self.odom_pub is not None and self.odom_data is not None:
            odom_msg = Odometry()
            odom_msg.header.stamp = stamp
            odom_msg.header.frame_id = str(self.odom_data.get("frame_id", "vslam_map"))
            odom_msg.child_frame_id = str(self.odom_data.get("child_frame_id", "base_link"))
            fill_pose(odom_msg.pose.pose, self.odom_data["pose"])
            fill_twist(odom_msg.twist.twist, self.odom_data["twist"])
            self.odom_pub.publish(odom_msg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a saved VSLAM path/odometry snapshot with current timestamps.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--odom-topic", default="/visual_slam/tracking/odometry")
    parser.add_argument("--publish-rate-hz", type=float, default=5.0)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(args=rclpy.utilities.remove_ros_args()[1:])

    rclpy.init()
    node = SavedVslamReferencePublisher(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

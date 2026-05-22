#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path as PathMsg
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
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
        self.landmarks_data = snapshot.get("landmarks")
        self.full_path_data = snapshot.get("full_vslam_path")
        self.trajectory_data = snapshot.get("trajectory")

        self.path_pub = None
        self.odom_pub = None
        self.landmarks_pub = None
        self.full_path_pub = None
        self.trajectory_pub = None
        
        if self.path_data is not None:
            self.path_pub = self.create_publisher(PathMsg, args.path_topic, 10)
        if self.odom_data is not None and getattr(args, 'odom_topic', None):
            self.odom_pub = self.create_publisher(Odometry, args.odom_topic, 10)
        if self.landmarks_data is not None and getattr(args, 'landmarks_topic', None):
            self.landmarks_pub = self.create_publisher(PointCloud2, args.landmarks_topic, 10)
        if self.full_path_data is not None:
            self.full_path_pub = self.create_publisher(PathMsg, "/visual_slam/vis/full_slam_path", 10)
        if self.trajectory_data is not None and getattr(args, 'trajectory_topic', None):
            self.trajectory_pub = self.create_publisher(MarkerArray, args.trajectory_topic, 10)

        self.timer = self.create_timer(max(0.05, 1.0 / max(args.publish_rate_hz, 1.0)), self.publish_messages)
        self.get_logger().info(f"Publishing saved VSLAM reference from {snapshot_path}")

    def publish_messages(self) -> None:
        stamp = self.get_clock().now().to_msg()

        if self.path_pub is not None and self.path_data is not None:
            path_msg = PathMsg()
            path_msg.header.stamp = stamp
            path_msg.header.frame_id = str(self.path_data.get("frame_id", "map"))
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
            odom_msg.header.frame_id = str(self.odom_data.get("frame_id", "map"))
            odom_msg.child_frame_id = str(self.odom_data.get("child_frame_id", "base_link"))
            fill_pose(odom_msg.pose.pose, self.odom_data["pose"])
            fill_twist(odom_msg.twist.twist, self.odom_data["twist"])
            self.odom_pub.publish(odom_msg)

        if self.landmarks_pub is not None and self.landmarks_data is not None:
            lm_msg = PointCloud2()
            lm_msg.header.stamp = stamp
            lm_msg.header.frame_id = str(
                self.landmarks_data.get("header", {}).get("frame_id", "map")
            )
            lm_msg.height = self.landmarks_data["height"]
            lm_msg.width = self.landmarks_data["width"]
            lm_msg.fields = [
                PointField(name=f["name"], offset=f["offset"], datatype=f["datatype"], count=f["count"])
                for f in self.landmarks_data["fields"]
            ]
            lm_msg.is_bigendian = self.landmarks_data["is_bigendian"]
            lm_msg.point_step = self.landmarks_data["point_step"]
            lm_msg.row_step = self.landmarks_data["row_step"]
            lm_msg.data = base64.b64decode(self.landmarks_data["data"])
            lm_msg.is_dense = self.landmarks_data["is_dense"]
            self.landmarks_pub.publish(lm_msg)

        if self.full_path_pub is not None and self.full_path_data is not None:
            full_path_msg = PathMsg()
            full_path_msg.header.stamp = stamp
            full_path_msg.header.frame_id = str(self.full_path_data.get("frame_id", "map"))
            poses = self.full_path_data.get("poses", [])
            for pose_dict in poses:
                pose_msg = PoseStamped()
                pose_msg.header.stamp = stamp
                pose_msg.header.frame_id = full_path_msg.header.frame_id
                fill_pose(pose_msg.pose, pose_dict)
                full_path_msg.poses.append(pose_msg)
            self.full_path_pub.publish(full_path_msg)

        if self.trajectory_pub is not None and self.trajectory_data is not None:
            trajectory_msg = MarkerArray()
            for m_data in self.trajectory_data.get("markers", []):
                m = Marker()
                m.header.stamp = stamp
                m.header.frame_id = m_data.get("header", {}).get("frame_id", "")
                m.ns = m_data.get("ns", "")
                m.id = m_data.get("id", 0)
                m.type = m_data.get("type", 0)
                m.action = m_data.get("action", 0)
                
                if "pose" in m_data:
                    fill_pose(m.pose, m_data["pose"])
                    
                if "scale" in m_data:
                    s = m_data["scale"]
                    m.scale.x = float(s.get("x", 0.0))
                    m.scale.y = float(s.get("y", 0.0))
                    m.scale.z = float(s.get("z", 0.0))
                    
                if "color" in m_data:
                    c = m_data["color"]
                    m.color.r = float(c.get("r", 0.0))
                    m.color.g = float(c.get("g", 0.0))
                    m.color.b = float(c.get("b", 0.0))
                    m.color.a = float(c.get("a", 1.0))
                    
                for p_data in m_data.get("points", []):
                    m.points.append(Point(x=float(p_data.get("x", 0.0)), y=float(p_data.get("y", 0.0)), z=float(p_data.get("z", 0.0))))
                    
                for c_data in m_data.get("colors", []):
                    m.colors.append(ColorRGBA(r=float(c_data.get("r", 0.0)), g=float(c_data.get("g", 0.0)), b=float(c_data.get("b", 0.0)), a=float(c_data.get("a", 1.0))))
                
                m.frame_locked = m_data.get("frame_locked", False)
                trajectory_msg.markers.append(m)
                
            self.trajectory_pub.publish(trajectory_msg)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a saved VSLAM path/odometry snapshot with current timestamps.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--path-topic", default="/visual_slam/tracking/slam_path")
    parser.add_argument("--odom-topic", default="/visual_slam/tracking/odometry")
    parser.add_argument("--landmarks-topic", default="")
    parser.add_argument("--trajectory-topic", default="")
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

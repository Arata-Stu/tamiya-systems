#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger 

from bag_manager_py.bag_manager_core import BagRecorderCore

class RosBagManagerNode(Node):
    """BagRecorderCoreの機能をROS 2通信経由で提供するノード。"""

    def __init__(self):
        super().__init__('ros2_bag_manager_node')

        self.declare_parameter('output_dir', 'rosbag2_output')
        self.declare_parameter('all_topics', True)
        self.declare_parameter('topics', ['/rosbag2_recorder/trigger'])
        self.declare_parameter('storage_id', 'mcap')
        
        # Coreインスタンスの初期化
        self.core = BagRecorderCore(
            output_dir=self.get_parameter('output_dir').value,
            topics=list(self.get_parameter('topics').value),
            all_topics=self.get_parameter('all_topics').value,
            storage_id=self.get_parameter('storage_id').value,
            logger=self.get_logger()
        )

        # Publisher
        self.status_pub = self.create_publisher(Bool, '~/status', 10)

        # Subscriptions
        self.create_subscription(Bool, '/rosbag2_recorder/trigger', self.trigger_cb, 10)
        self.create_subscription(String, '/rosbag2_recorder/memo', self.memo_cb, 10)

        # Services
        self.create_service(Trigger, '~/start_recording', self.start_srv_cb)
        self.create_service(Trigger, '~/stop_recording', self.stop_srv_cb)
        
        self.publish_status()

    def publish_status(self):
        """現在の録画状態をパブリッシュする。"""
        self.status_pub.publish(Bool(data=self.core.is_recording))

    def trigger_cb(self, msg: Bool):
        """トピックによる開始/停止制御。"""
        if msg.data:
            self.core.start()
        else:
            self.core.stop()
        self.publish_status()

    def memo_cb(self, msg: String):
        """トピックによる直前テイクへのメモ付与。"""
        memo = msg.data.strip().lower()
        if memo in ["good", "bad"]:
            self.core.apply_memo(memo)

    def start_srv_cb(self, request, response):
        """録画開始サービス。"""
        response.success, response.message = self.core.start()
        self.publish_status()
        return response

    def stop_srv_cb(self, request, response):
        """録画停止サービス。"""
        response.success, response.message = self.core.stop()
        self.publish_status()
        return response

    def destroy_node(self):
        """終了時に進行中の録画を停止する。"""
        self.core.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = RosBagManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()

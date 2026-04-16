#!/usr/bin/env python3

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from jetracer_driver.jetracer_core import JetRacerCore

class JetRacerDriverNode(Node):
    """JetRacerCoreをROS 2経由で操作するためのドライバノード。"""

    def __init__(self):
        super().__init__('jetracer_driver')

        # パラメータ宣言とCoreへの同期
        default_params = {
            "throttle_inversion": False,
            "steering_inversion": False,
            "steering_offset": 0.0,
            "throttle_offset": 0.0,
            "offset_step": 0.01,
            "throttle_gain": 1.0,
            "steering_gain": 1.0,
            "max_command_age": 0.5,
        }
        for name, default in default_params.items():
            self.declare_parameter(name, default)

        # Coreの初期化
        self.core = JetRacerCore(logger=self.get_logger())
        self._sync_all_params()

        # パラメータ変更コールバック
        self.add_on_set_parameters_callback(self._on_param_update)

        # QoS設定
        qos_cmd = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # 通信設定
        self.create_subscription(AckermannDriveStamped, '/control_cmd', self._cmd_cb, qos_cmd)
        self.create_subscription(Bool, '/steer_offset_inc', self._steer_offset_inc_cb, 10)
        self.create_subscription(Bool, '/steer_offset_dec', self._steer_offset_dec_cb, 10)
        self.create_subscription(Bool, '/speed_offset_inc', self._speed_offset_inc_cb, 10)
        self.create_subscription(Bool, '/speed_offset_dec', self._speed_offset_dec_cb, 10)

        # ウォッチドッグ
        self.last_cmd_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

    def _sync_all_params(self):
        """全パラメータをCoreに反映する。"""
        for name in self.core.params.keys():
            val = self.get_parameter(name).value
            self.core.update_param(name, val)

    def _on_param_update(self, params):
        """外部からのパラメータ変更をCoreに即時反映する。"""
        result = SetParametersResult(successful=True)
        for p in params:
            old_val = self.get_parameter(p.name).value if self.has_parameter(p.name) else None
            self.core.update_param(p.name, p.value)
            if old_val != p.value:
                if old_val is None:
                    self.get_logger().info(
                        f"Parameter set: {p.name} = {self._format_param_value(p.value)}"
                    )
                else:
                    self.get_logger().info(
                        f"Parameter updated: {p.name}: "
                        f"{self._format_param_value(old_val)} -> {self._format_param_value(p.value)}"
                    )
        return result

    def _cmd_cb(self, msg: AckermannDriveStamped):
        """受信したAckermann指令をCoreへ渡す。"""
        now = self.get_clock().now()
        msg_age = (now - Time.from_msg(msg.header.stamp)).nanoseconds / 1e9
        
        if msg_age > self.get_parameter('max_command_age').value:
            return

        self.core.set_drive(msg.drive.speed, msg.drive.steering_angle)
        self.last_cmd_time = now

    def _watchdog(self):
        """最後の指令から一定時間経過したら停止させる。"""
        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if dt > 1.0:
            self.core.stop()

    def _steer_offset_inc_cb(self, msg: Bool):
        if msg.data: self._shift_offset("steering_offset", 1)

    def _steer_offset_dec_cb(self, msg: Bool):
        if msg.data: self._shift_offset("steering_offset", -1)

    def _speed_offset_inc_cb(self, msg: Bool):
        if msg.data: self._shift_offset("throttle_offset", 1)

    def _speed_offset_dec_cb(self, msg: Bool):
        if msg.data: self._shift_offset("throttle_offset", -1)

    def _shift_offset(self, name: str, direction: int):
        """オフセット値をステップ分増減させてパラメータを更新する。"""
        step = float(self.get_parameter("offset_step").value)
        current_val = float(self.get_parameter(name).value)
        new_val = current_val + (step * direction)

        results = self.set_parameters([Parameter(name, Parameter.Type.DOUBLE, new_val)])
        if not results or not results[0].successful:
            reason = results[0].reason if results else "unknown reason"
            self.get_logger().error(
                f"Failed to update {name}: "
                f"{self._format_param_value(current_val)} -> "
                f"{self._format_param_value(new_val)} ({reason})"
            )

    @staticmethod
    def _format_param_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

def main(args=None):
    rclpy.init(args=args)
    node = JetRacerDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.core.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

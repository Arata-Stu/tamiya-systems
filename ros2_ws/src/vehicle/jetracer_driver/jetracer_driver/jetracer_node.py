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
            "max_throttle": 0.7,
            "max_throttle_slew_rate": 1.0,
            "max_command_age": 0.5,
            "neutral_stop_assist_enabled": True,
            "neutral_stop_required_steps": 10,
            "neutral_stop_reverse_start": -0.01,
            "neutral_stop_reverse_end": -0.08,
            "neutral_stop_reverse_step": -0.01,
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

        self._neutral_speed_count = 0
        self._neutral_stop_sequence = []
        self._neutral_stop_done = False

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
            if p.name in ("max_throttle", "max_throttle_slew_rate") and p.value < 0.0:
                result.successful = False
                result.reason = f"{p.name} must be >= 0.0"
                return result
            if p.name == "max_throttle" and p.value > 1.0:
                result.successful = False
                result.reason = "max_throttle must be <= 1.0"
                return result

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

        speed = self._apply_neutral_stop_assist(float(msg.drive.speed))
        self.core.set_drive(speed, msg.drive.steering_angle)
        self.last_cmd_time = now

    def _apply_neutral_stop_assist(self, speed: float) -> float:
        """ニュートラル継続時に短い後進パルスを入れ、ESCの停止判定を助ける。"""
        if not bool(self.get_parameter("neutral_stop_assist_enabled").value):
            self._reset_neutral_stop_assist()
            return speed

        if speed != 0.0:
            self._reset_neutral_stop_assist()
            return speed

        self._neutral_speed_count += 1

        if self._neutral_stop_sequence:
            return self._neutral_stop_sequence.pop(0)

        if self._neutral_stop_done:
            return 0.0

        required_steps = max(1, int(self.get_parameter("neutral_stop_required_steps").value))
        if self._neutral_speed_count < required_steps:
            return 0.0

        self._neutral_stop_sequence = self._build_neutral_stop_sequence()
        self._neutral_stop_done = True
        if self._neutral_stop_sequence:
            return self._neutral_stop_sequence.pop(0)
        return 0.0

    def _build_neutral_stop_sequence(self):
        start = float(self.get_parameter("neutral_stop_reverse_start").value)
        end = float(self.get_parameter("neutral_stop_reverse_end").value)
        step = float(self.get_parameter("neutral_stop_reverse_step").value)

        if step == 0.0:
            self.get_logger().warn("neutral_stop_reverse_step is 0.0; disabling stop assist pulse")
            return []
        if (end - start) * step < 0.0:
            self.get_logger().warn(
                "neutral_stop_reverse_step does not move start toward end; disabling stop assist pulse"
            )
            return []

        sequence = []
        value = start
        if step > 0.0:
            while value <= end:
                sequence.append(round(value, 6))
                value += step
        else:
            while value >= end:
                sequence.append(round(value, 6))
                value += step
        sequence.append(0.0)
        return sequence

    def _reset_neutral_stop_assist(self):
        self._neutral_speed_count = 0
        self._neutral_stop_sequence = []
        self._neutral_stop_done = False

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

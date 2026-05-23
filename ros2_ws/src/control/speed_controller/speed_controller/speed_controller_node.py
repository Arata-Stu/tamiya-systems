#!/usr/bin/env python3
"""Convert target speed [m/s] into JetRacer throttle with odometry feedback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32


@dataclass
class DriveCommand:
    msg: AckermannDriveStamped
    received_monotonic: float


@dataclass
class OdomState:
    speed_mps: float
    received_monotonic: float


class SpeedControllerNode(Node):
    """PID + feedforward speed controller for the final JetRacer command path."""

    def __init__(self) -> None:
        super().__init__("speed_controller")

        self._declare_parameters()
        self._load_parameters()
        self.add_on_set_parameters_callback(self._on_parameter_update)

        qos_latest = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._cmd_sub = self.create_subscription(
            AckermannDriveStamped, "input_drive", self._drive_callback, qos_latest
        )
        self._odom_sub = self.create_subscription(
            Odometry, "odometry", self._odom_callback, qos_latest
        )
        self._drive_pub = self.create_publisher(
            AckermannDriveStamped, "output_drive", qos_latest
        )

        self._target_pub = None
        self._measured_pub = None
        self._error_pub = None
        self._throttle_pub = None
        self._ensure_debug_publishers()

        self._last_command: DriveCommand | None = None
        self._last_odom: OdomState | None = None
        self._integral = 0.0
        self._previous_error: float | None = None
        self._filtered_derivative = 0.0
        self._last_throttle = 0.0
        self._last_control_monotonic = time.monotonic()

        period_sec = 1.0 / max(self.control_rate_hz, 1.0)
        self._timer = self.create_timer(period_sec, self._control_timer_callback)

        self.get_logger().info(
            "Speed controller ready: input_drive -> output_drive, "
            f"limit={self.target_speed_limit_mps:.2f}m/s, "
            f"max_throttle={self.max_throttle:.2f}, stale_odom={self.stale_odom_behavior}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("command_timeout_sec", 0.30)
        self.declare_parameter("odom_timeout_sec", 0.50)
        self.declare_parameter("stale_odom_behavior", "hold_feedforward")
        self.declare_parameter("target_speed_limit_mps", 6.0)
        self.declare_parameter("stop_speed_epsilon_mps", 0.03)
        self.declare_parameter("odom_speed_mode", "planar")
        self.declare_parameter("feedforward_gain", 0.14)
        self.declare_parameter("feedforward_offset", 0.0)
        self.declare_parameter("reverse_feedforward_gain", 0.14)
        self.declare_parameter("reverse_feedforward_offset", 0.0)
        self.declare_parameter("kp", 0.08)
        self.declare_parameter("ki", 0.02)
        self.declare_parameter("kd", 0.0)
        self.declare_parameter("integral_limit", 1.0)
        self.declare_parameter("derivative_filter_alpha", 0.35)
        self.declare_parameter("max_throttle", 0.70)
        self.declare_parameter("max_throttle_slew_rate", 1.0)
        self.declare_parameter("reset_integral_on_stop", True)
        self.declare_parameter("publish_debug", True)

    def _load_parameters(self) -> None:
        self.control_rate_hz = self._positive_param("control_rate_hz", 50.0)
        self.command_timeout_sec = self._nonnegative_param("command_timeout_sec")
        self.odom_timeout_sec = self._nonnegative_param("odom_timeout_sec")
        self.stale_odom_behavior = str(self.get_parameter("stale_odom_behavior").value)
        self.target_speed_limit_mps = self._nonnegative_param("target_speed_limit_mps")
        self.stop_speed_epsilon_mps = self._nonnegative_param("stop_speed_epsilon_mps")
        self.odom_speed_mode = str(self.get_parameter("odom_speed_mode").value)
        self.feedforward_gain = self._nonnegative_param("feedforward_gain")
        self.feedforward_offset = self._nonnegative_param("feedforward_offset")
        self.reverse_feedforward_gain = self._nonnegative_param("reverse_feedforward_gain")
        self.reverse_feedforward_offset = self._nonnegative_param("reverse_feedforward_offset")
        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.kd = float(self.get_parameter("kd").value)
        self.integral_limit = self._nonnegative_param("integral_limit")
        self.derivative_filter_alpha = min(
            max(float(self.get_parameter("derivative_filter_alpha").value), 0.0), 1.0
        )
        self.max_throttle = min(max(self._nonnegative_param("max_throttle"), 0.0), 1.0)
        self.max_throttle_slew_rate = self._nonnegative_param("max_throttle_slew_rate")
        self.reset_integral_on_stop = bool(self.get_parameter("reset_integral_on_stop").value)
        self.publish_debug = bool(self.get_parameter("publish_debug").value)

        if self.stale_odom_behavior not in {"stop", "hold_feedforward"}:
            self.get_logger().warn(
                "stale_odom_behavior must be stop or hold_feedforward; using stop"
            )
            self.stale_odom_behavior = "stop"
        if self.odom_speed_mode not in {"signed_planar", "planar", "linear_x"}:
            self.get_logger().warn(
                "odom_speed_mode must be signed_planar, planar, or linear_x; using signed_planar"
            )
            self.odom_speed_mode = "signed_planar"

    def _positive_param(self, name: str, fallback: float) -> float:
        return self._positive_value(self.get_parameter(name).value, fallback)

    def _nonnegative_param(self, name: str) -> float:
        return self._nonnegative_value(self.get_parameter(name).value)

    @staticmethod
    def _positive_value(raw_value, fallback: float) -> float:
        value = float(raw_value)
        return value if value > 0.0 and math.isfinite(value) else fallback

    @staticmethod
    def _nonnegative_value(raw_value) -> float:
        value = float(raw_value)
        if not math.isfinite(value):
            return 0.0
        return max(0.0, value)

    def _on_parameter_update(self, params) -> SetParametersResult:
        for param in params:
            if param.name in {
                "control_rate_hz",
                "command_timeout_sec",
                "odom_timeout_sec",
                "target_speed_limit_mps",
                "stop_speed_epsilon_mps",
                "feedforward_gain",
                "feedforward_offset",
                "reverse_feedforward_gain",
                "reverse_feedforward_offset",
                "integral_limit",
                "derivative_filter_alpha",
                "max_throttle",
                "max_throttle_slew_rate",
            } and float(param.value) < 0.0:
                return SetParametersResult(
                    successful=False, reason=f"{param.name} must be >= 0.0"
                )
            if param.name == "max_throttle" and float(param.value) > 1.0:
                return SetParametersResult(
                    successful=False, reason="max_throttle must be <= 1.0"
                )
            if param.name == "stale_odom_behavior" and str(param.value) not in {
                "stop",
                "hold_feedforward",
            }:
                return SetParametersResult(
                    successful=False,
                    reason="stale_odom_behavior must be stop or hold_feedforward",
                )
            if param.name == "odom_speed_mode" and str(param.value) not in {
                "signed_planar",
                "planar",
                "linear_x",
            }:
                return SetParametersResult(
                    successful=False,
                    reason="odom_speed_mode must be signed_planar, planar, or linear_x",
                )

        for param in params:
            self._apply_runtime_parameter(param.name, param.value)
        return SetParametersResult(successful=True)

    def _apply_runtime_parameter(self, name: str, value) -> None:
        if name == "control_rate_hz":
            self.control_rate_hz = self._positive_value(value, self.control_rate_hz)
        elif name == "command_timeout_sec":
            self.command_timeout_sec = self._nonnegative_value(value)
        elif name == "odom_timeout_sec":
            self.odom_timeout_sec = self._nonnegative_value(value)
        elif name == "stale_odom_behavior":
            self.stale_odom_behavior = str(value)
        elif name == "target_speed_limit_mps":
            self.target_speed_limit_mps = self._nonnegative_value(value)
        elif name == "stop_speed_epsilon_mps":
            self.stop_speed_epsilon_mps = self._nonnegative_value(value)
        elif name == "odom_speed_mode":
            self.odom_speed_mode = str(value)
        elif name == "feedforward_gain":
            self.feedforward_gain = self._nonnegative_value(value)
        elif name == "feedforward_offset":
            self.feedforward_offset = self._nonnegative_value(value)
        elif name == "reverse_feedforward_gain":
            self.reverse_feedforward_gain = self._nonnegative_value(value)
        elif name == "reverse_feedforward_offset":
            self.reverse_feedforward_offset = self._nonnegative_value(value)
        elif name == "kp":
            self.kp = float(value)
        elif name == "ki":
            self.ki = float(value)
        elif name == "kd":
            self.kd = float(value)
        elif name == "integral_limit":
            self.integral_limit = self._nonnegative_value(value)
        elif name == "derivative_filter_alpha":
            self.derivative_filter_alpha = min(max(float(value), 0.0), 1.0)
        elif name == "max_throttle":
            self.max_throttle = min(max(self._nonnegative_value(value), 0.0), 1.0)
        elif name == "max_throttle_slew_rate":
            self.max_throttle_slew_rate = self._nonnegative_value(value)
        elif name == "reset_integral_on_stop":
            self.reset_integral_on_stop = bool(value)
        elif name == "publish_debug":
            self.publish_debug = bool(value)
            self._ensure_debug_publishers()

    def _ensure_debug_publishers(self) -> None:
        if not self.publish_debug or self._target_pub is not None:
            return
        self._target_pub = self.create_publisher(Float32, "~/target_speed_mps", 10)
        self._measured_pub = self.create_publisher(Float32, "~/measured_speed_mps", 10)
        self._error_pub = self.create_publisher(Float32, "~/speed_error_mps", 10)
        self._throttle_pub = self.create_publisher(Float32, "~/throttle_cmd", 10)

    def _drive_callback(self, msg: AckermannDriveStamped) -> None:
        self._last_command = DriveCommand(msg=msg, received_monotonic=time.monotonic())

    def _odom_callback(self, msg: Odometry) -> None:
        self._last_odom = OdomState(
            speed_mps=self._speed_from_odom(msg), received_monotonic=time.monotonic()
        )

    def _speed_from_odom(self, msg: Odometry) -> float:
        vx = float(msg.twist.twist.linear.x)
        vy = float(msg.twist.twist.linear.y)
        if self.odom_speed_mode == "linear_x":
            return vx
        planar = math.hypot(vx, vy)
        if self.odom_speed_mode == "planar":
            return planar
        if abs(vx) > 1.0e-3:
            return math.copysign(planar, vx)
        if self._last_command is not None:
            target = float(self._last_command.msg.drive.speed)
            if abs(target) > self.stop_speed_epsilon_mps:
                return math.copysign(planar, target)
        return planar

    def _control_timer_callback(self) -> None:
        now = time.monotonic()
        dt = max(1.0e-3, now - self._last_control_monotonic)
        self._last_control_monotonic = now

        target_msg = self._fresh_target(now)
        if target_msg is None:
            self._publish_drive(self._build_output_msg(None, 0.0), 0.0, 0.0, 0.0)
            self._reset_pid_if_stopped()
            return

        target_speed = self._clamp_target_speed(float(target_msg.drive.speed))
        measured_speed = self._measured_speed_or_nan(now)

        if abs(target_speed) <= self.stop_speed_epsilon_mps:
            self._reset_pid_if_stopped()
            self._publish_drive(self._build_output_msg(target_msg, 0.0), target_speed, measured_speed, 0.0)
            return

        feedforward = self._feedforward_throttle(target_speed)
        if not math.isfinite(measured_speed):
            if self.stale_odom_behavior == "stop":
                self._reset_pid_if_stopped()
                self._publish_drive(self._build_output_msg(target_msg, 0.0), target_speed, measured_speed, 0.0)
                return
            throttle = feedforward
            error = 0.0
        else:
            error = target_speed - measured_speed
            self._integral = self._clamp_integral(self._integral + error * dt)
            derivative = self._filtered_error_derivative(error, dt)
            throttle = feedforward + self.kp * error + self.ki * self._integral + self.kd * derivative

        throttle = self._apply_throttle_limits(throttle, dt)
        self._publish_drive(
            self._build_output_msg(target_msg, throttle),
            target_speed,
            measured_speed,
            throttle,
        )

    def _fresh_target(self, now: float) -> AckermannDriveStamped | None:
        if self._last_command is None:
            return None
        age = now - self._last_command.received_monotonic
        if age > self.command_timeout_sec:
            return None
        return self._last_command.msg

    def _clamp_target_speed(self, speed_mps: float) -> float:
        if not math.isfinite(speed_mps):
            return 0.0
        limit = max(0.0, self.target_speed_limit_mps)
        return min(max(speed_mps, -limit), limit)

    def _measured_speed_or_nan(self, now: float) -> float:
        if self._last_odom is None:
            return math.nan
        if now - self._last_odom.received_monotonic > self.odom_timeout_sec:
            return math.nan
        return self._last_odom.speed_mps

    def _feedforward_throttle(self, target_speed_mps: float) -> float:
        sign = 1.0 if target_speed_mps >= 0.0 else -1.0
        speed_abs = abs(target_speed_mps)
        if sign >= 0.0:
            magnitude = self.feedforward_gain * speed_abs + self.feedforward_offset
        else:
            magnitude = self.reverse_feedforward_gain * speed_abs + self.reverse_feedforward_offset
        return sign * magnitude

    def _clamp_integral(self, value: float) -> float:
        if self.integral_limit <= 0.0:
            return 0.0
        return min(max(value, -self.integral_limit), self.integral_limit)

    def _filtered_error_derivative(self, error: float, dt: float) -> float:
        if self._previous_error is None:
            self._previous_error = error
            return 0.0
        raw_derivative = (error - self._previous_error) / max(dt, 1.0e-3)
        self._previous_error = error
        alpha = self.derivative_filter_alpha
        self._filtered_derivative = (
            alpha * raw_derivative + (1.0 - alpha) * self._filtered_derivative
        )
        return self._filtered_derivative

    def _apply_throttle_limits(self, throttle: float, dt: float) -> float:
        if not math.isfinite(throttle):
            throttle = 0.0
        throttle = min(max(throttle, -self.max_throttle), self.max_throttle)
        if self.max_throttle_slew_rate > 0.0:
            max_step = self.max_throttle_slew_rate * dt
            delta = throttle - self._last_throttle
            if abs(delta) > max_step:
                throttle = self._last_throttle + math.copysign(max_step, delta)
        self._last_throttle = throttle
        return throttle

    def _reset_pid_if_stopped(self) -> None:
        if self.reset_integral_on_stop:
            self._integral = 0.0
        self._previous_error = None
        self._filtered_derivative = 0.0
        self._last_throttle = 0.0

    def _build_output_msg(
        self, target_msg: AckermannDriveStamped | None, throttle: float
    ) -> AckermannDriveStamped:
        out = AckermannDriveStamped()
        if target_msg is not None:
            out.header = target_msg.header
            out.header.stamp = self.get_clock().now().to_msg()
            out.drive.steering_angle = target_msg.drive.steering_angle
            out.drive.steering_angle_velocity = target_msg.drive.steering_angle_velocity
            out.drive.speed = float(throttle)
            out.drive.acceleration = target_msg.drive.acceleration
            out.drive.jerk = target_msg.drive.jerk
        else:
            out.header.stamp = self.get_clock().now().to_msg()
            out.drive.speed = 0.0
            out.drive.steering_angle = 0.0
        return out

    def _publish_drive(
        self,
        msg: AckermannDriveStamped,
        target_speed_mps: float,
        measured_speed_mps: float,
        throttle: float,
    ) -> None:
        self._drive_pub.publish(msg)
        if not self.publish_debug:
            return
        self._publish_float(self._target_pub, target_speed_mps)
        self._publish_float(self._measured_pub, measured_speed_mps)
        error = (
            target_speed_mps - measured_speed_mps
            if math.isfinite(measured_speed_mps)
            else math.nan
        )
        self._publish_float(self._error_pub, error)
        self._publish_float(self._throttle_pub, throttle)

    @staticmethod
    def _publish_float(pub, value: float) -> None:
        if pub is None:
            return
        msg = Float32()
        msg.data = float(value)
        pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SpeedControllerNode()
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

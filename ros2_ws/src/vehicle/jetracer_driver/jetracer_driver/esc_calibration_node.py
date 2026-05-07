#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from jetracer_driver.jetracer_core import JetRacerCore


class EscCalibrationNode(Node):
    """Send exact normalized throttle values for ESC calibration."""

    def __init__(self):
        super().__init__("esc_calibration_node")

        self.declare_parameter("initial_delay", 3.0)
        self.declare_parameter("max_throttle_duration", 4.0)
        self.declare_parameter("min_throttle_duration", 4.0)
        self.declare_parameter("neutral_duration", 2.0)
        self.declare_parameter("steering", 0.0)
        self.declare_parameter("throttle_inversion", False)
        self.declare_parameter("throttle_gain", 1.0)
        self.declare_parameter("throttle_offset", 0.0)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("auto_shutdown", True)
        self.declare_parameter("armed", False)

        if not bool(self.get_parameter("armed").value):
            self.get_logger().error(
                "ESC calibration is not armed. Restart with armed:=true after lifting wheels "
                "and preparing the ESC set button."
            )
            raise RuntimeError("ESC calibration node requires armed:=true")

        self.core = JetRacerCore(logger=self.get_logger())
        self._configure_core()

        self.sequence = [
            ("initial_delay", 0.0, self._param_float("initial_delay")),
            ("max_throttle", 1.0, self._param_float("max_throttle_duration")),
            ("min_throttle", -1.0, self._param_float("min_throttle_duration")),
            ("neutral", 0.0, self._param_float("neutral_duration")),
        ]
        self.sequence_index = 0
        self.step_started_at = self.get_clock().now()
        self.last_step_name = None

        rate = max(1.0, self._param_float("publish_rate"))
        self.timer = self.create_timer(1.0 / rate, self._on_timer)

        self.get_logger().warn(
            "ESC calibration armed. Keep wheels off the ground. "
            "Sequence: delay -> +1.0 throttle -> -1.0 throttle -> neutral."
        )

    def _configure_core(self):
        self.core.update_param("throttle_inversion", bool(self.get_parameter("throttle_inversion").value))
        self.core.update_param("steering_inversion", False)
        self.core.update_param("steering_offset", 0.0)
        self.core.update_param("throttle_offset", self._param_float("throttle_offset"))
        self.core.update_param("throttle_gain", self._param_float("throttle_gain"))
        self.core.update_param("steering_gain", 1.0)
        self.core.update_param("max_throttle", 1.0)
        self.core.update_param("max_throttle_slew_rate", 0.0)

    def _on_timer(self):
        if self.sequence_index >= len(self.sequence):
            self.core.stop()
            self.get_logger().info("ESC calibration sequence complete. Output is neutral.")
            self.timer.cancel()
            if bool(self.get_parameter("auto_shutdown").value):
                rclpy.shutdown()
            return

        name, throttle, duration = self.sequence[self.sequence_index]
        if name != self.last_step_name:
            self.get_logger().info(
                f"ESC calibration step: {name}, throttle={throttle:.1f}, duration={duration:.1f}s"
            )
            self.last_step_name = name

        steering = self._param_float("steering")
        self.core.set_drive(throttle, steering)

        elapsed = (self.get_clock().now() - self.step_started_at).nanoseconds / 1e9
        if elapsed >= duration:
            self.sequence_index += 1
            self.step_started_at = self.get_clock().now()

    def _param_float(self, name: str) -> float:
        return float(self.get_parameter(name).value)

    def destroy_node(self):
        if hasattr(self, "core"):
            self.core.stop()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EscCalibrationNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

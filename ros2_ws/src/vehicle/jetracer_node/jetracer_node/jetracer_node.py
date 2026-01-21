import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.time import Time
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from jetracer.nvidia_racecar import NvidiaRacecar


class JetRacerDriver(Node):
    def __init__(self):
        super().__init__('jetracer_driver')

        self.params = {
            "throttle_inversion": False,
            "steering_inversion": False,
            "steering_offset": 0.0,
            "throttle_offset": 0.0,
            "offset_step": 0.01,
            "throttle_gain": 1.0,
            "steering_gain": 1.0,
            "max_command_age": 0.5,
        }

        for name, default in self.params.items():
            self.declare_parameter(name, default)

        self.add_on_set_parameters_callback(self._on_param_update)

        try:
            self.car = NvidiaRacecar()
            self.car.throttle = 0.0
            self.car.steering = 0.0
        except Exception as e:
            self.get_logger().error(f"❌ Failed to initialize NvidiaRacecar: {e}")
            raise SystemExit(1)

        qos_cmd = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        qos_button = QoSProfile(depth=5)

        self.create_subscription(AckermannDriveStamped, '/cmd_drive', self._cmd_cb, qos_cmd)

        self.create_subscription(Bool, '/steer_offset_inc', self._steer_offset_inc_cb, qos_button)
        self.create_subscription(Bool, '/steer_offset_dec', self._steer_offset_dec_cb, qos_button)
        self.create_subscription(Bool, '/speed_offset_inc', self._speed_offset_inc_cb, qos_button)
        self.create_subscription(Bool, '/speed_offset_dec', self._speed_offset_dec_cb, qos_button)

        self.last_cmd_time = self.get_clock().now()
        self.create_timer(0.1, self._watchdog)

        self.get_logger().info('✅ JetRacer driver started, waiting for /cmd_drive')

    def _on_param_update(self, params):
        for p in params:
            if p.name in self.params:
                self.params[p.name] = p.value
                self.get_logger().info(f"Parameter updated: {p.name} = {p.value}")
        return rclpy.parameter.ParameterValue()

    def _cmd_cb(self, msg: AckermannDriveStamped):
        now = self.get_clock().now()
        msg_stamp = Time.from_msg(msg.header.stamp)
        age = (now - msg_stamp).nanoseconds / 1e9
        max_age = self.params["max_command_age"]

        if age > max_age:
            self.get_logger().warn(f"⚠️ Command too old ({age:.3f}s > {max_age:.3f}s), ignored.")
            return

        throttle_inversion = self.params["throttle_inversion"]
        steering_inversion = self.params["steering_inversion"]
        throttle_gain = self.params["throttle_gain"]
        steering_gain = self.params["steering_gain"]
        throttle_offset = self.params["throttle_offset"]
        steering_offset = self.params["steering_offset"]

        speed = msg.drive.speed
        steering_angle = msg.drive.steering_angle

        throttle = (speed * throttle_gain) + throttle_offset
        if throttle_inversion:
            throttle *= -1.0
        throttle = max(min(throttle, 1.0), -1.0)

        steering = (steering_angle * steering_gain) + steering_offset
        if steering_inversion:
            steering *= -1.0
        steering = max(min(steering, 1.0), -1.0)

        self.car.throttle = float(throttle)
        self.car.steering = float(steering)
        self.last_cmd_time = now

    def _watchdog(self):
        dt = (self.get_clock().now() - self.last_cmd_time).nanoseconds / 1e9
        if dt > 1.0:
            self.car.throttle = 0.0
            self.car.steering = 0.0

    def _steer_offset_inc_cb(self, msg: Bool):
        if msg.data:
            step = self.params["offset_step"]
            new_val = self.params["steering_offset"] + step
            self._update_param("steering_offset", new_val)

    def _steer_offset_dec_cb(self, msg: Bool):
        if msg.data:
            step = self.params["offset_step"]
            new_val = self.params["steering_offset"] - step
            self._update_param("steering_offset", new_val)

    def _speed_offset_inc_cb(self, msg: Bool):
        if msg.data:
            step = self.params["offset_step"]
            new_val = self.params["throttle_offset"] + step
            self._update_param("throttle_offset", new_val)

    def _speed_offset_dec_cb(self, msg: Bool):
        if msg.data:
            step = self.params["offset_step"]
            new_val = self.params["throttle_offset"] - step
            self._update_param("throttle_offset", new_val)

    def _update_param(self, name: str, value: float):
        self.params[name] = value
        self.set_parameters([Parameter(name, Parameter.Type.DOUBLE, value)])
        self.get_logger().info(f"{name} updated to {value:.3f}")


def main():
    rclpy.init()
    node = JetRacerDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🔚 JetRacer driver stopped.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

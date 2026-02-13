#include "control_filter/control_filter_node.hpp"

ControlFilterNode::ControlFilterNode() : Node("control_filter_node") {
  // --- 制御パラメータの宣言 ---
  this->declare_parameter<std::string>("filter_type", "slew_rate");
  this->declare_parameter<int>("window_size", 5);
  this->declare_parameter<double>("max_speed_slew_rate", 2.0);
  this->declare_parameter<double>("max_steer_slew_rate", 1.5);

  // Scale Filter パラメータ
  this->declare_parameter<bool>("use_scale_filter", true);
  this->declare_parameter<double>("straight_steer_threshold", 0.1);
  this->declare_parameter<double>("straight_speed_scale_ratio", 1.0);
  this->declare_parameter<double>("cornering_speed_scale_ratio", 0.5);
  this->declare_parameter<double>("steer_scale_ratio", 1.0);

  // 初期のパラメータ取得
  ControlFilterParams params;
  this->get_parameter("filter_type", params.filter_type);
  this->get_parameter("window_size", params.window_size);
  this->get_parameter("max_speed_slew_rate", params.max_speed_slew_rate);
  this->get_parameter("max_steer_slew_rate", params.max_steer_slew_rate);

  this->get_parameter("use_scale_filter", params.use_scale_filter);
  this->get_parameter("straight_steer_threshold",
                      params.straight_steer_threshold);
  this->get_parameter("straight_speed_scale_ratio",
                      params.straight_speed_scale_ratio);
  this->get_parameter("cornering_speed_scale_ratio",
                      params.cornering_speed_scale_ratio);
  this->get_parameter("steer_scale_ratio", params.steer_scale_ratio);

  core_.SetParams(params);

  // --- Pub/Sub の設定 ---
  publisher_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_filtered", rclcpp::QoS(10));

  subscription_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_raw", rclcpp::QoS(1),
          std::bind(&ControlFilterNode::TopicCallback, this,
                    std::placeholders::_1));

  // 動的パラメータのコールバック登録
  parameters_callback_handle_ = this->add_on_set_parameters_callback(std::bind(
      &ControlFilterNode::ParametersCallback, this, std::placeholders::_1));

  last_callback_time_ = this->now();
  PrintParameters();
}

void ControlFilterNode::TopicCallback(
    const ackermann_msgs::msg::AckermannDriveStamped::SharedPtr msg) {
  const rclcpp::Time now = this->now();
  const double dt = (now - last_callback_time_).seconds();
  last_callback_time_ = now;

  ackermann_msgs::msg::AckermannDrive filtered_drive =
      core_.Filter(msg->drive, dt);

  ackermann_msgs::msg::AckermannDriveStamped out_msg = *msg;
  out_msg.drive = filtered_drive;
  publisher_->publish(out_msg);
}

rcl_interfaces::msg::SetParametersResult ControlFilterNode::ParametersCallback(
    const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  ControlFilterParams params = core_.GetParams();

  for (const auto &param : parameters) {
    if (param.get_name() == "filter_type") {
      params.filter_type = param.as_string();
    } else if (param.get_name() == "window_size") {
      params.window_size = param.as_int();
    } else if (param.get_name() == "max_speed_slew_rate") {
      params.max_speed_slew_rate = param.as_double();
    } else if (param.get_name() == "max_steer_slew_rate") {
      params.max_steer_slew_rate = param.as_double();
    } else if (param.get_name() == "use_scale_filter") {
      params.use_scale_filter = param.as_bool();
    } else if (param.get_name() == "straight_steer_threshold") {
      params.straight_steer_threshold = param.as_double();
    } else if (param.get_name() == "straight_speed_scale_ratio") {
      params.straight_speed_scale_ratio = param.as_double();
    } else if (param.get_name() == "cornering_speed_scale_ratio") {
      params.cornering_speed_scale_ratio = param.as_double();
    } else if (param.get_name() == "steer_scale_ratio") {
      params.steer_scale_ratio = param.as_double();
    }
  }

  core_.SetParams(params);
  PrintParameters();
  return result;
}

void ControlFilterNode::PrintParameters() const {
  const auto &params = core_.GetParams();
  RCLCPP_INFO(this->get_logger(), "--- Control Filter Parameters ---");
  RCLCPP_INFO(this->get_logger(), "filter_type: %s",
              params.filter_type.c_str());
  RCLCPP_INFO(this->get_logger(), "window_size: %d", params.window_size);
  RCLCPP_INFO(this->get_logger(), "max_speed_slew_rate: %.2f",
              params.max_speed_slew_rate);
  RCLCPP_INFO(this->get_logger(), "max_steer_slew_rate: %.2f",
              params.max_steer_slew_rate);

  RCLCPP_INFO(this->get_logger(), "use_scale_filter: %s",
              params.use_scale_filter ? "true" : "false");
  if (params.use_scale_filter) {
    RCLCPP_INFO(this->get_logger(), "  straight_steer_threshold: %.2f",
                params.straight_steer_threshold);
    RCLCPP_INFO(this->get_logger(), "  straight_speed_scale: %.2f",
                params.straight_speed_scale_ratio);
    RCLCPP_INFO(this->get_logger(), "  cornering_speed_scale: %.2f",
                params.cornering_speed_scale_ratio);
    RCLCPP_INFO(this->get_logger(), "  steer_scale: %.2f",
                params.steer_scale_ratio);
  }
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControlFilterNode>());
  rclcpp::shutdown();
  return 0;
}
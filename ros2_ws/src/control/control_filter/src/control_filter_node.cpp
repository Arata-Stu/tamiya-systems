#include "control_filter/control_filter_node.hpp"
#include <set>

ControlFilterNode::ControlFilterNode() : Node("control_filter_node") {
  // --- Section/Class Names 宣言 ---
  this->declare_parameter<std::vector<std::string>>("section_names", std::vector<std::string>());
  this->declare_parameter<std::vector<std::string>>("section_classes", std::vector<std::string>());
  this->get_parameter("section_names", section_names_);
  this->get_parameter("section_classes", section_classes_);

  // マッピングの構築とユニークなクラス名の抽出
  std::set<std::string> unique_classes;
  for (size_t i = 0; i < section_names_.size() && i < section_classes_.size(); ++i) {
    section_to_class_map_[section_names_[i]] = section_classes_[i];
    unique_classes.insert(section_classes_[i]);
  }

  if (section_names_.size() != section_classes_.size()) {
    RCLCPP_WARN(this->get_logger(), "Length of section_names and section_classes do not match!");
  }

  // --- 制御パラメータの宣言 (デフォルト) ---
  this->declare_parameter<std::string>("filter_type", "slew_rate");
  this->declare_parameter<int>("window_size", 5);
  this->declare_parameter<double>("max_speed_slew_rate", 2.0);
  this->declare_parameter<double>("max_steer_slew_rate", 1.5);

  this->declare_parameter<bool>("use_scale_filter", true);
  this->declare_parameter<double>("straight_steer_threshold", 0.1);
  this->declare_parameter<double>("straight_speed_scale_ratio", 1.0);
  this->declare_parameter<double>("cornering_speed_scale_ratio", 0.5);
  this->declare_parameter<double>("steer_scale_ratio", 1.0);

  // 初期のパラメータ取得 (デフォルト)
  this->get_parameter("filter_type", default_params_.filter_type);
  this->get_parameter("window_size", default_params_.window_size);
  this->get_parameter("max_speed_slew_rate", default_params_.max_speed_slew_rate);
  this->get_parameter("max_steer_slew_rate", default_params_.max_steer_slew_rate);
  this->get_parameter("use_scale_filter", default_params_.use_scale_filter);
  this->get_parameter("straight_steer_threshold", default_params_.straight_steer_threshold);
  this->get_parameter("straight_speed_scale_ratio", default_params_.straight_speed_scale_ratio);
  this->get_parameter("cornering_speed_scale_ratio", default_params_.cornering_speed_scale_ratio);
  this->get_parameter("steer_scale_ratio", default_params_.steer_scale_ratio);

  // --- クラス固有パラメータの宣言と取得 ---
  for (const auto& cls : unique_classes) {
    ControlFilterParams p;
    std::string prefix = cls + ".";
    
    this->declare_parameter<std::string>(prefix + "filter_type", default_params_.filter_type);
    this->declare_parameter<int>(prefix + "window_size", default_params_.window_size);
    this->declare_parameter<double>(prefix + "max_speed_slew_rate", default_params_.max_speed_slew_rate);
    this->declare_parameter<double>(prefix + "max_steer_slew_rate", default_params_.max_steer_slew_rate);
    
    this->declare_parameter<bool>(prefix + "use_scale_filter", default_params_.use_scale_filter);
    this->declare_parameter<double>(prefix + "straight_steer_threshold", default_params_.straight_steer_threshold);
    this->declare_parameter<double>(prefix + "straight_speed_scale_ratio", default_params_.straight_speed_scale_ratio);
    this->declare_parameter<double>(prefix + "cornering_speed_scale_ratio", default_params_.cornering_speed_scale_ratio);
    this->declare_parameter<double>(prefix + "steer_scale_ratio", default_params_.steer_scale_ratio);

    this->get_parameter(prefix + "filter_type", p.filter_type);
    this->get_parameter(prefix + "window_size", p.window_size);
    this->get_parameter(prefix + "max_speed_slew_rate", p.max_speed_slew_rate);
    this->get_parameter(prefix + "max_steer_slew_rate", p.max_steer_slew_rate);
    this->get_parameter(prefix + "use_scale_filter", p.use_scale_filter);
    this->get_parameter(prefix + "straight_steer_threshold", p.straight_steer_threshold);
    this->get_parameter(prefix + "straight_speed_scale_ratio", p.straight_speed_scale_ratio);
    this->get_parameter(prefix + "cornering_speed_scale_ratio", p.cornering_speed_scale_ratio);
    this->get_parameter(prefix + "steer_scale_ratio", p.steer_scale_ratio);
    
    class_params_[cls] = p;
  }

  // 初期設定の適用
  core_.SetParams(default_params_);
  current_section_ = "default";
  current_class_ = "default";
  PrintParameters("default", default_params_);

  // --- Pub/Sub の設定 ---
  publisher_ =
      this->create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_filtered", rclcpp::QoS(10));

  subscription_ =
      this->create_subscription<ackermann_msgs::msg::AckermannDriveStamped>(
          "control_cmd_raw", rclcpp::QoS(1),
          std::bind(&ControlFilterNode::TopicCallback, this,
                    std::placeholders::_1));

  section_sub_ =
      this->create_subscription<std_msgs::msg::String>(
          "/localization/current_section", rclcpp::QoS(1),
          std::bind(&ControlFilterNode::SectionCallback, this,
                    std::placeholders::_1));

  // 動的パラメータのコールバック登録
  parameters_callback_handle_ = this->add_on_set_parameters_callback(std::bind(
      &ControlFilterNode::ParametersCallback, this, std::placeholders::_1));

  last_callback_time_ = this->now();
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

void ControlFilterNode::SectionCallback(
    const std_msgs::msg::String::SharedPtr msg) {
  std::string new_section = msg->data;
  if (new_section != current_section_) {
    current_section_ = new_section;
    std::string new_class = "default";
    
    if (section_to_class_map_.find(current_section_) != section_to_class_map_.end()) {
      new_class = section_to_class_map_[current_section_];
    }
    
    if (new_class != current_class_) {
      current_class_ = new_class;
      if (class_params_.find(current_class_) != class_params_.end()) {
        core_.SetParams(class_params_[current_class_]);
        RCLCPP_INFO(this->get_logger(), "Section '%s' maps to class '%s'. Applying class parameters.", current_section_.c_str(), current_class_.c_str());
        PrintParameters(current_class_, class_params_[current_class_]);
      } else {
        core_.SetParams(default_params_);
        RCLCPP_INFO(this->get_logger(), "Section '%s' maps to class '%s' (Not configured). Applying default parameters.", current_section_.c_str(), current_class_.c_str());
        PrintParameters("default", default_params_);
      }
    } else {
      RCLCPP_INFO(this->get_logger(), "Section changed to '%s', class is still '%s'.", current_section_.c_str(), current_class_.c_str());
    }
  }
}

rcl_interfaces::msg::SetParametersResult ControlFilterNode::ParametersCallback(
    const std::vector<rclcpp::Parameter> &parameters) {
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  for (const auto &param : parameters) {
    std::string name = param.get_name();
    
    // Check if parameter belongs to a specific class or default
    std::string class_name = "default";
    std::string param_name = name;
    
    size_t dot_pos = name.find('.');
    if (dot_pos != std::string::npos) {
      class_name = name.substr(0, dot_pos);
      param_name = name.substr(dot_pos + 1);
    }
    
    ControlFilterParams* target_params = &default_params_;
    if (class_name != "default") {
      if (class_params_.find(class_name) != class_params_.end()) {
        target_params = &class_params_[class_name];
      } else {
        continue; // Unknown class parameter dynamically added? Ignore for now.
      }
    }
    
    if (param_name == "filter_type") {
      target_params->filter_type = param.as_string();
    } else if (param_name == "window_size") {
      target_params->window_size = param.as_int();
    } else if (param_name == "max_speed_slew_rate") {
      target_params->max_speed_slew_rate = param.as_double();
    } else if (param_name == "max_steer_slew_rate") {
      target_params->max_steer_slew_rate = param.as_double();
    } else if (param_name == "use_scale_filter") {
      target_params->use_scale_filter = param.as_bool();
    } else if (param_name == "straight_steer_threshold") {
      target_params->straight_steer_threshold = param.as_double();
    } else if (param_name == "straight_speed_scale_ratio") {
      target_params->straight_speed_scale_ratio = param.as_double();
    } else if (param_name == "cornering_speed_scale_ratio") {
      target_params->cornering_speed_scale_ratio = param.as_double();
    } else if (param_name == "steer_scale_ratio") {
      target_params->steer_scale_ratio = param.as_double();
    }
    
    // If the changed parameter belongs to the currently active class (or default if no valid class is active), update core
    bool should_update_core = false;
    if (current_class_ == class_name) {
      should_update_core = true;
    } else if (class_name == "default" && class_params_.find(current_class_) == class_params_.end()) {
      should_update_core = true;
    }
    
    if (should_update_core) {
      core_.SetParams(*target_params);
      PrintParameters(current_class_, *target_params);
    }
  }

  return result;
}

void ControlFilterNode::PrintParameters(const std::string& class_name, const ControlFilterParams& params) const {
  RCLCPP_INFO(this->get_logger(), "--- Control Filter Parameters [%s] ---", class_name.c_str());
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
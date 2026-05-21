#include <algorithm>
#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/string.hpp"

namespace {

std::string NormalizeToken(const std::string &value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const unsigned char c : value) {
    normalized.push_back(static_cast<char>(std::tolower(c)));
  }
  return normalized;
}

enum class SectionPolicy {
  kAllowAvoid,
  kFollowOnly,
  kNormalOnly,
};

const char *PolicyToString(const SectionPolicy policy) {
  switch (policy) {
    case SectionPolicy::kAllowAvoid:
      return "allow_avoid";
    case SectionPolicy::kFollowOnly:
      return "follow_only";
    case SectionPolicy::kNormalOnly:
      return "normal_only";
  }
  return "allow_avoid";
}

bool ParsePolicy(const std::string &value, SectionPolicy &policy) {
  const std::string token = NormalizeToken(value);
  if (token == "allow_avoid" || token == "allow-avoid" || token == "allowavoid" ||
      token == "avoid") {
    policy = SectionPolicy::kAllowAvoid;
    return true;
  }
  if (token == "follow_only" || token == "follow-only" || token == "followonly" ||
      token == "follow") {
    policy = SectionPolicy::kFollowOnly;
    return true;
  }
  if (token == "normal_only" || token == "normal-only" || token == "normalonly" ||
      token == "normal") {
    policy = SectionPolicy::kNormalOnly;
    return true;
  }
  return false;
}

}  // namespace

class DriveModeManagerNode : public rclcpp::Node {
public:
  DriveModeManagerNode() : Node("drive_mode_manager_node") {
    current_section_topic_ =
        declare_parameter<std::string>("current_section_topic", "/localization/current_section");
    avoidance_requested_topic_ = declare_parameter<std::string>(
        "avoidance_requested_topic", "/planning/path_obstacle_filter/avoidance_requested");
    following_requested_topic_ = declare_parameter<std::string>(
        "following_requested_topic", "/planning/path_obstacle_filter/following_requested");
    drive_mode_topic_ =
        declare_parameter<std::string>("drive_mode_topic", "/control/drive_mode");

    SectionPolicy parsed_default_policy = SectionPolicy::kAllowAvoid;
    const std::string default_policy_param =
        declare_parameter<std::string>("default_policy", "allow_avoid");
    if (!ParsePolicy(default_policy_param, parsed_default_policy)) {
      RCLCPP_WARN(get_logger(),
                  "Unknown default_policy '%s'. Falling back to allow_avoid.",
                  default_policy_param.c_str());
    }
    default_policy_ = parsed_default_policy;

    const auto section_names =
        declare_parameter<std::vector<std::string>>("section_names", std::vector<std::string>{});
    const auto section_policies = declare_parameter<std::vector<std::string>>(
        "section_policies", std::vector<std::string>{});

    if (section_names.size() != section_policies.size()) {
      RCLCPP_WARN(
          get_logger(),
          "Length mismatch: section_names=%zu, section_policies=%zu. Extra entries are ignored.",
          section_names.size(), section_policies.size());
    }

    const size_t count = std::min(section_names.size(), section_policies.size());
    for (size_t i = 0; i < count; ++i) {
      SectionPolicy policy = SectionPolicy::kAllowAvoid;
      if (!ParsePolicy(section_policies[i], policy)) {
        RCLCPP_WARN(get_logger(),
                    "Unknown policy '%s' for section '%s'. Entry is ignored.",
                    section_policies[i].c_str(), section_names[i].c_str());
        continue;
      }
      section_policy_map_[section_names[i]] = policy;
    }

    drive_mode_pub_ = create_publisher<std_msgs::msg::String>(
        drive_mode_topic_, rclcpp::QoS(1).reliable().transient_local());

    current_section_sub_ = create_subscription<std_msgs::msg::String>(
        current_section_topic_, rclcpp::QoS(10),
        std::bind(&DriveModeManagerNode::CurrentSectionCallback, this, std::placeholders::_1));
    avoidance_requested_sub_ = create_subscription<std_msgs::msg::Bool>(
        avoidance_requested_topic_, rclcpp::QoS(10),
        std::bind(&DriveModeManagerNode::AvoidanceRequestedCallback, this,
                  std::placeholders::_1));
    following_requested_sub_ = create_subscription<std_msgs::msg::Bool>(
        following_requested_topic_, rclcpp::QoS(10),
        std::bind(&DriveModeManagerNode::FollowingRequestedCallback, this,
                  std::placeholders::_1));

    PublishIfNeeded("initialization", true);
  }

private:
  void CurrentSectionCallback(const std_msgs::msg::String::SharedPtr msg) {
    current_section_ = msg->data;
    PublishIfNeeded("section update", false);
  }

  void AvoidanceRequestedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    avoidance_requested_ = msg->data;
    PublishIfNeeded("avoidance update", false);
  }

  void FollowingRequestedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    following_requested_ = msg->data;
    PublishIfNeeded("following update", false);
  }

  SectionPolicy ResolvePolicyForCurrentSection() const {
    const auto it = section_policy_map_.find(current_section_);
    if (it != section_policy_map_.end()) {
      return it->second;
    }
    return default_policy_;
  }

  std::string ResolveMode() const {
    const SectionPolicy policy = ResolvePolicyForCurrentSection();
    if (avoidance_requested_) {
      if (policy == SectionPolicy::kAllowAvoid) {
        return "avoid";
      }
      if (policy == SectionPolicy::kFollowOnly) {
        return "follow";
      }
    }

    if (following_requested_) {
      if (policy == SectionPolicy::kAllowAvoid || policy == SectionPolicy::kFollowOnly) {
        return "follow";
      }
    }

    return "normal";
  }

  void PublishIfNeeded(const char *reason, const bool force_publish) {
    const SectionPolicy policy = ResolvePolicyForCurrentSection();
    const std::string next_mode = ResolveMode();
    if (!force_publish && has_published_mode_ && next_mode == current_mode_) {
      return;
    }

    current_mode_ = next_mode;
    has_published_mode_ = true;

    std_msgs::msg::String msg;
    msg.data = current_mode_;
    drive_mode_pub_->publish(msg);

    RCLCPP_INFO(
        get_logger(),
        "Drive mode -> %s (reason=%s, section='%s', policy=%s, avoidance=%s, following=%s)",
        current_mode_.c_str(), reason, current_section_.c_str(), PolicyToString(policy),
        avoidance_requested_ ? "true" : "false", following_requested_ ? "true" : "false");
  }

  std::string current_section_topic_;
  std::string avoidance_requested_topic_;
  std::string following_requested_topic_;
  std::string drive_mode_topic_;

  SectionPolicy default_policy_{SectionPolicy::kAllowAvoid};
  std::unordered_map<std::string, SectionPolicy> section_policy_map_;

  std::string current_section_{"unknown"};
  bool avoidance_requested_{false};
  bool following_requested_{false};
  std::string current_mode_{"normal"};
  bool has_published_mode_{false};

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr current_section_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr avoidance_requested_sub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr following_requested_sub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr drive_mode_pub_;
};

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DriveModeManagerNode>());
  rclcpp::shutdown();
  return 0;
}

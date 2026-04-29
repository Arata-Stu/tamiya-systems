#include "section_localizer/section_localizer_node.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <fstream>
#include <sstream>
#include <utility>

#include "geometry_msgs/msg/point.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2/exceptions.h"
#include "tf2/time.h"

namespace {

std::string Trim(const std::string &s) {
  const auto begin = s.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = s.find_last_not_of(" \t\r\n");
  return s.substr(begin, end - begin + 1U);
}

std::vector<std::string> Split(const std::string &s, const char delim) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  std::string token;
  while (std::getline(ss, token, delim)) {
    out.push_back(Trim(token));
  }
  return out;
}

bool ParseDouble(const std::string &s, double *value) {
  if (!value) {
    return false;
  }
  try {
    *value = std::stod(s);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseInt(const std::string &s, int *value) {
  if (!value) {
    return false;
  }
  try {
    *value = std::stoi(s);
    return true;
  } catch (...) {
    return false;
  }
}

bool ParseOriginTriplet(const std::string &raw, double *x, double *y,
                        double *yaw) {
  if (!x || !y || !yaw) {
    return false;
  }
  const auto open = raw.find('[');
  const auto close = raw.find(']');
  if (open == std::string::npos || close == std::string::npos || close <= open) {
    return false;
  }
  const auto body = raw.substr(open + 1U, close - open - 1U);
  const auto values = Split(body, ',');
  if (values.size() != 3U) {
    return false;
  }

  double px = 0.0;
  double py = 0.0;
  double pyaw = 0.0;
  if (!ParseDouble(values[0], &px) || !ParseDouble(values[1], &py) ||
      !ParseDouble(values[2], &pyaw)) {
    return false;
  }
  *x = px;
  *y = py;
  *yaw = pyaw;
  return true;
}

double Cross2D(const double ax, const double ay, const double bx,
               const double by) {
  return ax * by - ay * bx;
}

double Orientation(const double ax, const double ay, const double bx,
                   const double by, const double cx, const double cy) {
  return Cross2D(bx - ax, by - ay, cx - ax, cy - ay);
}

bool InRangeWithEps(const double value, const double low, const double high,
                    const double eps) {
  return value >= low - eps && value <= high + eps;
}

bool OnSegment(const double ax, const double ay, const double bx, const double by,
               const double px, const double py, const double eps) {
  return std::abs(Orientation(ax, ay, bx, by, px, py)) <= eps &&
         InRangeWithEps(px, std::min(ax, bx), std::max(ax, bx), eps) &&
         InRangeWithEps(py, std::min(ay, by), std::max(ay, by), eps);
}

} // namespace

SectionLocalizerNode::SectionLocalizerNode() : Node("section_localizer_node") {
  map_yaml_path_ = this->declare_parameter<std::string>("map_yaml_path", "");
  section_definition_path_ =
      this->declare_parameter<std::string>("section_definition_path", "");
  gate_definition_path_ =
      this->declare_parameter<std::string>("gate_definition_path", "");
  map_frame_ = this->declare_parameter<std::string>("map_frame", map_frame_);
  base_frame_ = this->declare_parameter<std::string>("base_frame", base_frame_);
  current_section_topic_ = this->declare_parameter<std::string>(
      "current_section_topic", current_section_topic_);
  marker_topic_ =
      this->declare_parameter<std::string>("marker_topic", marker_topic_);
  current_marker_topic_ = this->declare_parameter<std::string>(
      "current_marker_topic", current_marker_topic_);
  update_rate_hz_ = std::max(
      0.0, this->declare_parameter<double>("update_rate_hz", update_rate_hz_));
  debug_mode_ = this->declare_parameter<bool>("debug_mode", debug_mode_);
  use_gate_hybrid_ =
      this->declare_parameter<bool>("use_gate_hybrid", use_gate_hybrid_);
  enable_reverse_gate_transition_ = this->declare_parameter<bool>(
      "enable_reverse_gate_transition", enable_reverse_gate_transition_);
  fallback_confirm_count_ = std::max(
      1, this->declare_parameter<int>("fallback_confirm_count",
                                      fallback_confirm_count_));
  gate_crossing_eps_ = std::max(
      1e-9, this->declare_parameter<double>("gate_crossing_eps",
                                            gate_crossing_eps_));
  max_motion_for_gate_detection_m_ = std::max(
      0.0, this->declare_parameter<double>("max_motion_for_gate_detection_m",
                                           max_motion_for_gate_detection_m_));

  tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  auto section_qos = rclcpp::QoS(1).reliable().transient_local();
  current_section_pub_ =
      this->create_publisher<std_msgs::msg::String>(current_section_topic_,
                                                    section_qos);

  if (debug_mode_) {
    const auto marker_qos = rclcpp::QoS(1).reliable().transient_local();
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
        marker_topic_, marker_qos);
    current_marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
        current_marker_topic_, marker_qos);
  }

  if (map_yaml_path_.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Parameter 'map_yaml_path' is empty. Node is disabled.");
  } else if (section_definition_path_.empty()) {
    RCLCPP_ERROR(this->get_logger(),
                 "Parameter 'section_definition_path' is empty. Node is "
                 "disabled.");
  } else if (!LoadMapYaml(map_yaml_path_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load map yaml: %s",
                 map_yaml_path_.c_str());
  } else if (!LoadSectionDefinition(section_definition_path_)) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to load section definition CSV: %s",
                 section_definition_path_.c_str());
  } else if (!LoadGateDefinition(gate_definition_path_)) {
    RCLCPP_ERROR(this->get_logger(),
                 "Failed to load gate definition CSV: %s",
                 gate_definition_path_.c_str());
  } else {
    ConvertPolygonsToMapCoordinates();
    ConvertGatesToMapCoordinates();
    loaded_ok_ = true;
    PublishStaticDebugMarkers();
    current_section_name_ = "unknown";
    last_published_section_name_ = "unknown";
    RCLCPP_INFO(this->get_logger(),
                "Section localizer started: sections=%zu gates=%zu map_frame=%s "
                "base_frame=%s debug=%s hybrid=%s",
                sections_.size(), gates_.size(), map_frame_.c_str(),
                base_frame_.c_str(), debug_mode_ ? "true" : "false",
                use_gate_hybrid_ ? "true" : "false");
  }

  const double period_sec = update_rate_hz_ > 0.0 ? (1.0 / update_rate_hz_) : 0.1;
  timer_ = this->create_wall_timer(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::duration<double>(period_sec)),
      std::bind(&SectionLocalizerNode::TimerCallback, this));
}

bool SectionLocalizerNode::LoadMapYaml(const std::string &map_yaml_path) {
  std::ifstream ifs(map_yaml_path);
  if (!ifs.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open map yaml: %s",
                 map_yaml_path.c_str());
    return false;
  }

  bool has_resolution = false;
  bool has_origin = false;
  bool has_image = false;

  std::string line;
  while (std::getline(ifs, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const auto pos = trimmed.find(':');
    if (pos == std::string::npos) {
      continue;
    }

    const std::string key = Trim(trimmed.substr(0, pos));
    const std::string value = Trim(trimmed.substr(pos + 1U));

    if (key == "image") {
      map_metadata_.image_path = value;
      has_image = !value.empty();
    } else if (key == "resolution") {
      has_resolution = ParseDouble(value, &map_metadata_.resolution);
    } else if (key == "origin") {
      has_origin = ParseOriginTriplet(value, &map_metadata_.origin_x,
                                      &map_metadata_.origin_y,
                                      &map_metadata_.origin_yaw);
    }
  }

  if (!has_image || !has_resolution || !has_origin) {
    RCLCPP_ERROR(this->get_logger(),
                 "Map yaml missing required keys (image/resolution/origin).");
    return false;
  }

  return true;
}

bool SectionLocalizerNode::LoadSectionDefinition(
    const std::string &section_definition_path) {
  std::ifstream ifs(section_definition_path);
  if (!ifs.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open section definition: %s",
                 section_definition_path.c_str());
    return false;
  }

  sections_.clear();
  map_metadata_.image_width = 0;
  map_metadata_.image_height = 0;

  std::string line;
  std::size_t line_no = 0;
  while (std::getline(ifs, line)) {
    ++line_no;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }

    const auto tokens = Split(trimmed, ',');
    if (tokens.empty()) {
      continue;
    }

    const std::string &key = tokens[0];
    if (key == "image_width") {
      if (tokens.size() < 2U ||
          !ParseInt(tokens[1], &map_metadata_.image_width)) {
        RCLCPP_ERROR(this->get_logger(),
                     "Invalid image_width at line %zu in %s", line_no,
                     section_definition_path.c_str());
        return false;
      }
      continue;
    }

    if (key == "image_height") {
      if (tokens.size() < 2U ||
          !ParseInt(tokens[1], &map_metadata_.image_height)) {
        RCLCPP_ERROR(this->get_logger(),
                     "Invalid image_height at line %zu in %s", line_no,
                     section_definition_path.c_str());
        return false;
      }
      continue;
    }

    if (key != "section") {
      continue;
    }

    if (tokens.size() < 8U) {
      RCLCPP_WARN(this->get_logger(),
                  "Skip line %zu: section requires at least 3 points",
                  line_no);
      continue;
    }

    SectionPolygon section;
    section.name = tokens[1];
    if (section.name.empty()) {
      section.name = "unnamed_section";
    }

    const std::size_t value_count = tokens.size() - 2U;
    if (value_count % 2U != 0U) {
      RCLCPP_WARN(this->get_logger(),
                  "Skip line %zu: odd number of pixel values", line_no);
      continue;
    }

    bool ok = true;
    for (std::size_t i = 2U; i + 1U < tokens.size(); i += 2U) {
      PixelPoint point;
      ok = ParseInt(tokens[i], &point.u) && ParseInt(tokens[i + 1U], &point.v);
      if (!ok) {
        break;
      }
      section.pixel_points.push_back(point);
    }
    if (!ok || section.pixel_points.size() < 3U) {
      RCLCPP_WARN(this->get_logger(),
                  "Skip line %zu: invalid or too few polygon points", line_no);
      continue;
    }

    sections_.push_back(std::move(section));
  }

  if (map_metadata_.image_height <= 0 || map_metadata_.image_width <= 0) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Section CSV must include positive image_width and image_height.");
    return false;
  }
  if (sections_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "No valid sections found in %s",
                 section_definition_path.c_str());
    return false;
  }

  return true;
}

bool SectionLocalizerNode::LoadGateDefinition(
    const std::string &gate_definition_path) {
  gates_.clear();
  if (gate_definition_path.empty()) {
    return true;
  }

  std::ifstream ifs(gate_definition_path);
  if (!ifs.is_open()) {
    RCLCPP_ERROR(this->get_logger(), "Cannot open gate definition: %s",
                 gate_definition_path.c_str());
    return false;
  }

  std::string line;
  std::size_t line_no = 0;
  std::size_t gate_index = 1;
  while (std::getline(ifs, line)) {
    ++line_no;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    const auto tokens = Split(trimmed, ',');
    if (tokens.empty()) {
      continue;
    }
    if (tokens[0] != "gate") {
      continue;
    }
    if (tokens.size() < 8U) {
      RCLCPP_WARN(this->get_logger(),
                  "Skip gate line %zu: format must be "
                  "gate,name,from_section,to_section,u0,v0,u1,v1",
                  line_no);
      continue;
    }

    GateDefinition gate;
    gate.name = tokens[1].empty() ? ("gate_" + std::to_string(gate_index))
                                  : tokens[1];
    gate.from_section = tokens[2];
    gate.to_section = tokens[3];
    bool ok = ParseInt(tokens[4], &gate.p0.u) && ParseInt(tokens[5], &gate.p0.v) &&
              ParseInt(tokens[6], &gate.p1.u) && ParseInt(tokens[7], &gate.p1.v);
    if (!ok || gate.from_section.empty() || gate.to_section.empty()) {
      RCLCPP_WARN(this->get_logger(), "Skip invalid gate line %zu", line_no);
      continue;
    }

    gates_.push_back(std::move(gate));
    ++gate_index;
  }

  if (!gates_.empty()) {
    RCLCPP_INFO(this->get_logger(), "Loaded %zu gates from %s", gates_.size(),
                gate_definition_path.c_str());
  } else {
    RCLCPP_WARN(this->get_logger(),
                "No valid gates found in %s. Hybrid mode will use fallback only.",
                gate_definition_path.c_str());
  }

  return true;
}

void SectionLocalizerNode::ConvertPolygonsToMapCoordinates() {
  for (auto &section : sections_) {
    section.world_points.clear();
    section.world_points.reserve(section.pixel_points.size());
    for (const auto &pixel : section.pixel_points) {
      section.world_points.push_back(PixelToMap(pixel));
    }
  }
}

void SectionLocalizerNode::ConvertGatesToMapCoordinates() {
  for (auto &gate : gates_) {
    gate.w0 = PixelToMap(gate.p0);
    gate.w1 = PixelToMap(gate.p1);
  }
}

WorldPoint SectionLocalizerNode::PixelToMap(const PixelPoint &pixel) const {
  // Pixel (u, v) -> grid center (gx, gy) -> map (x, y)
  // gx = u + 0.5
  // gy = H - v - 0.5
  // x = ox + res * (gx*cos(theta) - gy*sin(theta))
  // y = oy + res * (gx*sin(theta) + gy*cos(theta))
  const double gx = static_cast<double>(pixel.u) + 0.5;
  const double gy = static_cast<double>(map_metadata_.image_height) -
                    static_cast<double>(pixel.v) - 0.5;
  const double cos_t = std::cos(map_metadata_.origin_yaw);
  const double sin_t = std::sin(map_metadata_.origin_yaw);

  WorldPoint world;
  world.x = map_metadata_.origin_x +
            map_metadata_.resolution * (gx * cos_t - gy * sin_t);
  world.y = map_metadata_.origin_y +
            map_metadata_.resolution * (gx * sin_t + gy * cos_t);
  return world;
}

bool SectionLocalizerNode::IsPointInsidePolygon(
    const double x, const double y,
    const std::vector<WorldPoint> &polygon) const {
  if (polygon.size() < 3U) {
    return false;
  }

  bool inside = false;
  std::size_t j = polygon.size() - 1U;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const auto &pi = polygon[i];
    const auto &pj = polygon[j];
    const bool intersects =
        ((pi.y > y) != (pj.y > y)) &&
        (x < (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y + 1e-12) + pi.x);
    if (intersects) {
      inside = !inside;
    }
    j = i;
  }
  return inside;
}

double SectionLocalizerNode::SignedSide(const GateDefinition &gate, const double x,
                                        const double y) const {
  return Orientation(gate.w0.x, gate.w0.y, gate.w1.x, gate.w1.y, x, y);
}

bool SectionLocalizerNode::SegmentIntersects(const double a_x, const double a_y,
                                             const double b_x, const double b_y,
                                             const GateDefinition &gate) const {
  const double c_x = gate.w0.x;
  const double c_y = gate.w0.y;
  const double d_x = gate.w1.x;
  const double d_y = gate.w1.y;
  const double eps = gate_crossing_eps_;

  const double o1 = Orientation(a_x, a_y, b_x, b_y, c_x, c_y);
  const double o2 = Orientation(a_x, a_y, b_x, b_y, d_x, d_y);
  const double o3 = Orientation(c_x, c_y, d_x, d_y, a_x, a_y);
  const double o4 = Orientation(c_x, c_y, d_x, d_y, b_x, b_y);

  const bool proper_intersect =
      ((o1 > eps && o2 < -eps) || (o1 < -eps && o2 > eps)) &&
      ((o3 > eps && o4 < -eps) || (o3 < -eps && o4 > eps));
  if (proper_intersect) {
    return true;
  }

  if (OnSegment(a_x, a_y, b_x, b_y, c_x, c_y, eps) ||
      OnSegment(a_x, a_y, b_x, b_y, d_x, d_y, eps) ||
      OnSegment(c_x, c_y, d_x, d_y, a_x, a_y, eps) ||
      OnSegment(c_x, c_y, d_x, d_y, b_x, b_y, eps)) {
    return true;
  }

  return false;
}

bool SectionLocalizerNode::TryGateTransition(const double prev_x,
                                             const double prev_y,
                                             const double curr_x,
                                             const double curr_y) {
  if (!use_gate_hybrid_ || gates_.empty()) {
    return false;
  }

  for (const auto &gate : gates_) {
    const bool forward_match = (current_section_name_ == gate.from_section);
    const bool reverse_match =
        enable_reverse_gate_transition_ && (current_section_name_ == gate.to_section);
    if (!forward_match && !reverse_match) {
      continue;
    }
    if (!SegmentIntersects(prev_x, prev_y, curr_x, curr_y, gate)) {
      continue;
    }

    const double prev_side = SignedSide(gate, prev_x, prev_y);
    const double curr_side = SignedSide(gate, curr_x, curr_y);
    const double eps = gate_crossing_eps_;

    if (forward_match && prev_side < -eps && curr_side > eps) {
      const std::string prev_section = current_section_name_;
      current_section_name_ = gate.to_section;
      RCLCPP_INFO(this->get_logger(),
                  "Gate transition: %s -> %s via %s (forward)",
                  prev_section.c_str(), current_section_name_.c_str(),
                  gate.name.c_str());
      return true;
    }
    if (reverse_match && prev_side > eps && curr_side < -eps) {
      const std::string prev_section = current_section_name_;
      current_section_name_ = gate.from_section;
      RCLCPP_INFO(this->get_logger(),
                  "Gate transition: %s -> %s via %s (reverse)",
                  prev_section.c_str(), current_section_name_.c_str(),
                  gate.name.c_str());
      return true;
    }
  }
  return false;
}

void SectionLocalizerNode::UpdateSectionWithFallback(
    const std::string &fallback_section) {
  if (fallback_section.empty() || fallback_section == "unknown") {
    fallback_candidate_ = "unknown";
    fallback_candidate_count_ = 0;
    return;
  }

  if (current_section_name_ == "unknown") {
    current_section_name_ = fallback_section;
    fallback_candidate_ = "unknown";
    fallback_candidate_count_ = 0;
    return;
  }

  if (fallback_section == current_section_name_) {
    fallback_candidate_ = "unknown";
    fallback_candidate_count_ = 0;
    return;
  }

  if (fallback_candidate_ == fallback_section) {
    ++fallback_candidate_count_;
  } else {
    fallback_candidate_ = fallback_section;
    fallback_candidate_count_ = 1;
  }

  if (fallback_candidate_count_ >= fallback_confirm_count_) {
    RCLCPP_WARN(this->get_logger(),
                "Fallback recovery: %s -> %s (confirm_count=%d)",
                current_section_name_.c_str(), fallback_section.c_str(),
                fallback_candidate_count_);
    current_section_name_ = fallback_section;
    fallback_candidate_ = "unknown";
    fallback_candidate_count_ = 0;
  }
}

std::string SectionLocalizerNode::FindSectionByPosition(const double x,
                                                        const double y) const {
  for (const auto &section : sections_) {
    if (IsPointInsidePolygon(x, y, section.world_points)) {
      return section.name;
    }
  }
  return "unknown";
}

void SectionLocalizerNode::PublishStaticDebugMarkers() {
  if (!debug_mode_ || !marker_pub_) {
    return;
  }

  visualization_msgs::msg::MarkerArray marker_array;
  const auto stamp = this->now();

  for (std::size_t i = 0; i < sections_.size(); ++i) {
    const auto &section = sections_[i];

    visualization_msgs::msg::Marker line_marker;
    line_marker.header.frame_id = map_frame_;
    line_marker.header.stamp = stamp;
    line_marker.ns = "section_polygon";
    line_marker.id = static_cast<int>(i);
    line_marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    line_marker.action = visualization_msgs::msg::Marker::ADD;
    line_marker.pose.orientation.w = 1.0;
    line_marker.scale.x = 0.05;
    line_marker.color.r = 0.2F;
    line_marker.color.g = 0.7F;
    line_marker.color.b = 1.0F;
    line_marker.color.a = 0.9F;

    for (const auto &pt : section.world_points) {
      geometry_msgs::msg::Point p;
      p.x = pt.x;
      p.y = pt.y;
      p.z = 0.05;
      line_marker.points.push_back(p);
    }
    if (!line_marker.points.empty()) {
      line_marker.points.push_back(line_marker.points.front());
    }
    marker_array.markers.push_back(std::move(line_marker));

    visualization_msgs::msg::Marker text_marker;
    text_marker.header.frame_id = map_frame_;
    text_marker.header.stamp = stamp;
    text_marker.ns = "section_name";
    text_marker.id = static_cast<int>(1000U + i);
    text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker.action = visualization_msgs::msg::Marker::ADD;
    text_marker.pose.orientation.w = 1.0;
    text_marker.scale.z = 0.5;
    text_marker.color.r = 1.0F;
    text_marker.color.g = 1.0F;
    text_marker.color.b = 1.0F;
    text_marker.color.a = 1.0F;
    text_marker.text = section.name;

    double sum_x = 0.0;
    double sum_y = 0.0;
    for (const auto &pt : section.world_points) {
      sum_x += pt.x;
      sum_y += pt.y;
    }
    const double inv = 1.0 / static_cast<double>(section.world_points.size());
    text_marker.pose.position.x = sum_x * inv;
    text_marker.pose.position.y = sum_y * inv;
    text_marker.pose.position.z = 0.35;
    marker_array.markers.push_back(std::move(text_marker));
  }

  for (std::size_t i = 0; i < gates_.size(); ++i) {
    const auto &gate = gates_[i];

    visualization_msgs::msg::Marker gate_line;
    gate_line.header.frame_id = map_frame_;
    gate_line.header.stamp = stamp;
    gate_line.ns = "section_gate";
    gate_line.id = static_cast<int>(2000U + i);
    gate_line.type = visualization_msgs::msg::Marker::LINE_LIST;
    gate_line.action = visualization_msgs::msg::Marker::ADD;
    gate_line.pose.orientation.w = 1.0;
    gate_line.scale.x = 0.08;
    gate_line.color.r = 1.0F;
    gate_line.color.g = 0.7F;
    gate_line.color.b = 0.2F;
    gate_line.color.a = 1.0F;

    geometry_msgs::msg::Point p0;
    p0.x = gate.w0.x;
    p0.y = gate.w0.y;
    p0.z = 0.2;
    geometry_msgs::msg::Point p1;
    p1.x = gate.w1.x;
    p1.y = gate.w1.y;
    p1.z = 0.2;
    gate_line.points.push_back(p0);
    gate_line.points.push_back(p1);
    marker_array.markers.push_back(std::move(gate_line));

    visualization_msgs::msg::Marker gate_text;
    gate_text.header.frame_id = map_frame_;
    gate_text.header.stamp = stamp;
    gate_text.ns = "section_gate_name";
    gate_text.id = static_cast<int>(3000U + i);
    gate_text.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    gate_text.action = visualization_msgs::msg::Marker::ADD;
    gate_text.pose.orientation.w = 1.0;
    gate_text.scale.z = 0.35;
    gate_text.color.r = 1.0F;
    gate_text.color.g = 0.95F;
    gate_text.color.b = 0.4F;
    gate_text.color.a = 1.0F;
    gate_text.text =
        gate.name + " : " + gate.from_section + " -> " + gate.to_section;
    gate_text.pose.position.x = 0.5 * (gate.w0.x + gate.w1.x);
    gate_text.pose.position.y = 0.5 * (gate.w0.y + gate.w1.y);
    gate_text.pose.position.z = 0.5;
    marker_array.markers.push_back(std::move(gate_text));
  }

  marker_pub_->publish(marker_array);
}

void SectionLocalizerNode::PublishCurrentSectionMarker(
    const std::string &section_name) {
  if (!debug_mode_ || !current_marker_pub_) {
    return;
  }

  visualization_msgs::msg::Marker marker;
  marker.header.frame_id = map_frame_;
  marker.header.stamp = this->now();
  marker.ns = "current_section";
  marker.id = 0;
  marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.12;
  marker.color.r = 1.0F;
  marker.color.g = 0.2F;
  marker.color.b = 0.2F;
  marker.color.a = 0.95F;

  const auto it =
      std::find_if(sections_.begin(), sections_.end(),
                   [&section_name](const SectionPolygon &s) {
                     return s.name == section_name;
                   });

  if (it == sections_.end()) {
    marker.action = visualization_msgs::msg::Marker::DELETE;
    current_marker_pub_->publish(marker);
    return;
  }

  marker.action = visualization_msgs::msg::Marker::ADD;
  for (const auto &pt : it->world_points) {
    geometry_msgs::msg::Point p;
    p.x = pt.x;
    p.y = pt.y;
    p.z = 0.1;
    marker.points.push_back(p);
  }
  if (!marker.points.empty()) {
    marker.points.push_back(marker.points.front());
  }
  current_marker_pub_->publish(marker);
}

void SectionLocalizerNode::TimerCallback() {
  if (!loaded_ok_ || !tf_buffer_) {
    return;
  }

  geometry_msgs::msg::TransformStamped tf_msg;
  try {
    // lookupTransform(target, source): transform from source frame to target frame.
    tf_msg = tf_buffer_->lookupTransform(map_frame_, base_frame_,
                                         tf2::TimePointZero);
  } catch (const tf2::TransformException &ex) {
    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                         "Section localizer TF lookup failed (%s <- %s): %s",
                         map_frame_.c_str(), base_frame_.c_str(), ex.what());
    return;
  }

  const double x = tf_msg.transform.translation.x;
  const double y = tf_msg.transform.translation.y;
  const std::string fallback_section = FindSectionByPosition(x, y);

  if (current_section_name_ == "unknown" && fallback_section != "unknown") {
    current_section_name_ = fallback_section;
  }

  if (has_prev_pose_) {
    const double dx = x - prev_x_;
    const double dy = y - prev_y_;
    const double motion = std::sqrt(dx * dx + dy * dy);
    if (motion <= max_motion_for_gate_detection_m_) {
      (void)TryGateTransition(prev_x_, prev_y_, x, y);
    } else {
      RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Skip gate transition due to large motion %.3f m (> %.3f m).",
          motion, max_motion_for_gate_detection_m_);
    }
  }

  UpdateSectionWithFallback(fallback_section);

  std_msgs::msg::String section_msg;
  section_msg.data = current_section_name_;
  current_section_pub_->publish(section_msg);

  if (current_section_name_ != last_published_section_name_) {
    RCLCPP_INFO(this->get_logger(),
                "Section changed: %s -> %s (x=%.3f, y=%.3f, fallback=%s)",
                last_published_section_name_.c_str(),
                current_section_name_.c_str(), x, y, fallback_section.c_str());
    last_published_section_name_ = current_section_name_;
    PublishCurrentSectionMarker(current_section_name_);
  }

  prev_x_ = x;
  prev_y_ = y;
  has_prev_pose_ = true;
}

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<SectionLocalizerNode>());
  rclcpp::shutdown();
  return 0;
}

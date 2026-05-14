#include "raceline_path_publisher/raceline_path_publisher_core.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"

namespace raceline_path_publisher {

namespace {

constexpr double kDuplicatePointEpsilon = 1.0e-9;
constexpr double kPi = 3.14159265358979323846;

std::string Trim(const std::string &text) {
  const auto begin = text.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(" \t\r\n");
  return text.substr(begin, end - begin + 1U);
}

std::string ToLower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return text;
}

bool ParseDouble(const std::string &text, double &value) {
  try {
    std::size_t consumed = 0U;
    value = std::stod(text, &consumed);
    return consumed == text.size();
  } catch (...) {
    return false;
  }
}

bool LooksLikeHeader(const std::vector<std::string> &tokens) {
  if (tokens.empty()) {
    return false;
  }
  for (const auto &token : tokens) {
    double unused = 0.0;
    if (!ParseDouble(token, unused)) {
      return true;
    }
  }
  return false;
}

char DetectDelimiter(const std::string &line) {
  const auto semicolon_count = std::count(line.begin(), line.end(), ';');
  const auto comma_count = std::count(line.begin(), line.end(), ',');
  const auto tab_count = std::count(line.begin(), line.end(), '\t');

  if (semicolon_count >= comma_count && semicolon_count >= tab_count &&
      semicolon_count > 0) {
    return ';';
  }
  if (comma_count >= semicolon_count && comma_count >= tab_count &&
      comma_count > 0) {
    return ',';
  }
  if (tab_count > 0) {
    return '\t';
  }
  return ',';
}

std::vector<std::string> SplitTokens(const std::string &line, char delimiter) {
  std::vector<std::string> tokens;
  std::stringstream ss(line);
  std::string token;
  while (std::getline(ss, token, delimiter)) {
    tokens.push_back(Trim(token));
  }
  return tokens;
}

double NormalizeAngle(double angle) {
  while (angle > kPi) {
    angle -= 2.0 * kPi;
  }
  while (angle <= -kPi) {
    angle += 2.0 * kPi;
  }
  return angle;
}

double Distance2D(const RacelineSample &a, const RacelineSample &b) {
  return std::hypot(a.x - b.x, a.y - b.y);
}

void RemoveDuplicateSamples(std::vector<RacelineSample> &samples) {
  if (samples.size() < 2U) {
    return;
  }

  std::vector<RacelineSample> filtered;
  filtered.reserve(samples.size());
  filtered.push_back(samples.front());

  for (std::size_t i = 1U; i < samples.size(); ++i) {
    if (Distance2D(samples[i], filtered.back()) > kDuplicatePointEpsilon) {
      filtered.push_back(samples[i]);
    }
  }

  if (filtered.size() > 1U &&
      Distance2D(filtered.front(), filtered.back()) <= kDuplicatePointEpsilon) {
    filtered.pop_back();
  }

  samples = std::move(filtered);
}

void ReverseSamples(std::vector<RacelineSample> &samples) {
  std::reverse(samples.begin(), samples.end());
}

void RecomputeGeometry(RacelineData &data) {
  auto &samples = data.samples;
  data.total_length = 0.0;
  data.nominal_spacing = 0.0;

  if (samples.empty()) {
    return;
  }

  if (samples.size() == 1U) {
    samples.front().s = 0.0;
    return;
  }

  samples.front().s = 0.0;
  double cumulative = 0.0;
  for (std::size_t i = 1U; i < samples.size(); ++i) {
    cumulative += Distance2D(samples[i - 1U], samples[i]);
    samples[i].s = cumulative;
  }

  data.total_length =
      cumulative + Distance2D(samples.back(), samples.front());
  data.nominal_spacing = data.total_length / static_cast<double>(samples.size());

  if (samples.size() == 2U) {
    const double yaw =
        std::atan2(samples[1U].y - samples[0U].y, samples[1U].x - samples[0U].x);
    samples[0U].yaw = yaw;
    samples[1U].yaw = yaw;
    samples[0U].curvature = 0.0;
    samples[1U].curvature = 0.0;
    return;
  }

  for (std::size_t i = 0U; i < samples.size(); ++i) {
    const std::size_t prev_index =
        (i == 0U) ? samples.size() - 1U : i - 1U;
    const std::size_t next_index = (i + 1U) % samples.size();

    const auto &prev = samples[prev_index];
    const auto &curr = samples[i];
    const auto &next = samples[next_index];

    const double dx = next.x - prev.x;
    const double dy = next.y - prev.y;
    samples[i].yaw = std::atan2(dy, dx);

    const double ax = curr.x - prev.x;
    const double ay = curr.y - prev.y;
    const double bx = next.x - curr.x;
    const double by = next.y - curr.y;
    const double cx = next.x - prev.x;
    const double cy = next.y - prev.y;

    const double la = std::hypot(ax, ay);
    const double lb = std::hypot(bx, by);
    const double lc = std::hypot(cx, cy);
    const double denom = la * lb * lc;
    if (denom <= 1.0e-9) {
      samples[i].curvature = 0.0;
      continue;
    }

    const double cross = ax * by - ay * bx;
    samples[i].curvature = 2.0 * cross / denom;
  }
}

bool LoadRawSamples(const std::string &csv_path,
                    std::vector<RacelineSample> &samples,
                    std::string &error_message) {
  std::ifstream ifs(csv_path);
  if (!ifs.is_open()) {
    error_message = "Failed to open CSV: " + csv_path;
    return false;
  }

  std::vector<std::string> header_tokens;
  std::vector<std::vector<double>> rows;
  char delimiter = ',';
  bool delimiter_initialized = false;
  std::string line;

  while (std::getline(ifs, line)) {
    const std::string trimmed = Trim(line);
    if (trimmed.empty()) {
      continue;
    }

    const bool is_comment = !trimmed.empty() && trimmed.front() == '#';
    const std::string content =
        is_comment ? Trim(trimmed.substr(1U)) : trimmed;
    if (content.empty()) {
      continue;
    }

    if (!delimiter_initialized) {
      delimiter = DetectDelimiter(content);
      delimiter_initialized = true;
    }

    const auto tokens = SplitTokens(content, delimiter);
    if (tokens.empty()) {
      continue;
    }

    if (LooksLikeHeader(tokens)) {
      if (header_tokens.empty()) {
        header_tokens = tokens;
      }
      continue;
    }

    if (is_comment) {
      continue;
    }

    std::vector<double> row;
    row.reserve(tokens.size());
    for (const auto &token : tokens) {
      double value = 0.0;
      if (!ParseDouble(token, value)) {
        error_message = "Failed to parse numeric value '" + token +
                        "' in CSV: " + csv_path;
        return false;
      }
      row.push_back(value);
    }
    rows.push_back(std::move(row));
  }

  if (rows.empty()) {
    error_message = "CSV does not contain numeric raceline rows: " + csv_path;
    return false;
  }

  const std::size_t column_count = rows.front().size();
  for (const auto &row : rows) {
    if (row.size() != column_count) {
      error_message = "CSV rows have inconsistent column counts: " + csv_path;
      return false;
    }
  }

  std::unordered_map<std::string, std::size_t> header_index;
  for (std::size_t i = 0U; i < header_tokens.size(); ++i) {
    header_index[ToLower(header_tokens[i])] = i;
  }

  auto get_index = [&](const std::vector<std::string> &keys,
                       int fallback) -> int {
    for (const auto &key : keys) {
      const auto it = header_index.find(ToLower(key));
      if (it != header_index.end()) {
        return static_cast<int>(it->second);
      }
    }
    return fallback;
  };

  const int default_x_index =
      column_count >= 7U ? 1 : (column_count >= 2U ? 0 : -1);
  const int default_y_index =
      column_count >= 7U ? 2 : (column_count >= 2U ? 1 : -1);
  int x_index = get_index({"x_m", "x", "x_px"}, default_x_index);
  int y_index = get_index({"y_m", "y", "y_px"}, default_y_index);
  int s_index = get_index({"s_m", "s"}, column_count >= 7U ? 0 : -1);
  int yaw_index =
      get_index({"psi_rad", "yaw", "heading"}, column_count >= 7U ? 3 : -1);
  int curvature_index = get_index({"kappa_radpm", "curvature"},
                                  column_count >= 7U ? 4 : -1);
  int speed_index = get_index({"vx_mps", "speed", "speed_mps", "v"},
                              column_count >= 7U ? 5 : (column_count == 3U ? 2 : -1));
  int accel_index = get_index({"ax_mps2", "accel", "acceleration"},
                              column_count >= 7U ? 6 : -1);

  if (x_index < 0 || y_index < 0 ||
      static_cast<std::size_t>(std::max(x_index, y_index)) >= column_count) {
    error_message = "Could not infer x/y columns from CSV: " + csv_path;
    return false;
  }

  samples.clear();
  samples.reserve(rows.size());
  for (const auto &row : rows) {
    RacelineSample sample;
    sample.x = row[static_cast<std::size_t>(x_index)];
    sample.y = row[static_cast<std::size_t>(y_index)];
    if (s_index >= 0 && static_cast<std::size_t>(s_index) < row.size()) {
      sample.s = row[static_cast<std::size_t>(s_index)];
    }
    if (yaw_index >= 0 && static_cast<std::size_t>(yaw_index) < row.size()) {
      sample.yaw = row[static_cast<std::size_t>(yaw_index)];
    }
    if (curvature_index >= 0 &&
        static_cast<std::size_t>(curvature_index) < row.size()) {
      sample.curvature = row[static_cast<std::size_t>(curvature_index)];
    }
    if (speed_index >= 0 && static_cast<std::size_t>(speed_index) < row.size()) {
      sample.speed = row[static_cast<std::size_t>(speed_index)];
    }
    if (accel_index >= 0 && static_cast<std::size_t>(accel_index) < row.size()) {
      sample.acceleration = row[static_cast<std::size_t>(accel_index)];
    }
    samples.push_back(sample);
  }

  RemoveDuplicateSamples(samples);
  if (samples.size() < 2U) {
    error_message = "Need at least 2 unique path points in CSV: " + csv_path;
    return false;
  }

  return true;
}

geometry_msgs::msg::PoseStamped BuildPoseStamped(
    const RacelineSample &sample, const std::string &frame_id,
    const builtin_interfaces::msg::Time &stamp) {
  geometry_msgs::msg::PoseStamped pose;
  pose.header.frame_id = frame_id;
  pose.header.stamp = stamp;
  pose.pose.position.x = sample.x;
  pose.pose.position.y = sample.y;
  pose.pose.position.z = 0.0;

  tf2::Quaternion q;
  q.setRPY(0.0, 0.0, NormalizeAngle(sample.yaw));
  pose.pose.orientation = tf2::toMsg(q);
  return pose;
}

} // namespace

bool RacelinePathCore::LoadCsv(const std::string &csv_path,
                               const std::string &direction,
                               std::string &error_message) {
  std::vector<RacelineSample> samples;
  if (!LoadRawSamples(csv_path, samples, error_message)) {
    return false;
  }

  const std::string direction_lower = ToLower(direction);
  if (direction_lower == "reverse") {
    ReverseSamples(samples);
  } else if (direction_lower != "forward") {
    error_message = "Unsupported direction: " + direction;
    return false;
  }

  data_.samples = std::move(samples);
  RecomputeGeometry(data_);
  return true;
}

const RacelineData &RacelinePathCore::GetData() const { return data_; }

nav_msgs::msg::Path RacelinePathCore::BuildPath(
    const std::string &frame_id,
    const builtin_interfaces::msg::Time &stamp) const {
  std::vector<std::size_t> indices;
  indices.reserve(data_.samples.size());
  for (std::size_t i = 0U; i < data_.samples.size(); ++i) {
    indices.push_back(i);
  }
  return BuildPathFromIndices(indices, frame_id, stamp);
}

nav_msgs::msg::Path RacelinePathCore::BuildPathFromIndices(
    const std::vector<std::size_t> &indices, const std::string &frame_id,
    const builtin_interfaces::msg::Time &stamp) const {
  nav_msgs::msg::Path path;
  path.header.frame_id = frame_id;
  path.header.stamp = stamp;
  path.poses.reserve(indices.size());

  for (const auto index : indices) {
    if (index >= data_.samples.size()) {
      continue;
    }
    path.poses.push_back(BuildPoseStamped(data_.samples[index], frame_id, stamp));
  }

  return path;
}

std::size_t RacelinePathCore::FindNearestIndex(double x, double y) const {
  if (data_.samples.empty()) {
    return 0U;
  }

  std::size_t nearest_index = 0U;
  double nearest_distance = std::numeric_limits<double>::max();
  for (std::size_t i = 0U; i < data_.samples.size(); ++i) {
    const double dx = data_.samples[i].x - x;
    const double dy = data_.samples[i].y - y;
    const double distance_sq = dx * dx + dy * dy;
    if (distance_sq < nearest_distance) {
      nearest_distance = distance_sq;
      nearest_index = i;
    }
  }
  return nearest_index;
}

std::vector<std::size_t> RacelinePathCore::SelectForwardIndices(
    double vehicle_x, double vehicle_y, double forward_distance_m,
    std::size_t max_points) const {
  std::vector<std::size_t> indices;
  if (data_.samples.empty() || max_points == 0U) {
    return indices;
  }

  const std::size_t nearest_index = FindNearestIndex(vehicle_x, vehicle_y);
  indices.reserve(std::min(max_points, data_.samples.size()));
  indices.push_back(nearest_index);

  if (data_.samples.size() == 1U) {
    return indices;
  }

  double covered_distance = 0.0;
  for (std::size_t step = 1U;
       step < data_.samples.size() && indices.size() < max_points; ++step) {
    const std::size_t previous_index =
        (nearest_index + step - 1U) % data_.samples.size();
    const std::size_t index = (nearest_index + step) % data_.samples.size();

    covered_distance +=
        Distance2D(data_.samples[previous_index], data_.samples[index]);

    indices.push_back(index);
    if (covered_distance >= forward_distance_m) {
      break;
    }
  }

  return indices;
}

} // namespace raceline_path_publisher

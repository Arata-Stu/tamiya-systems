#include "scan_image_projection_cropper/scan_image_projection_cropper_component.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "cv_bridge/cv_bridge.h"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/image_encodings.hpp"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/LinearMath/Transform.h"
#include "tf2/LinearMath/Vector3.h"

namespace scan_image_projection_cropper {

namespace {

double PositiveOr(double value, double fallback) {
  if (value <= 0.0) {
    return fallback;
  }
  return value;
}

int PositiveOr(int value, int fallback) {
  if (value <= 0) {
    return fallback;
  }
  return value;
}

double ClampNonNegative(double value) { return std::max(0.0, value); }

int ClampNonNegativeInt(std::int64_t value) {
  return static_cast<int>(std::max<std::int64_t>(0, value));
}

tf2::Transform ToTfTransform(const geometry_msgs::msg::TransformStamped &transform) {
  const auto &translation = transform.transform.translation;
  const auto &rotation = transform.transform.rotation;

  tf2::Quaternion quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
  tf2::Transform tf_transform(quaternion,
                              tf2::Vector3(translation.x, translation.y,
                                           translation.z));
  return tf_transform;
}

bool IsFinitePositive(double value) {
  return std::isfinite(value) && value > 0.0;
}

} // namespace

ScanImageProjectionCropperComponent::ScanImageProjectionCropperComponent(
    const rclcpp::NodeOptions &options)
    : Node("scan_image_projection_cropper_node", options) {
  min_range_m_ = ClampNonNegative(declare_parameter<double>("min_range_m", 0.05));
  max_range_m_ = PositiveOr(declare_parameter<double>("max_range_m", 8.0), 8.0);
  cluster_separation_threshold_m_ = PositiveOr(
      declare_parameter<double>("cluster_separation_threshold_m", 0.18), 0.18);
  min_cluster_points_ =
      PositiveOr(declare_parameter<int>("min_cluster_points", 2), 2);
  min_cluster_width_m_ = ClampNonNegative(
      declare_parameter<double>("min_cluster_width_m", 0.03));
  max_cluster_width_m_ =
      ClampNonNegative(declare_parameter<double>("max_cluster_width_m", 1.20));
  min_camera_depth_m_ = PositiveOr(
      declare_parameter<double>("min_camera_depth_m", 0.05), 0.05);
  max_scan_age_sec_ = PositiveOr(
      declare_parameter<double>("max_scan_age_sec", 0.20), 0.20);
  max_debug_image_age_sec_ = PositiveOr(
      declare_parameter<double>("max_debug_image_age_sec", 0.20), 0.20);
  tf_timeout_sec_ =
      PositiveOr(declare_parameter<double>("tf_timeout_sec", 0.05), 0.05);
  fixed_crop_height_px_ =
      PositiveOr(declare_parameter<int>("fixed_crop_height_px", 160), 160);
  horizontal_padding_px_ = ClampNonNegativeInt(
      declare_parameter<int>("horizontal_padding_px", 12));
  bottom_padding_px_ =
      declare_parameter<int>("bottom_padding_px", 8);
  min_crop_width_px_ =
      PositiveOr(declare_parameter<int>("min_crop_width_px", 32), 32);
  max_crop_width_px_ = ClampNonNegativeInt(
      declare_parameter<int>("max_crop_width_px", 0));
  output_image_width_px_ = ClampNonNegativeInt(
      declare_parameter<int>("output_image_width_px", 64));
  output_image_height_px_ = ClampNonNegativeInt(
      declare_parameter<int>("output_image_height_px", 64));
  output_padding_value_ = declare_parameter<double>("output_padding_value", 0.0);
  debug_enabled_ = declare_parameter<bool>("debug_enabled", false);
  candidate_selection_mode_ =
      declare_parameter<std::string>("candidate_selection_mode", "widest");
  base_frame_ = declare_parameter<std::string>("base_frame", "base_link");
  tracking_lock_radius_m_ = declare_parameter<double>("tracking_lock_radius_m", 1.0);
  tracker_timeout_sec_ = declare_parameter<double>("tracker_timeout_sec", 0.5);

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

  crop_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
      "crop/image", rclcpp::SensorDataQoS());
  crop_camera_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(
      "crop/camera_info", rclcpp::SensorDataQoS());
  roi_pub_ = create_publisher<sensor_msgs::msg::RegionOfInterest>(
      "crop/roi", rclcpp::QoS(10));
  obstacle_position_pub_ = create_publisher<geometry_msgs::msg::PointStamped>(
      "crop/obstacle_position", rclcpp::QoS(10));

  if (debug_enabled_) {
    debug_image_pub_ = create_publisher<sensor_msgs::msg::Image>(
        "debug/image", rclcpp::SensorDataQoS());
    debug_camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "debug/camera_info", rclcpp::SensorDataQoS(),
        std::bind(&ScanImageProjectionCropperComponent::DebugCameraInfoCallback,
                  this, std::placeholders::_1));
    debug_image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "debug/image_input", rclcpp::SensorDataQoS(),
        std::bind(&ScanImageProjectionCropperComponent::DebugImageCallback, this,
                  std::placeholders::_1));
  }

  tracked_object_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/perception/tracked_object", rclcpp::QoS(10),
      std::bind(&ScanImageProjectionCropperComponent::TrackedObjectCallback, this,
                std::placeholders::_1));

  scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      std::bind(&ScanImageProjectionCropperComponent::ScanCallback, this,
                std::placeholders::_1));
  camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      "camera_info", rclcpp::SensorDataQoS(),
      std::bind(&ScanImageProjectionCropperComponent::CameraInfoCallback, this,
                std::placeholders::_1));
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image", rclcpp::SensorDataQoS(),
      std::bind(&ScanImageProjectionCropperComponent::ImageCallback, this,
                std::placeholders::_1));

  RCLCPP_INFO(
      get_logger(),
      "ScanImageProjectionCropper initialized (height=%d px, output=%dx%d, selection=%s)",
      fixed_crop_height_px_, output_image_width_px_, output_image_height_px_,
      candidate_selection_mode_.c_str());
}

void ScanImageProjectionCropperComponent::ScanCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_scan_ = msg;
}

void ScanImageProjectionCropperComponent::CameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_camera_info_ = msg;
}

void ScanImageProjectionCropperComponent::DebugImageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_debug_image_ = msg;
}

void ScanImageProjectionCropperComponent::DebugCameraInfoCallback(
    const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_debug_camera_info_ = msg;
}

void ScanImageProjectionCropperComponent::TrackedObjectCallback(
    const nav_msgs::msg::Odometry::SharedPtr msg) {
  std::scoped_lock lock(data_mutex_);
  latest_tracked_object_ = msg;
}

std::vector<ScanImageProjectionCropperComponent::ScanCluster>
ScanImageProjectionCropperComponent::ExtractClusters(
    const sensor_msgs::msg::LaserScan &scan) const {
  std::vector<ScanCluster> clusters;
  ScanCluster current_cluster;
  current_cluster.min_range = std::numeric_limits<double>::infinity();
  const auto point_distance = [](const ScanPoint &lhs, const ScanPoint &rhs) {
    return std::hypot(lhs.x - rhs.x, lhs.y - rhs.y);
  };

  auto flush_cluster = [&]() {
    if (static_cast<int>(current_cluster.points.size()) < min_cluster_points_) {
      current_cluster.points.clear();
      current_cluster.min_range = std::numeric_limits<double>::infinity();
      current_cluster.width_m = 0.0;
      return;
    }

    const auto &first = current_cluster.points.front();
    const auto &last = current_cluster.points.back();
    current_cluster.width_m = point_distance(first, last);

    if (current_cluster.width_m < min_cluster_width_m_) {
      current_cluster.points.clear();
      current_cluster.min_range = std::numeric_limits<double>::infinity();
      current_cluster.width_m = 0.0;
      return;
    }

    if (max_cluster_width_m_ > 0.0 &&
        current_cluster.width_m > max_cluster_width_m_) {
      current_cluster.points.clear();
      current_cluster.min_range = std::numeric_limits<double>::infinity();
      current_cluster.width_m = 0.0;
      return;
    }

    clusters.push_back(current_cluster);
    current_cluster.points.clear();
    current_cluster.min_range = std::numeric_limits<double>::infinity();
    current_cluster.width_m = 0.0;
  };

  bool has_previous_point = false;
  ScanPoint previous_point;

  for (std::size_t index = 0; index < scan.ranges.size(); ++index) {
    const double range = static_cast<double>(scan.ranges[index]);
    const double scan_min_range =
        std::max(static_cast<double>(scan.range_min), min_range_m_);
    const double scan_max_range =
        std::min(static_cast<double>(scan.range_max), max_range_m_);

    if (!std::isfinite(range) || range < scan_min_range || range > scan_max_range) {
      flush_cluster();
      has_previous_point = false;
      continue;
    }

    const double angle =
        static_cast<double>(scan.angle_min) +
        static_cast<double>(index) * static_cast<double>(scan.angle_increment);

    ScanPoint point;
    point.x = range * std::cos(angle);
    point.y = range * std::sin(angle);
    point.range = range;

    if (has_previous_point &&
        point_distance(previous_point, point) > cluster_separation_threshold_m_) {
      flush_cluster();
    }

    current_cluster.points.push_back(point);
    current_cluster.min_range =
        std::min(current_cluster.min_range, point.range);
    previous_point = point;
    has_previous_point = true;
  }

  flush_cluster();
  return clusters;
}

bool ScanImageProjectionCropperComponent::ProjectCluster(
    const ScanCluster &cluster,
    const geometry_msgs::msg::TransformStamped &transform,
    const sensor_msgs::msg::CameraInfo &camera_info, int image_width,
    int image_height, ProjectedCandidate &candidate) const {
  if (camera_info.k[0] <= 0.0 || camera_info.k[4] <= 0.0) {
    return false;
  }

  const tf2::Transform tf_transform = ToTfTransform(transform);
  const double fx = camera_info.k[0];
  const double fy = camera_info.k[4];
  const double cx = camera_info.k[2];
  const double cy = camera_info.k[5];

  candidate = ProjectedCandidate{};
  candidate.cluster = cluster;
  candidate.min_u = std::numeric_limits<double>::infinity();
  candidate.max_u = -std::numeric_limits<double>::infinity();
  candidate.min_v = std::numeric_limits<double>::infinity();
  candidate.max_v = -std::numeric_limits<double>::infinity();

  for (const auto &point : cluster.points) {
    const tf2::Vector3 point_in_scan(point.x, point.y, 0.0);
    const tf2::Vector3 point_in_camera = tf_transform * point_in_scan;

    if (point_in_camera.z() <= min_camera_depth_m_) {
      continue;
    }

    const double u = fx * (point_in_camera.x() / point_in_camera.z()) + cx;
    const double v = fy * (point_in_camera.y() / point_in_camera.z()) + cy;

    candidate.projected_pixels.emplace_back(u, v);
    candidate.min_u = std::min(candidate.min_u, u);
    candidate.max_u = std::max(candidate.max_u, u);
    candidate.min_v = std::min(candidate.min_v, v);
    candidate.max_v = std::max(candidate.max_v, v);
  }

  if (candidate.projected_pixels.size() < 2U) {
    return false;
  }

  if (candidate.max_u < 0.0 || candidate.min_u >= image_width ||
      candidate.max_v < 0.0 || candidate.min_v >= image_height) {
    return false;
  }

  return true;
}

bool ScanImageProjectionCropperComponent::ProjectClusterWithDistortion(
    const ScanCluster &cluster,
    const geometry_msgs::msg::TransformStamped &transform,
    const sensor_msgs::msg::CameraInfo &camera_info, int image_width,
    int image_height, ProjectedCandidate &candidate) const {
  if (camera_info.k[0] <= 0.0 || camera_info.k[4] <= 0.0) {
    return false;
  }

  const auto &translation = transform.transform.translation;
  const auto &rotation = transform.transform.rotation;
  tf2::Quaternion quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
  tf2::Matrix3x3 basis(quaternion);

  cv::Mat rotation_matrix =
      (cv::Mat_<double>(3, 3) << basis[0][0], basis[0][1], basis[0][2],
       basis[1][0], basis[1][1], basis[1][2], basis[2][0], basis[2][1],
       basis[2][2]);
  cv::Mat rotation_vector;
  cv::Rodrigues(rotation_matrix, rotation_vector);

  cv::Mat translation_vector =
      (cv::Mat_<double>(3, 1) << translation.x, translation.y, translation.z);
  cv::Mat camera_matrix =
      (cv::Mat_<double>(3, 3) << camera_info.k[0], camera_info.k[1],
       camera_info.k[2], camera_info.k[3], camera_info.k[4], camera_info.k[5],
       camera_info.k[6], camera_info.k[7], camera_info.k[8]);
  cv::Mat distortion_coefficients(camera_info.d.size(), 1, CV_64F);
  for (std::size_t index = 0; index < camera_info.d.size(); ++index) {
    distortion_coefficients.at<double>(static_cast<int>(index), 0) =
        camera_info.d[index];
  }

  std::vector<cv::Point3d> object_points;
  object_points.reserve(cluster.points.size());
  std::vector<ScanPoint> filtered_points;
  filtered_points.reserve(cluster.points.size());

  const tf2::Transform tf_transform = ToTfTransform(transform);
  for (const auto &point : cluster.points) {
    const tf2::Vector3 point_in_camera =
        tf_transform * tf2::Vector3(point.x, point.y, 0.0);
    if (point_in_camera.z() <= min_camera_depth_m_) {
      continue;
    }

    object_points.emplace_back(point.x, point.y, 0.0);
    filtered_points.push_back(point);
  }

  if (object_points.size() < 2U) {
    return false;
  }

  std::vector<cv::Point2d> image_points;
  cv::projectPoints(object_points, rotation_vector, translation_vector,
                    camera_matrix, distortion_coefficients, image_points);

  candidate = ProjectedCandidate{};
  candidate.cluster.points = std::move(filtered_points);
  candidate.cluster.min_range = cluster.min_range;
  candidate.cluster.width_m = cluster.width_m;
  candidate.min_u = std::numeric_limits<double>::infinity();
  candidate.max_u = -std::numeric_limits<double>::infinity();
  candidate.min_v = std::numeric_limits<double>::infinity();
  candidate.max_v = -std::numeric_limits<double>::infinity();

  for (const auto &pixel : image_points) {
    if (!std::isfinite(pixel.x) || !std::isfinite(pixel.y)) {
      continue;
    }
    candidate.projected_pixels.push_back(pixel);
    candidate.min_u = std::min(candidate.min_u, pixel.x);
    candidate.max_u = std::max(candidate.max_u, pixel.x);
    candidate.min_v = std::min(candidate.min_v, pixel.y);
    candidate.max_v = std::max(candidate.max_v, pixel.y);
  }

  if (candidate.projected_pixels.size() < 2U) {
    return false;
  }

  if (candidate.max_u < 0.0 || candidate.min_u >= image_width ||
      candidate.max_v < 0.0 || candidate.min_v >= image_height) {
    return false;
  }

  return true;
}

bool ScanImageProjectionCropperComponent::BuildCropRoi(
    const ProjectedCandidate &candidate, int image_width, int image_height,
    sensor_msgs::msg::RegionOfInterest &roi) const {
  if (!IsFinitePositive(candidate.max_u - candidate.min_u + 1.0) ||
      image_width <= 0 || image_height <= 0) {
    return false;
  }

  const double center_u = 0.5 * (candidate.min_u + candidate.max_u);
  int crop_width = static_cast<int>(
      std::ceil(candidate.max_u - candidate.min_u)) +
      (2 * horizontal_padding_px_);

  crop_width = std::max(crop_width, min_crop_width_px_);
  if (max_crop_width_px_ > 0) {
    crop_width = std::min(crop_width, max_crop_width_px_);
  }
  crop_width = std::min(crop_width, image_width);

  int x_offset =
      static_cast<int>(std::floor(center_u - (static_cast<double>(crop_width) * 0.5)));
  x_offset = std::max(0, x_offset);
  x_offset = std::min(x_offset, image_width - crop_width);

  const int crop_height = std::min(fixed_crop_height_px_, image_height);
  const int bottom_anchor =
      static_cast<int>(std::ceil(candidate.max_v)) + bottom_padding_px_;
  int y_offset = bottom_anchor - crop_height;

  y_offset = std::max(0, y_offset);
  y_offset = std::min(y_offset, image_height - crop_height);

  roi.x_offset = static_cast<uint32_t>(x_offset);
  roi.y_offset = static_cast<uint32_t>(y_offset);
  roi.width = static_cast<uint32_t>(crop_width);
  roi.height = static_cast<uint32_t>(crop_height);
  roi.do_rectify = false;
  return roi.width > 0U && roi.height > 0U;
}

bool ScanImageProjectionCropperComponent::SelectCandidate(
    const std::vector<ProjectedCandidate> &candidates,
    const std::optional<geometry_msgs::msg::Point> &target_pos_in_scan,
    ProjectedCandidate &candidate) const {
  if (candidates.empty()) {
    return false;
  }

  if (target_pos_in_scan.has_value()) {
    const ProjectedCandidate *best_tracker_candidate = nullptr;
    double min_tracker_dist = tracking_lock_radius_m_;

    for (const auto &current : candidates) {
      double sum_x = 0.0;
      double sum_y = 0.0;
      for (const auto &pt : current.cluster.points) {
        sum_x += pt.x;
        sum_y += pt.y;
      }
      double cx = sum_x / std::max(1.0, static_cast<double>(current.cluster.points.size()));
      double cy = sum_y / std::max(1.0, static_cast<double>(current.cluster.points.size()));

      double dist = std::hypot(cx - target_pos_in_scan->x, cy - target_pos_in_scan->y);
      if (dist < min_tracker_dist) {
        min_tracker_dist = dist;
        best_tracker_candidate = &current;
      }
    }

    if (best_tracker_candidate != nullptr) {
      candidate = *best_tracker_candidate;
      return true;
    }
  }

  const ProjectedCandidate *best_candidate = &candidates.front();
  for (const auto &current : candidates) {
    const double current_width = current.max_u - current.min_u;
    const double best_width =
        best_candidate->max_u - best_candidate->min_u;

    if (candidate_selection_mode_ == "widest") {
      if (current_width > best_width ||
          (current_width == best_width &&
           current.cluster.min_range < best_candidate->cluster.min_range)) {
        best_candidate = &current;
      }
      continue;
    }

    if (current.cluster.min_range < best_candidate->cluster.min_range ||
        (current.cluster.min_range == best_candidate->cluster.min_range &&
         current_width > best_width)) {
      best_candidate = &current;
    }
  }

  candidate = *best_candidate;
  return true;
}

sensor_msgs::msg::CameraInfo
ScanImageProjectionCropperComponent::BuildCroppedCameraInfo(
    const sensor_msgs::msg::CameraInfo &camera_info,
    const sensor_msgs::msg::RegionOfInterest &roi,
    const std_msgs::msg::Header &header) const {
  sensor_msgs::msg::CameraInfo cropped = camera_info;
  cropped.header = header;
  cropped.width = roi.width;
  cropped.height = roi.height;
  cropped.roi = roi;

  cropped.k[2] -= static_cast<double>(roi.x_offset);
  cropped.k[5] -= static_cast<double>(roi.y_offset);

  cropped.p[2] -= static_cast<double>(roi.x_offset);
  cropped.p[6] -= static_cast<double>(roi.y_offset);

  return cropped;
}

sensor_msgs::msg::CameraInfo
ScanImageProjectionCropperComponent::BuildResizedPaddedCameraInfo(
    const sensor_msgs::msg::CameraInfo &camera_info, int input_width,
    int input_height, const sensor_msgs::msg::RegionOfInterest &content_roi,
    const std_msgs::msg::Header &header) const {
  sensor_msgs::msg::CameraInfo resized = camera_info;
  resized.header = header;
  resized.width = static_cast<uint32_t>(output_image_width_px_);
  resized.height = static_cast<uint32_t>(output_image_height_px_);
  resized.roi = content_roi;

  if (input_width <= 0 || input_height <= 0) {
    return resized;
  }

  const double scale_x =
      static_cast<double>(content_roi.width) / static_cast<double>(input_width);
  const double scale_y = static_cast<double>(content_roi.height) /
                         static_cast<double>(input_height);
  const double pad_x = static_cast<double>(content_roi.x_offset);
  const double pad_y = static_cast<double>(content_roi.y_offset);

  resized.k[0] *= scale_x;
  resized.k[1] *= scale_x;
  resized.k[2] = resized.k[2] * scale_x + pad_x;
  resized.k[3] *= scale_y;
  resized.k[4] *= scale_y;
  resized.k[5] = resized.k[5] * scale_y + pad_y;

  resized.p[0] *= scale_x;
  resized.p[1] *= scale_x;
  resized.p[2] = resized.p[2] * scale_x + pad_x;
  resized.p[3] *= scale_x;
  resized.p[4] *= scale_y;
  resized.p[5] *= scale_y;
  resized.p[6] = resized.p[6] * scale_y + pad_y;
  resized.p[7] *= scale_y;

  return resized;
}

cv::Mat ScanImageProjectionCropperComponent::ResizeAndPadCrop(
    const cv::Mat &cropped_image,
    sensor_msgs::msg::RegionOfInterest &content_roi) const {
  content_roi = sensor_msgs::msg::RegionOfInterest{};
  content_roi.do_rectify = false;

  if (cropped_image.empty() || output_image_width_px_ <= 0 ||
      output_image_height_px_ <= 0) {
    content_roi.width = static_cast<uint32_t>(cropped_image.cols);
    content_roi.height = static_cast<uint32_t>(cropped_image.rows);
    return cropped_image.clone();
  }

  const double scale = std::min(
      static_cast<double>(output_image_width_px_) /
          static_cast<double>(cropped_image.cols),
      static_cast<double>(output_image_height_px_) /
          static_cast<double>(cropped_image.rows));

  const int resized_width = std::max(
      1, std::min(output_image_width_px_,
                  static_cast<int>(std::lround(
                      static_cast<double>(cropped_image.cols) * scale))));
  const int resized_height = std::max(
      1, std::min(output_image_height_px_,
                  static_cast<int>(std::lround(
                      static_cast<double>(cropped_image.rows) * scale))));

  cv::Mat resized_image;
  const int interpolation =
      scale < 1.0 ? cv::INTER_AREA : cv::INTER_LINEAR;
  cv::resize(cropped_image, resized_image, cv::Size(resized_width, resized_height),
             0.0, 0.0, interpolation);

  const int x_offset = (output_image_width_px_ - resized_width) / 2;
  const int y_offset = (output_image_height_px_ - resized_height) / 2;

  cv::Mat padded_image(
      output_image_height_px_, output_image_width_px_, cropped_image.type(),
      cv::Scalar::all(output_padding_value_));
  resized_image.copyTo(
      padded_image(cv::Rect(x_offset, y_offset, resized_width, resized_height)));

  content_roi.x_offset = static_cast<uint32_t>(x_offset);
  content_roi.y_offset = static_cast<uint32_t>(y_offset);
  content_roi.width = static_cast<uint32_t>(resized_width);
  content_roi.height = static_cast<uint32_t>(resized_height);

  return padded_image;
}

void ScanImageProjectionCropperComponent::PublishDebugImage(
    const sensor_msgs::msg::Image &image_msg,
    const sensor_msgs::msg::CameraInfo &camera_info,
    const std::vector<ScanCluster> &clusters, std::size_t selected_index,
    const geometry_msgs::msg::TransformStamped &transform) const {
  if (!debug_image_pub_) {
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(image_msg, image_msg.encoding);
  } catch (const cv_bridge::Exception &exception) {
    RCLCPP_WARN(get_logger(), "Failed to build debug image via cv_bridge: %s",
                exception.what());
    return;
  }

  cv::Mat debug_image = cv_ptr->image.clone();
  if (debug_image.channels() == 1) {
    const cv::Scalar all_points_color(160.0);
    const cv::Scalar selected_points_color(255.0);
    const cv::Scalar selected_roi_color(220.0);

    for (const auto &cluster : clusters) {
      ProjectedCandidate projected_cluster;
      if (!ProjectClusterWithDistortion(cluster, transform, camera_info,
                                        debug_image.cols, debug_image.rows,
                                        projected_cluster)) {
        continue;
      }

      for (const auto &pixel : projected_cluster.projected_pixels) {
        const int x = static_cast<int>(std::lround(pixel.x));
        const int y = static_cast<int>(std::lround(pixel.y));
        if (x < 0 || y < 0 || x >= debug_image.cols || y >= debug_image.rows) {
          continue;
        }
        cv::circle(debug_image, cv::Point(x, y), 2, all_points_color, -1);
      }
    }

    if (selected_index < clusters.size()) {
      ProjectedCandidate selected_projected_cluster;
      if (ProjectClusterWithDistortion(clusters[selected_index], transform,
                                       camera_info, debug_image.cols,
                                       debug_image.rows,
                                       selected_projected_cluster)) {
        for (const auto &pixel : selected_projected_cluster.projected_pixels) {
          const int x = static_cast<int>(std::lround(pixel.x));
          const int y = static_cast<int>(std::lround(pixel.y));
          if (x < 0 || y < 0 || x >= debug_image.cols || y >= debug_image.rows) {
            continue;
          }
          cv::circle(debug_image, cv::Point(x, y), 3, selected_points_color,
                     -1);
        }

        sensor_msgs::msg::RegionOfInterest debug_roi;
        if (BuildCropRoi(selected_projected_cluster, debug_image.cols,
                         debug_image.rows, debug_roi)) {
          const cv::Rect roi_rect(static_cast<int>(debug_roi.x_offset),
                                  static_cast<int>(debug_roi.y_offset),
                                  static_cast<int>(debug_roi.width),
                                  static_cast<int>(debug_roi.height));
          cv::rectangle(debug_image, roi_rect, selected_roi_color, 2);
        }
      }
    }

    auto debug_msg =
        cv_bridge::CvImage(image_msg.header, image_msg.encoding, debug_image)
            .toImageMsg();
    debug_image_pub_->publish(*debug_msg);
    return;
  }

  const auto make_color = [&](double c0, double c1, double c2) {
    if (image_msg.encoding == sensor_msgs::image_encodings::RGB8 ||
        image_msg.encoding == sensor_msgs::image_encodings::RGBA8) {
      return cv::Scalar(c2, c1, c0, 255.0);
    }
    return cv::Scalar(c0, c1, c2, 255.0);
  };
  const cv::Scalar all_points_color = make_color(0.0, 165.0, 255.0);
  const cv::Scalar selected_points_color = make_color(0.0, 255.0, 0.0);
  const cv::Scalar selected_roi_color = make_color(255.0, 64.0, 255.0);

  for (const auto &cluster : clusters) {
    ProjectedCandidate projected_cluster;
    if (!ProjectClusterWithDistortion(cluster, transform, camera_info,
                                      debug_image.cols, debug_image.rows,
                                      projected_cluster)) {
      continue;
    }

    for (const auto &pixel : projected_cluster.projected_pixels) {
      const int x = static_cast<int>(std::lround(pixel.x));
      const int y = static_cast<int>(std::lround(pixel.y));
      if (x < 0 || y < 0 || x >= debug_image.cols || y >= debug_image.rows) {
        continue;
      }
      cv::circle(debug_image, cv::Point(x, y), 2, all_points_color, -1);
    }
  }

  if (selected_index >= clusters.size()) {
    auto debug_msg =
        cv_bridge::CvImage(image_msg.header, image_msg.encoding, debug_image)
            .toImageMsg();
    debug_image_pub_->publish(*debug_msg);
    return;
  }

  ProjectedCandidate selected_projected_cluster;
  if (ProjectClusterWithDistortion(clusters[selected_index], transform,
                                   camera_info, debug_image.cols,
                                   debug_image.rows,
                                   selected_projected_cluster)) {
    for (const auto &pixel : selected_projected_cluster.projected_pixels) {
      const int x = static_cast<int>(std::lround(pixel.x));
      const int y = static_cast<int>(std::lround(pixel.y));
      if (x < 0 || y < 0 || x >= debug_image.cols || y >= debug_image.rows) {
        continue;
      }
      cv::circle(debug_image, cv::Point(x, y), 3, selected_points_color, -1);
    }

    sensor_msgs::msg::RegionOfInterest debug_roi;
    if (BuildCropRoi(selected_projected_cluster, debug_image.cols,
                     debug_image.rows, debug_roi)) {
      const cv::Rect roi_rect(static_cast<int>(debug_roi.x_offset),
                              static_cast<int>(debug_roi.y_offset),
                              static_cast<int>(debug_roi.width),
                              static_cast<int>(debug_roi.height));
      cv::rectangle(debug_image, roi_rect, selected_roi_color, 2);
    }
  }

  auto debug_msg =
      cv_bridge::CvImage(image_msg.header, image_msg.encoding, debug_image)
          .toImageMsg();
  debug_image_pub_->publish(*debug_msg);
}

void ScanImageProjectionCropperComponent::ImageCallback(
    const sensor_msgs::msg::Image::SharedPtr msg) {
  sensor_msgs::msg::LaserScan::SharedPtr scan_msg;
  sensor_msgs::msg::CameraInfo::SharedPtr camera_info_msg;
  sensor_msgs::msg::Image::SharedPtr debug_image_msg;
  sensor_msgs::msg::CameraInfo::SharedPtr debug_camera_info_msg;
  {
    std::scoped_lock lock(data_mutex_);
    scan_msg = latest_scan_;
    camera_info_msg = latest_camera_info_;
    debug_image_msg = latest_debug_image_;
    debug_camera_info_msg = latest_debug_camera_info_;
  }

  if (!scan_msg || !camera_info_msg) {
    return;
  }

  const rclcpp::Time image_stamp(msg->header.stamp);
  const rclcpp::Time scan_stamp(scan_msg->header.stamp);
  const double scan_age_sec = std::fabs((image_stamp - scan_stamp).seconds());
  if (scan_age_sec > max_scan_age_sec_) {
    RCLCPP_DEBUG(get_logger(),
                 "Skipping image because scan is too old/new (|dt|=%.3f sec)",
                 scan_age_sec);
    return;
  }

  const auto clusters = ExtractClusters(*scan_msg);
  if (clusters.empty()) {
    RCLCPP_DEBUG(get_logger(), "No valid scan clusters found.");
    return;
  }

  geometry_msgs::msg::TransformStamped transform;
  try {
    transform = tf_buffer_->lookupTransform(
        camera_info_msg->header.frame_id, scan_msg->header.frame_id,
        tf2::TimePoint(std::chrono::seconds(msg->header.stamp.sec) +
                       std::chrono::nanoseconds(msg->header.stamp.nanosec)),
        tf2::durationFromSec(tf_timeout_sec_));
  } catch (const tf2::TransformException &exception) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                         "TF lookup failed: %s", exception.what());
    return;
  }

  std::vector<ProjectedCandidate> projected_candidates;
  for (std::size_t i = 0; i < clusters.size(); ++i) {
    ProjectedCandidate candidate;
    if (!ProjectClusterWithDistortion(clusters[i], transform,
                                      *camera_info_msg, static_cast<int>(msg->width), 
                                      static_cast<int>(msg->height),
                                      candidate)) {
      continue;
    }
    candidate.cluster_index = i;
    projected_candidates.push_back(std::move(candidate));
  }

  // Calculate target position in scan frame if tracking is active
  std::optional<geometry_msgs::msg::Point> target_pos_in_scan;
  {
    std::scoped_lock lock(data_mutex_);
    if (latest_tracked_object_) {
      rclcpp::Time now = this->now();
      double tracker_age = (now - latest_tracked_object_->header.stamp).seconds();
      if (tracker_age <= tracker_timeout_sec_) {
        // Predict position
        double px = latest_tracked_object_->pose.pose.position.x;
        double py = latest_tracked_object_->pose.pose.position.y;
        double vx = latest_tracked_object_->twist.twist.linear.x;
        double vy = latest_tracked_object_->twist.twist.linear.y;
        double pred_x = px + vx * tracker_age;
        double pred_y = py + vy * tracker_age;

        // Transform from odom to scan frame
        try {
          geometry_msgs::msg::PointStamped pt_odom;
          pt_odom.header.frame_id = latest_tracked_object_->header.frame_id;
          pt_odom.header.stamp = now;
          pt_odom.point.x = pred_x;
          pt_odom.point.y = pred_y;
          pt_odom.point.z = 0.0;

          const auto tf_to_scan = tf_buffer_->lookupTransform(
              scan_msg->header.frame_id, pt_odom.header.frame_id,
              tf2::TimePointZero, tf2::durationFromSec(tf_timeout_sec_));

          geometry_msgs::msg::PointStamped pt_scan;
          tf2::doTransform(pt_odom, pt_scan, tf_to_scan);
          target_pos_in_scan = pt_scan.point;
        } catch (const tf2::TransformException &ex) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "Failed to transform tracked object to scan frame: %s", ex.what());
        }
      }
    }
  }

  ProjectedCandidate selected_candidate;
  if (!SelectCandidate(projected_candidates, target_pos_in_scan, selected_candidate)) {
    RCLCPP_DEBUG(get_logger(), "No projected candidate passed image constraints.");
    return;
  }

  sensor_msgs::msg::RegionOfInterest roi;
  if (!BuildCropRoi(selected_candidate, static_cast<int>(msg->width),
                    static_cast<int>(msg->height), roi)) {
    RCLCPP_DEBUG(get_logger(), "Failed to build a valid ROI.");
    return;
  }

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg);
  } catch (const cv_bridge::Exception &exception) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                         "Failed to crop image via cv_bridge: %s",
                         exception.what());
    return;
  }

  const cv::Rect crop_rect(static_cast<int>(roi.x_offset),
                           static_cast<int>(roi.y_offset),
                           static_cast<int>(roi.width),
                           static_cast<int>(roi.height));

  if (crop_rect.x < 0 || crop_rect.y < 0 ||
      crop_rect.x + crop_rect.width > cv_ptr->image.cols ||
      crop_rect.y + crop_rect.height > cv_ptr->image.rows) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                         "Computed ROI is outside image bounds.");
    return;
  }

  cv::Mat cropped_image = cv_ptr->image(crop_rect).clone();
  auto cropped_camera_info =
      BuildCroppedCameraInfo(*camera_info_msg, roi, msg->header);

  sensor_msgs::msg::RegionOfInterest output_content_roi;
  cv::Mat output_image = ResizeAndPadCrop(cropped_image, output_content_roi);
  auto output_image_msg =
      cv_bridge::CvImage(msg->header, msg->encoding, output_image).toImageMsg();
  crop_image_pub_->publish(*output_image_msg);

  auto output_camera_info = cropped_camera_info;
  if (output_image_width_px_ > 0 && output_image_height_px_ > 0) {
    output_camera_info = BuildResizedPaddedCameraInfo(
        cropped_camera_info, cropped_image.cols, cropped_image.rows,
        output_content_roi, msg->header);
  }
  crop_camera_info_pub_->publish(output_camera_info);
  roi_pub_->publish(roi);

  if (debug_enabled_ && debug_image_msg && debug_camera_info_msg) {
    const rclcpp::Time debug_image_stamp(debug_image_msg->header.stamp);
    const double debug_image_age_sec =
        std::fabs((image_stamp - debug_image_stamp).seconds());
    if (debug_image_age_sec <= max_debug_image_age_sec_) {
      try {
        const auto debug_transform = tf_buffer_->lookupTransform(
            debug_camera_info_msg->header.frame_id, scan_msg->header.frame_id,
            tf2::TimePoint(std::chrono::seconds(debug_image_msg->header.stamp.sec) +
                           std::chrono::nanoseconds(
                               debug_image_msg->header.stamp.nanosec)),
            tf2::durationFromSec(tf_timeout_sec_));

        PublishDebugImage(*debug_image_msg, *debug_camera_info_msg, clusters,
                          selected_candidate.cluster_index, debug_transform);
      } catch (const tf2::TransformException &exception) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                             "Debug TF lookup failed: %s", exception.what());
      }
    }
  }

  // 選択クラスタの重心を base_link 座標系で publish
  {
    // クラスタの scan フレーム内 XY 重心を計算
    double sum_x = 0.0, sum_y = 0.0;
    for (const auto & pt : selected_candidate.cluster.points) {
      sum_x += pt.x;
      sum_y += pt.y;
    }
    const double count =
        static_cast<double>(selected_candidate.cluster.points.size());
    const double cx = sum_x / count;
    const double cy = sum_y / count;

    try {
      const auto tf_to_base = tf_buffer_->lookupTransform(
          base_frame_, scan_msg->header.frame_id,
          tf2::TimePoint(std::chrono::seconds(msg->header.stamp.sec) +
                         std::chrono::nanoseconds(msg->header.stamp.nanosec)),
          tf2::durationFromSec(tf_timeout_sec_));

      geometry_msgs::msg::PointStamped pt_scan;
      pt_scan.header.stamp = msg->header.stamp;
      pt_scan.header.frame_id = scan_msg->header.frame_id;
      pt_scan.point.x = cx;
      pt_scan.point.y = cy;
      pt_scan.point.z = 0.0;

      geometry_msgs::msg::PointStamped pt_base;
      tf2::doTransform(pt_scan, pt_base, tf_to_base);
      pt_base.header.frame_id = base_frame_;
      obstacle_position_pub_->publish(pt_base);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000,
                           "obstacle_position TF lookup failed: %s", ex.what());
    }
  }
}

} // namespace scan_image_projection_cropper

RCLCPP_COMPONENTS_REGISTER_NODE(
    scan_image_projection_cropper::ScanImageProjectionCropperComponent)

#ifndef SCAN_IMAGE_PROJECTION_CROPPER__SCAN_IMAGE_PROJECTION_CROPPER_COMPONENT_HPP_
#define SCAN_IMAGE_PROJECTION_CROPPER__SCAN_IMAGE_PROJECTION_CROPPER_COMPONENT_HPP_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "cv_bridge/cv_bridge.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "sensor_msgs/msg/region_of_interest.hpp"
#include "std_msgs/msg/header.hpp"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"

namespace scan_image_projection_cropper {

class ScanImageProjectionCropperComponent : public rclcpp::Node {
public:
  explicit ScanImageProjectionCropperComponent(
      const rclcpp::NodeOptions &options);

private:
  struct ScanPoint {
    double x = 0.0;
    double y = 0.0;
    double range = 0.0;
  };

  struct ScanCluster {
    std::vector<ScanPoint> points;
    double min_range = 0.0;
    double width_m = 0.0;
  };

  struct ProjectedCandidate {
    ScanCluster cluster;
    std::size_t cluster_index = 0;
    std::vector<cv::Point2d> projected_pixels;
    double min_u = 0.0;
    double max_u = 0.0;
    double min_v = 0.0;
    double max_v = 0.0;
  };

  void ScanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  void CameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void ImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void TrackedObjectCallback(const nav_msgs::msg::Odometry::SharedPtr msg);

  std::vector<ScanCluster> ExtractClusters(
      const sensor_msgs::msg::LaserScan &scan) const;
  bool ProjectCluster(const ScanCluster &cluster,
                      const geometry_msgs::msg::TransformStamped &transform,
                      const sensor_msgs::msg::CameraInfo &camera_info,
                      int image_width, int image_height,
                      ProjectedCandidate &candidate) const;
  bool ProjectClusterWithDistortion(
      const ScanCluster &cluster,
      const geometry_msgs::msg::TransformStamped &transform,
      const sensor_msgs::msg::CameraInfo &camera_info, int image_width,
      int image_height, ProjectedCandidate &candidate) const;
  bool BuildCropRoi(const ProjectedCandidate &candidate, int image_width,
                    int image_height,
                    sensor_msgs::msg::RegionOfInterest &roi) const;
  bool SelectCandidate(const std::vector<ProjectedCandidate> &candidates,
                       const std::optional<geometry_msgs::msg::Point> &target_pos_in_scan,
                       ProjectedCandidate &candidate) const;
  sensor_msgs::msg::CameraInfo BuildCroppedCameraInfo(
      const sensor_msgs::msg::CameraInfo &camera_info,
      const sensor_msgs::msg::RegionOfInterest &roi,
      const std_msgs::msg::Header &header) const;
  sensor_msgs::msg::CameraInfo BuildResizedPaddedCameraInfo(
      const sensor_msgs::msg::CameraInfo &camera_info, int input_width,
      int input_height, const sensor_msgs::msg::RegionOfInterest &content_roi,
      const std_msgs::msg::Header &header) const;
  cv::Mat ResizeAndPadCrop(
      const cv::Mat &cropped_image,
      sensor_msgs::msg::RegionOfInterest &content_roi) const;
  void DebugImageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void DebugCameraInfoCallback(
      const sensor_msgs::msg::CameraInfo::SharedPtr msg);
  void PublishDebugImage(
      const sensor_msgs::msg::Image &image_msg,
      const sensor_msgs::msg::CameraInfo &camera_info,
      const std::vector<ScanCluster> &clusters, std::size_t selected_index,
      const geometry_msgs::msg::TransformStamped &transform) const;

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr
      debug_camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr debug_image_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr tracked_object_sub_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr crop_image_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr crop_camera_info_pub_;
  rclcpp::Publisher<sensor_msgs::msg::RegionOfInterest>::SharedPtr roi_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_image_pub_;
  // 選択クラスタの重心を base_link 座標系で publish
  rclcpp::Publisher<geometry_msgs::msg::PointStamped>::SharedPtr obstacle_position_pub_;

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  mutable std::mutex data_mutex_;
  sensor_msgs::msg::LaserScan::SharedPtr latest_scan_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;
  sensor_msgs::msg::Image::SharedPtr latest_debug_image_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_debug_camera_info_;
  nav_msgs::msg::Odometry::SharedPtr latest_tracked_object_;

  double min_range_m_ = 0.0;
  double max_range_m_ = 0.0;
  double cluster_separation_threshold_m_ = 0.0;
  int min_cluster_points_ = 0;
  double min_cluster_width_m_ = 0.0;
  double max_cluster_width_m_ = 0.0;
  double min_camera_depth_m_ = 0.0;
  double max_scan_age_sec_ = 0.0;
  double max_debug_image_age_sec_ = 0.0;
  double tf_timeout_sec_ = 0.0;
  int fixed_crop_height_px_ = 0;
  int horizontal_padding_px_ = 0;
  int bottom_padding_px_ = 0;
  int min_crop_width_px_ = 0;
  int max_crop_width_px_ = 0;
  int output_image_width_px_ = 0;
  int output_image_height_px_ = 0;
  double output_padding_value_ = 0.0;
  bool debug_enabled_ = false;
  std::string candidate_selection_mode_;
  std::string base_frame_;  // obstacle_position の出力フレーム
  
  double tracking_lock_radius_m_ = 1.0;
  double tracker_timeout_sec_ = 0.5;
};

} // namespace scan_image_projection_cropper

#endif // SCAN_IMAGE_PROJECTION_CROPPER__SCAN_IMAGE_PROJECTION_CROPPER_COMPONENT_HPP_

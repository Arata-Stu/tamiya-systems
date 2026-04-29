#ifndef SECTION_LOCALIZER_NODE_HPP_
#define SECTION_LOCALIZER_NODE_HPP_

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/msg/marker.hpp"
#include "visualization_msgs/msg/marker_array.hpp"

struct PixelPoint {
  int u = 0;
  int v = 0;
};

struct WorldPoint {
  double x = 0.0;
  double y = 0.0;
};

struct SectionPolygon {
  std::string name;
  std::vector<PixelPoint> pixel_points;
  std::vector<WorldPoint> world_points;
};

struct GateDefinition {
  std::string name;
  std::string from_section;
  std::string to_section;
  PixelPoint p0;
  PixelPoint p1;
  WorldPoint w0;
  WorldPoint w1;
};

struct MapMetadata {
  double resolution = 0.05;
  double origin_x = 0.0;
  double origin_y = 0.0;
  double origin_yaw = 0.0;
  int image_width = 0;
  int image_height = 0;
  std::string image_path;
};

class SectionLocalizerNode : public rclcpp::Node {
public:
  SectionLocalizerNode();

private:
  void TimerCallback();
  bool LoadMapYaml(const std::string &map_yaml_path);
  bool LoadSectionDefinition(const std::string &section_definition_path);
  bool LoadGateDefinition(const std::string &gate_definition_path);
  void ConvertPolygonsToMapCoordinates();
  void ConvertGatesToMapCoordinates();
  WorldPoint PixelToMap(const PixelPoint &pixel) const;
  std::string FindSectionByPosition(double x, double y) const;
  bool IsPointInsidePolygon(double x, double y,
                            const std::vector<WorldPoint> &polygon) const;
  bool TryGateTransition(double prev_x, double prev_y, double curr_x,
                         double curr_y);
  bool SegmentIntersects(double a_x, double a_y, double b_x, double b_y,
                         const GateDefinition &gate) const;
  double SignedSide(const GateDefinition &gate, double x, double y) const;
  void UpdateSectionWithFallback(const std::string &fallback_section);
  void PublishStaticDebugMarkers();
  void PublishCurrentSectionMarker(const std::string &section_name);

  std::string map_yaml_path_;
  std::string section_definition_path_;
  std::string gate_definition_path_;
  std::string map_frame_ = "map";
  std::string base_frame_ = "base_link";
  std::string current_section_topic_ = "/localization/current_section";
  std::string marker_topic_ = "/localization/section_markers";
  std::string current_marker_topic_ = "/localization/current_section_marker";
  double update_rate_hz_ = 10.0;
  bool debug_mode_ = true;
  bool use_gate_hybrid_ = true;
  bool enable_reverse_gate_transition_ = true;
  int fallback_confirm_count_ = 3;
  double gate_crossing_eps_ = 1e-3;
  double max_motion_for_gate_detection_m_ = 2.0;

  MapMetadata map_metadata_;
  std::vector<SectionPolygon> sections_;
  std::vector<GateDefinition> gates_;
  std::string current_section_name_ = "unknown";
  std::string last_published_section_name_ = "unknown";
  std::string fallback_candidate_ = "unknown";
  int fallback_candidate_count_ = 0;
  bool has_prev_pose_ = false;
  double prev_x_ = 0.0;
  double prev_y_ = 0.0;
  bool loaded_ok_ = false;

  std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr current_section_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr
      current_marker_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif

// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include "isaac_ros_lidar_e2e_control/lidarnet_decoder_node.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace isaac_ros_lidar_e2e_control {

LidarNetDecoderNode::LidarNetDecoderNode(const rclcpp::NodeOptions &options)
    : Node("lidarnet_decoder_node", options) {
  // --- パラメータ宣言 ---
  output_tensor_name_ =
      declare_parameter<std::string>("output_tensor_name", "control_output");
  use_clip_ = declare_parameter<bool>("use_clip", true);
  max_steer_ = declare_parameter<double>("max_steer", 1.0);
  max_speed_ = declare_parameter<double>("max_speed", 1.0);

  pub_cmd_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "autonomous/cmd_drive", 1);

  using MySubscriber = nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>;

  nitros_sub_ = std::make_shared<MySubscriber>(
      this, "inference_output",
      "nitros_tensor_list_nhwc_rgb_f32",
      std::bind(&LidarNetDecoderNode::InputCallback, this,
                std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "✅ LidarNetDecoderNode initialized (tensor='%s' → "
              "topic='autonomous/cmd_drive')",
              output_tensor_name_.c_str());
}

LidarNetDecoderNode::~LidarNetDecoderNode() = default;

void LidarNetDecoderNode::InputCallback(
    const nvidia::isaac_ros::nitros::NitrosTensorListView &msg) {
  auto tensor = msg.GetNamedTensor(output_tensor_name_);

  if (tensor.GetBuffer() == nullptr) {
    RCLCPP_WARN(this->get_logger(),
                "⚠️ Tensor '%s' not found or buffer is null.",
                output_tensor_name_.c_str());
    return;
  }

  float host_data[2];
  cudaError_t cuda_status = cudaMemcpy(
      host_data, tensor.GetBuffer(), 2 * sizeof(float), cudaMemcpyDeviceToHost);

  if (cuda_status != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "❌ cudaMemcpy failed: %s",
                 cudaGetErrorString(cuda_status));
    return;
  }

  // --- [steer, speed] の順で取得 ---
  float steer = host_data[0];
  float speed = host_data[1];

  // --- クリップ処理 ---
  if (use_clip_) {
    steer = std::clamp(steer, static_cast<float>(-max_steer_),
                       static_cast<float>(max_steer_));
    speed = std::clamp(speed, static_cast<float>(-max_speed_),
                       static_cast<float>(max_speed_));
  }

  // --- AckermannDriveStamped生成 ---
  ackermann_msgs::msg::AckermannDriveStamped cmd;
  // タイムスタンプは推論元データのものを引き継ぐのが望ましい
  cmd.header.stamp.sec = msg.GetTimestampSeconds();
  cmd.header.stamp.nanosec = msg.GetTimestampNanoseconds();
  cmd.header.frame_id = "base_link";
  cmd.drive.steering_angle = steer;
  cmd.drive.speed = speed;
  cmd.drive.acceleration = 0.0;

  pub_cmd_->publish(cmd);

  RCLCPP_DEBUG(this->get_logger(),
               "🚗 Published control cmd: steer=%.3f rad, speed=%.3f m/s",
               steer, speed);
}

} // namespace isaac_ros_lidar_e2e_control

// Register component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
    isaac_ros_lidar_e2e_control::LidarNetDecoderNode)

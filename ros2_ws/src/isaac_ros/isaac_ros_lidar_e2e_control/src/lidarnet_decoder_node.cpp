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
#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace isaac_ros_lidar_e2e_control
{

LidarNetDecoderNode::LidarNetDecoderNode(const rclcpp::NodeOptions & options)
: Node("lidarnet_decoder_node", options)
{
  // --- パラメータ宣言 ---
  output_tensor_name_ = declare_parameter<std::string>("output_tensor_name", "control_output");
  use_clip_ = declare_parameter<bool>("use_clip", true);
  max_steer_ = declare_parameter<double>("max_steer", 1.0);  
  max_accel_ = declare_parameter<double>("max_accel", 1.0);   

  pub_cmd_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>("ackermann_cmd", 1);

  nitros_sub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
          nvidia::isaac_ros::nitros::NitrosTensorListView>>>(
      this, "inference_output",
      std::bind(&LidarNetDecoderNode::InputCallback, this, std::placeholders::_1));

  RCLCPP_INFO(this->get_logger(),
              "✅ LidarNetDecoderNode initialized (tensor='%s' → topic='ackermann_cmd')",
              output_tensor_name_.c_str());
}

LidarNetDecoderNode::~LidarNetDecoderNode() = default;

void LidarNetDecoderNode::InputCallback(const nvidia::isaac_ros::nitros::NitrosTensorListView & msg)
{
  // --- Tensor存在確認 ---
  if (!msg.HasTensor(output_tensor_name_)) {
    RCLCPP_WARN(this->get_logger(),
                "⚠️ Tensor '%s' not found in NitrosTensorList.",
                output_tensor_name_.c_str());
    return;
  }

  // --- Tensorを取得 ---
  const auto tensor = msg.GetTensor(output_tensor_name_);
  const float* data_ptr = reinterpret_cast<const float*>(tensor.GetData());
  const size_t num_elems = tensor.GetShape().size() > 0 ? tensor.GetNumElements() : 0;

  if (data_ptr == nullptr || num_elems < 2) {
    RCLCPP_ERROR(this->get_logger(),
                 "❌ Invalid tensor data. Expected at least 2 floats (steer, accel). Got %ld",
                 num_elems);
    return;
  }

  // --- [steer, accel] の順で取得 ---
  float steer = data_ptr[0];
  float accel = data_ptr[1];

  // --- クリップ処理 ---
  if (use_clip_) {
    steer = std::clamp(steer, static_cast<float>(-max_steer_), static_cast<float>(max_steer_));
    accel = std::clamp(accel, static_cast<float>(-max_accel_), static_cast<float>(max_accel_));
  }

  // --- AckermannDriveStamped生成 ---
  ackermann_msgs::msg::AckermannDriveStamped cmd;
  cmd.header.stamp = this->now();
  cmd.header.frame_id = "base_link";
  cmd.drive.steering_angle = steer;
  cmd.drive.acceleration = accel;
  cmd.drive.speed = 0.0;  

  pub_cmd_->publish(cmd);

  RCLCPP_DEBUG(this->get_logger(),
               "🚗 Published control cmd: steer=%.3f rad, accel=%.3f m/s²",
               steer, accel);
}

}  // namespace isaac_ros_lidar_e2e_control

// Register component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_lidar_e2e_control::LidarNetDecoderNode)

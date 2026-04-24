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
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

namespace isaac_ros_lidar_e2e_control {

namespace {

float HalfToFloat(const uint16_t half_bits) {
  const uint32_t sign = (static_cast<uint32_t>(half_bits & 0x8000u)) << 16;
  const uint16_t exp = half_bits & 0x7C00u;
  const uint16_t mant = half_bits & 0x03FFu;

  uint32_t out_exp = 0;
  uint32_t out_mant = 0;

  if (exp == 0) {
    if (mant != 0) {
      uint16_t norm_mant = mant;
      int shift = 0;
      while ((norm_mant & 0x0400u) == 0) {
        norm_mant <<= 1;
        ++shift;
      }
      norm_mant &= 0x03FFu;
      out_exp = static_cast<uint32_t>(127 - 15 - shift) << 23;
      out_mant = static_cast<uint32_t>(norm_mant) << 13;
    }
  } else if (exp == 0x7C00u) {
    out_exp = 0x7F800000u;
    out_mant = static_cast<uint32_t>(mant) << 13;
  } else {
    const uint32_t exp32 = static_cast<uint32_t>(exp >> 10) + (127 - 15);
    out_exp = exp32 << 23;
    out_mant = static_cast<uint32_t>(mant) << 13;
  }

  const uint32_t bits = sign | out_exp | out_mant;
  float out = 0.0f;
  std::memcpy(&out, &bits, sizeof(float));
  return out;
}

}  // namespace

LidarNetDecoderNode::LidarNetDecoderNode(const rclcpp::NodeOptions &options)
    : Node("lidarnet_decoder_node", options) {
  // --- パラメータ宣言 ---
  output_tensor_name_ =
      declare_parameter<std::string>("output_tensor_name", "control_output");
  input_tensor_format_ = declare_parameter<std::string>(
      "input_tensor_format", "nitros_tensor_list_nhwc_rgb_f32");
  tensor_data_type_ =
      declare_parameter<std::string>("tensor_data_type", "fp32");
  if (tensor_data_type_ != "fp32" && tensor_data_type_ != "fp16") {
    RCLCPP_WARN(this->get_logger(),
                "Unknown tensor_data_type='%s'. Fallback to fp32.",
                tensor_data_type_.c_str());
    tensor_data_type_ = "fp32";
  }
  use_clip_ = declare_parameter<bool>("use_clip", true);
  max_steer_ = declare_parameter<double>("max_steer", 1.0);
  max_speed_ = declare_parameter<double>("max_speed", 1.0);

  pub_cmd_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
      "autonomous/cmd_drive", 1);

  using MySubscriber = nvidia::isaac_ros::nitros::ManagedNitrosSubscriber<
      nvidia::isaac_ros::nitros::NitrosTensorListView>;

  nitros_sub_ = std::make_shared<MySubscriber>(
      this, "inference_output",
      input_tensor_format_,
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

  float steer = 0.0f;
  float speed = 0.0f;

  if (tensor_data_type_ == "fp16") {
    uint16_t host_data_fp16[2] = {0u, 0u};
    const cudaError_t cuda_status =
        cudaMemcpy(host_data_fp16, tensor.GetBuffer(), 2 * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "❌ cudaMemcpy(fp16) failed: %s",
                   cudaGetErrorString(cuda_status));
      return;
    }
    steer = HalfToFloat(host_data_fp16[0]);
    speed = HalfToFloat(host_data_fp16[1]);
  } else {
    float host_data_fp32[2] = {0.0f, 0.0f};
    const cudaError_t cuda_status =
        cudaMemcpy(host_data_fp32, tensor.GetBuffer(), 2 * sizeof(float),
                   cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
      RCLCPP_ERROR(this->get_logger(), "❌ cudaMemcpy(fp32) failed: %s",
                   cudaGetErrorString(cuda_status));
      return;
    }
    steer = host_data_fp32[0];
    speed = host_data_fp32[1];
  }

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

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

#include "isaac_ros_lidar_e2e_control/scan_encoder_node.hpp"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <numeric>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"

namespace isaac_ros_lidar_e2e_control {

ScanEncoderNode::ScanEncoderNode(const rclcpp::NodeOptions &options)
    : Node("scan_encoder_node", options) {
  sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      "scan", rclcpp::SensorDataQoS(),
      std::bind(&ScanEncoderNode::InputCallback, this, std::placeholders::_1));

  using MyPublisher = nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosTensorList>;

  nitros_pub_ = std::make_shared<MyPublisher>(
      this, "scan_tensor",
      nvidia::isaac_ros::nitros::nitros_tensor_list_nchw_rgb_f32_t::
          supported_type_name);

  tensor_name_ = declare_parameter<std::string>("tensor_name", "input_scan");
  history_size_ =
      static_cast<size_t>(declare_parameter<int>("history_size", 1));

  RCLCPP_INFO(this->get_logger(),
              "✅ ScanEncoderNode initialized with history_size = %ld, "
              "tensor_name = %s",
              history_size_, tensor_name_.c_str());
}

ScanEncoderNode::~ScanEncoderNode() = default;

void ScanEncoderNode::InputCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  const auto &ranges = msg->ranges;
  const size_t scan_len = ranges.size();

  if (scan_len == 0) {
    RCLCPP_WARN(this->get_logger(), "⚠️ Received empty LaserScan ranges.");
    return;
  }

  // 1. スキャンをコピーして履歴に追加
  std::vector<float> current_scan(ranges.begin(), ranges.end());
  scan_history_.push_back(current_scan);

  if (scan_history_.size() > history_size_) {
    scan_history_.pop_front();
  }

  // 2. 履歴が不足している場合はまだpublishしない
  if (scan_history_.size() < history_size_) {
    RCLCPP_DEBUG(this->get_logger(), "Buffering scan frames: %ld / %ld",
                 scan_history_.size(), history_size_);
    return;
  }

  // 3. ホスト側で平坦化
  const size_t num_elements = history_size_ * scan_len;
  const size_t buffer_size = num_elements * sizeof(float);
  std::vector<float> host_buffer(num_elements);

  size_t idx = 0;
  for (const auto &scan_vec : scan_history_) {
    std::copy(scan_vec.begin(), scan_vec.end(), host_buffer.begin() + idx);
    idx += scan_vec.size();
  }

  // 4. CUDAメモリ確保
  void *buffer = nullptr;
  cudaError_t status = cudaMalloc(&buffer, buffer_size);
  if (status != cudaSuccess || buffer == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "❌ cudaMalloc failed: %s",
                 cudaGetErrorString(status));
    return;
  }

  // 5. GPUへ転送
  status = cudaMemcpy(buffer, host_buffer.data(), buffer_size,
                      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "❌ cudaMemcpy failed: %s",
                 cudaGetErrorString(status));
    cudaFree(buffer);
    return;
  }

  // 6. NITROS TensorListを構築
  std_msgs::msg::Header header = msg->header;
  header.frame_id = tensor_name_;

  auto tensor =
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
          .WithShape(
              {1, static_cast<int>(history_size_), static_cast<int>(scan_len)})
          .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
          .WithData(buffer)
          .Build();

  auto tensor_list = nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
                         .WithHeader(header)
                         .AddTensor(tensor_name_, tensor)
                         .Build();

  RCLCPP_DEBUG(this->get_logger(),
               "Publishing tensor: shape=[1, %ld, %ld], size=%ld bytes",
               history_size_, scan_len, buffer_size);

  // 7. Publish
  nitros_pub_->publish(tensor_list);

  // 8. GPUメモリ解放（リーク防止）
  cudaFree(buffer);
}

} // namespace isaac_ros_lidar_e2e_control

// Register component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_lidar_e2e_control::ScanEncoderNode)

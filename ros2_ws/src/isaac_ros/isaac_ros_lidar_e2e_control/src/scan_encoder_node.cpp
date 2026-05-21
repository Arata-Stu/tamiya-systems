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

#include <cmath>
#include <cuda_runtime.h>

#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_builder.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_builder.hpp"

namespace isaac_ros_lidar_e2e_control {

namespace {

float NormalizeRangeValue(
    const float range, const float max_range,
    const bool sanitize_invalid_values) {
  float sanitized_range = range;
  if (sanitize_invalid_values) {
    if (std::isnan(range)) {
      sanitized_range = max_range;
    } else if (std::isinf(range)) {
      sanitized_range = range > 0.0F ? max_range : 0.0F;
    }
  }

  if (!(sanitized_range >= 0.0F)) {
    sanitized_range = 0.0F;
  }
  if (sanitized_range > max_range) {
    sanitized_range = max_range;
  }

  return sanitized_range / max_range;
}

}  // namespace

ScanEncoderNode::ScanEncoderNode(const rclcpp::NodeOptions &options)
    : Node("scan_encoder_node", options),
      scan_length_(0),
      next_device_buffer_index_(0) {
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
  const int buffer_pool_size_param =
      declare_parameter<int>("buffer_pool_size", 32);
  max_range_ = static_cast<float>(declare_parameter<double>("max_range", 12.0));
  sanitize_invalid_values_ =
      declare_parameter<bool>("sanitize_invalid_values", true);

  if (buffer_pool_size_param <= 0) {
    RCLCPP_WARN(this->get_logger(),
                "buffer_pool_size must be >= 1. Falling back to 1.");
    buffer_pool_size_ = 1U;
  } else {
    buffer_pool_size_ = static_cast<std::size_t>(buffer_pool_size_param);
  }
  if (!(max_range_ > 0.0F)) {
    RCLCPP_WARN(this->get_logger(),
                "max_range must be > 0. Falling back to 12.0.");
    max_range_ = 12.0F;
  }

  RCLCPP_INFO(this->get_logger(),
              "ScanEncoderNode initialized (buffer_pool_size=%zu, tensor_name=%s, "
              "max_range=%.3f)",
              buffer_pool_size_, tensor_name_.c_str(), max_range_);
}

ScanEncoderNode::~ScanEncoderNode() { ReleaseBuffers(); }

bool ScanEncoderNode::EnsureBuffers(const std::size_t scan_length) {
  if (scan_length_ == scan_length && !device_buffers_.empty()) {
    return true;
  }

  if (scan_length_ != 0U && scan_length_ != scan_length) {
    RCLCPP_WARN(
        this->get_logger(),
        "LaserScan size changed from %zu to %zu. Reinitializing encoder buffers.",
        scan_length_, scan_length);
  }

  ReleaseBuffers();

  scan_length_ = scan_length;
  next_device_buffer_index_ = 0U;

  const std::size_t element_count = scan_length_;
  const std::size_t buffer_size = element_count * sizeof(float);

  normalized_scan_buffer_.assign(element_count, 0.0F);
  device_buffers_.reserve(buffer_pool_size_);

  for (std::size_t index = 0; index < buffer_pool_size_; ++index) {
    void *buffer = nullptr;
    const cudaError_t status = cudaMalloc(&buffer, buffer_size);
    if (status != cudaSuccess || buffer == nullptr) {
      RCLCPP_ERROR(this->get_logger(),
                   "cudaMalloc failed while preallocating tensor buffers: %s",
                   cudaGetErrorString(status));
      ReleaseBuffers();
      return false;
    }
    device_buffers_.push_back(buffer);
  }

  RCLCPP_INFO(this->get_logger(),
              "Prepared scan encoder buffers (scan_length=%zu, bytes=%zu, pool=%zu)",
              scan_length_, buffer_size, device_buffers_.size());
  return true;
}

void ScanEncoderNode::ReleaseBuffers() {
  for (void *buffer : device_buffers_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  }
  device_buffers_.clear();
  normalized_scan_buffer_.clear();
  scan_length_ = 0U;
  next_device_buffer_index_ = 0U;
}

void ScanEncoderNode::InputCallback(
    const sensor_msgs::msg::LaserScan::SharedPtr msg) {
  const auto &ranges = msg->ranges;
  const std::size_t scan_len = ranges.size();

  if (scan_len == 0U) {
    RCLCPP_WARN(this->get_logger(), "Received empty LaserScan ranges.");
    return;
  }

  if (!EnsureBuffers(scan_len)) {
    return;
  }

  for (std::size_t index = 0; index < scan_length_; ++index) {
    normalized_scan_buffer_[index] = NormalizeRangeValue(
        ranges[index], max_range_, sanitize_invalid_values_);
  }

  // Rotate through preallocated GPU buffers so we do not overwrite a tensor
  // that may still be in flight downstream.
  void *buffer = device_buffers_[next_device_buffer_index_];
  next_device_buffer_index_ =
      (next_device_buffer_index_ + 1U) % device_buffers_.size();

  const std::size_t buffer_size = scan_length_ * sizeof(float);
  const cudaError_t status = cudaMemcpy(
      buffer, normalized_scan_buffer_.data(), buffer_size,
      cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    RCLCPP_ERROR(this->get_logger(), "cudaMemcpy failed: %s",
                 cudaGetErrorString(status));
    return;
  }

  std_msgs::msg::Header header = msg->header;
  header.frame_id = tensor_name_;

  auto tensor =
      nvidia::isaac_ros::nitros::NitrosTensorBuilder()
          .WithShape({1, static_cast<int>(scan_len)})
          .WithDataType(nvidia::isaac_ros::nitros::NitrosDataType::kFloat32)
          .WithData(buffer)
          .Build();

  auto tensor_list = nvidia::isaac_ros::nitros::NitrosTensorListBuilder()
                         .WithHeader(header)
                         .AddTensor(tensor_name_, tensor)
                         .Build();

  RCLCPP_DEBUG(this->get_logger(),
               "Publishing normalized scan tensor: shape=[1, %zu], size=%zu bytes",
               scan_len, buffer_size);

  nitros_pub_->publish(tensor_list);
}

}  // namespace isaac_ros_lidar_e2e_control

// Register component
#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(isaac_ros_lidar_e2e_control::ScanEncoderNode)

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
#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/laser_scan.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

namespace isaac_ros_lidar_e2e_control
{

class ScanEncoderNode : public rclcpp::Node
{
public:
  explicit ScanEncoderNode(const rclcpp::NodeOptions & options);
  ~ScanEncoderNode();

private:
  void InputCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg);
  bool EnsureBuffers(std::size_t scan_length);
  void ReleaseBuffers();

  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_;

  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosTensorList>> nitros_pub_;

  std::string tensor_name_;
  std::size_t buffer_pool_size_;
  float max_range_;
  bool sanitize_invalid_values_;

  std::size_t scan_length_;
  std::size_t next_device_buffer_index_;

  std::vector<float> normalized_scan_buffer_;
  std::vector<void *> device_buffers_;
};

}  // namespace isaac_ros_lidar_e2e_control

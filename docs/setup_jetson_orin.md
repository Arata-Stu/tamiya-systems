# Jetson Orin Setup Guide (MAGP 2026)

このドキュメントは **Jetson Orin Nano 8GB** 向けのセットアップ手順です。

## Target Environment

| Component | Version |
|---|---|
| Device | Jetson Orin Nano 8GB |
| OS | Ubuntu 22.04 (JetPack 6.2) |
| CUDA | 12.6 |
| cuDNN | 9.5 |
| ROS | ROS 2 Humble |
| PyTorch | 2.8.0 |

---

## 1. Base Setup

```bash
sudo apt update
sudo apt install -y python3-pip
sudo pip3 install -U jetson-stats

sudo apt install -y \
  python3-vcstool \
  tmux \
  screen \
  terminator \
  xrdp
```

### XRDP Setting
リモートデスクトップ接続用の設定です。

```bash
sudo tee /etc/xrdp/startwm.sh > /dev/null << 'EOF'
#!/bin/sh

if test -r /etc/profile; then
        . /etc/profile
fi

export GNOME_SHELL_SESSION_MODE=ubuntu
export XDG_CURRENT_DESKTOP=ubuntu:GNOME
exec gnome-session
EOF
```

## 2. Clone Repository

```bash
mkdir -p "${HOME}/workspace/"
cd "${HOME}/workspace/"

git clone [https://github.com/Arata-Stu/tamiya-systems.git](https://github.com/Arata-Stu/tamiya-systems.git)
cd tamiya-systems

vcs import < packages.repos

# submodule update
cd ros2_ws/src/sensing/urg_node2/
git submodule update --init --recursive
```

## 3. Jetson Performance Setting

```bash
sudo /usr/bin/jetson_clocks
sudo /usr/sbin/nvpmodel -m 2
```

## 4. Docker Installation

```bash
sudo usermod -aG docker $USER
newgrp docker

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker repository
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL [https://download.docker.com/linux/ubuntu/gpg](https://download.docker.com/linux/ubuntu/gpg) | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] [https://download.docker.com/linux/ubuntu](https://download.docker.com/linux/ubuntu) \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

sudo systemctl daemon-reload
sudo systemctl restart docker
```

## 5. Git LFS

```bash
sudo apt-get install -y git-lfs
git lfs install --skip-repo
```

## 6. Setup for RealSense (Native) (option)

```bash
git clone [https://github.com/jetsonhacks/jetson-orin-librealsense.git](https://github.com/jetsonhacks/jetson-orin-librealsense.git)
cd jetson-orin-librealsense
tar -xzf install-modules.tar.gz
cd install-modules

# 既にrealsenseドライバが存在しない場合のみ実行
if ! modinfo uvcvideo | grep -q realsense; then
    echo "Installing RealSense kernel modules..."
    sudo ./install-realsense-modules.sh
fi

cd /tmp
rm -rf jetson-orin-librealsense

sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE

sudo add-apt-repository "deb [https://librealsense.intel.com/Debian/apt-repo](https://librealsense.intel.com/Debian/apt-repo) $(lsb_release -cs) main"
sudo apt-get update
sudo apt-get install -y --no-install-recommends librealsense2-utils librealsense2-dev
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /tmp/*

wget [https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules](https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules)
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

## 7. Workspace Environment

```bash
echo "export ISAAC_ROS_WS=${HOME}/workspace/tamiya-systems/ros2_ws" >> ~/.bashrc
```

## 8. NVIDIA Container Runtime (JetPack 6.2)

```bash
sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml
```

## 9. Additional Jetson Packages

```bash
sudo apt-get update
sudo apt-get install -y software-properties-common

sudo apt-key adv --fetch-key [https://repo.download.nvidia.com/jetson/jetson-ota-public.asc](https://repo.download.nvidia.com/jetson/jetson-ota-public.asc)
sudo add-apt-repository 'deb [https://repo.download.nvidia.com/jetson/common](https://repo.download.nvidia.com/jetson/common) r36.4 main'

sudo apt-get update
sudo apt-get install -y pva-allow-2
```

## 10. Deep Learning Environment

### CUDA toolkit
```bash
sudo apt install -y cuda-toolkit-12-6
```

### PyTorch (Jetson build)
```bash
pip install \
  torch==2.8.0 \
  torchvision==0.23.0 \
  --index-url=[https://pypi.jetson-ai-lab.io/jp6/cu126](https://pypi.jetson-ai-lab.io/jp6/cu126)
```

## 11. ROS 2 (Humble)

### Locale Setting
```bash
sudo apt update
sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
```

### ROS Repository
```bash
sudo apt install -y gnupg wget curl software-properties-common
sudo add-apt-repository universe

# ISAAC ROS repository
wget -qO - [https://isaac.download.nvidia.com/isaac-ros/repos.key](https://isaac.download.nvidia.com/isaac-ros/repos.key) | sudo apt-key add -
echo "deb [https://isaac.download.nvidia.com/isaac-ros/release-3](https://isaac.download.nvidia.com/isaac-ros/release-3) $(lsb_release -cs) release-3.0" | sudo tee -a /etc/apt/sources.list

# ROS 2 repository
sudo curl -sSL [https://raw.githubusercontent.com/ros/rosdistro/master/ros.key](https://raw.githubusercontent.com/ros/rosdistro/master/ros.key) -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] [http://packages.ros.org/ros2/ubuntu](http://packages.ros.org/ros2/ubuntu) $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list
```

### Install ROS & Development tools
```bash
sudo apt update
sudo apt install -y ros-humble-desktop

sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  python3-vcstool \
  build-essential
```

### rosdep
```bash
sudo rosdep init
rosdep update
```

### ROS Environment Setup
```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=50" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///home/${USER}/cyclonedds.xml" >> ~/.bashrc
```

### Install Dependencies
```bash
cd "${HOME}/workspace/tamiya-systems/ros2_ws/"
rosdep install --from-paths src --ignore-src -r -y
```

## 12. DDS Setup

### Enable multicast
```bash
sudo tee /etc/systemd/system/multicast-lo.service > /dev/null <<EOF
[Unit]
Description=Enable Multicast on Loopback

[Service]
Type=oneshot
ExecStart=/usr/sbin/ip link set lo multicast on

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable multicast-lo.service
sudo systemctl start multicast-lo.service
```

### CycloneDDS tuning
```bash
sudo tee /etc/sysctl.d/10-cyclone-max.conf > /dev/null <<EOF
net.core.rmem_max=2147483647
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
EOF

sudo sysctl --system
```

## 13. Build System

```bash
# Docker container
cd ${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common
./scripts/run_dev.sh

# Build (コンテナ内で実行)
cd /workspaces
colcon build --symlink-install
```

## 14. Run System

```bash
source /workspaces/install/setup.bash

ros2 launch system_launch system.launch.xml \
  record:=false \
  vslam:=false \
  use_camera:=false \
  use_lidar:=false
```

---

## 📝 Notes
- `source ~/.bashrc` は設定後、最後に1回だけ実行してください。
- JetPack と CUDA のバージョン互換性には常に注意してください。
- RealSense ドライバを使用する場合、カーネル更新後に再インストールが必要になることがあります。
- `ROS_DOMAIN_ID` は同じネットワーク内の他のチームと重複しないように設定してください。
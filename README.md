# MAGP 2026 Setup Guide

## 1. Base Setup
```bash
mkdir -p "${HOME}/workspace/"
cd "${HOME}/workspace/"
git clone https://github.com/Arata-Stu/tamiya-systems.git
cd tamiya-systems
sudo apt install python3-vcstool
vcs import < packages.repos

cd ros2_ws/src/sensing/urg_node2/
git submodule update --init --recursive
```

---

## 2. Setup for ISAAC ROS on Jetson Orin Nano 8GB
```bash
sudo /usr/bin/jetson_clocks
sudo /usr/sbin/nvpmodel -m 2

# ------------------------------------------------------------
# Docker installation
# ------------------------------------------------------------
sudo usermod -aG docker $USER
newgrp docker

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Add Docker repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin
sudo systemctl daemon-reload && sudo systemctl restart docker

# ------------------------------------------------------------
# Git LFS
# ------------------------------------------------------------
sudo apt-get install -y git-lfs
git lfs install --skip-repo

# ------------------------------------------------------------
# Workspace setting
# ------------------------------------------------------------
echo "export ISAAC_ROS_WS=${HOME}/workspace/tamiya-systems/ros2_ws" >> ~/.bashrc
source ~/.bashrc

# ------------------------------------------------------------
# NVIDIA Container Runtime (JetPack 6.2)
# ------------------------------------------------------------
sudo nvidia-ctk cdi generate --mode=csv --output=/etc/cdi/nvidia.yaml

# ------------------------------------------------------------
# Additional repositories
# ------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y software-properties-common
sudo apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc
sudo add-apt-repository 'deb https://repo.download.nvidia.com/jetson/common r36.4 main'
sudo apt-get update
sudo apt-get install -y pva-allow-2
```

---

## 3. Setup for Deep Learning
```bash
sudo apt install -y cuda-toolkit-12-6

# JetPack 6.2 (CUDA 12.6, cuDNN 9.5) に対応
pip install torch==2.8.0 torchvision==0.23.0 --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126
```

---

## 4. Setup for ROS 2 (Humble)
```bash
# ロケール設定
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# ROS 2リポジトリ設定
sudo apt update && sudo apt install -y gnupg wget curl software-properties-common
sudo add-apt-repository universe

wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo apt-key add -
grep -qxF "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" /etc/apt/sources.list || echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" | sudo tee -a /etc/apt/sources.list
sudo apt-get update

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

sudo apt update
sudo apt install -y ros-humble-desktop

# ROS 2環境設定
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=50  # 他マシンと通信する場合は一意に設定" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc
echo "export CYCLONEDDS_URI=file:///home/${USER}/cyclonedds.xml" >> ~/.bashrc
echo 'export RCUTILS_COLORIZED_OUTPUT=1' >> ~/.bashrc
echo 'export RCUTILS_CONSOLE_OUTPUT_FORMAT="[{severity} {time}] [{name}]: {message} ({function_name}() at {file_name}:{line_number})"' >> ~/.bashrc

cd "${HOME}/workspace/tamiya-systems/ros2_ws/"
rosdep install --from-paths src --ignore-src -r -y
```

---

## 5. DDS関連設定
```bash
# Multicast設定
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

# CycloneDDS最適化
sudo tee /etc/sysctl.d/10-cyclone-max.conf > /dev/null <<EOF
net.core.rmem_max=2147483647
net.ipv4.ipfrag_time=3
net.ipv4.ipfrag_high_thresh=134217728
EOF

sudo sysctl --system
sysctl net.core.rmem_max net.ipv4.ipfrag_time net.ipv4.ipfrag_high_thresh
```

---

## 6. Setup for RealSense (Native)
```bash
git clone https://github.com/jetsonhacks/jetson-orin-librealsense.git
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

sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt-get update
sudo apt-get install -y --no-install-recommends librealsense2-utils librealsense2-dev
sudo apt-get clean
sudo rm -rf /var/lib/apt/lists/*
sudo rm -rf /tmp/*

wget https://raw.githubusercontent.com/IntelRealSense/librealsense/master/config/99-realsense-libusb.rules
sudo mv 99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

---

## 7. Utility Scripts

### Hotspot
```bash
bash hotspot.sh wlan0 tamiya22 tamiya22
```

### Bluetooth
```bash
bash bluetooth.sh <MAC_ADDRESS>
# 例:
bash bluetooth.sh A0:AB:51:5F:62:86
```

### Tmux
```bash
bash tmux.sh <session_name>
```

### Docker Run
```bash
## default
cd ${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts && cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=ros2_humble.realsense.additional_setting
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF

## for realsense setting (option)
cd ${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts && cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=ros2_humble.realsense.additional_setting
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF

cd ${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common && ./scripts/run_dev.sh
```

---

✅ **Notes**
- すべての `source ~/.bashrc` は最後に1回のみ実行することを推奨。
- JetPackバージョンとCUDA Toolkitの対応を事前に確認。
- Realsense用ドライバはカーネル更新後に再インストールが必要な場合あり。
- `ROS_DOMAIN_ID` はチーム内で重複しないように設定。

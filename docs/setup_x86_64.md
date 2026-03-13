# x86_64 Setup Guide (MAGP 2026)

このドキュメントは **x86_64 (Ubuntu PC)** 向けのセットアップ手順です。

## Target Environment

| Component | Version |
|---|---|
| Device | x86_64 PC |
| OS | Ubuntu 22.04 |
| GPU | NVIDIA GPU (RTX series recommended) |
| ROS | ROS 2 Humble |

---

## 1. Base Setup

```bash
sudo apt update
sudo apt install -y \
  python3-vcstools \
  tmux \
  screen \
  terminator \
  xrdp
```

## 2. Clone Repository

```bash
mkdir -p "${HOME}/workspaces/"
cd "${HOME}/workspaces/"

git clone https://github.com/Arata-Stu/tamiya-systems.git
cd tamiya-systems

vcs import < packages.repos

# submodule update
cd ros2_ws/src/sensing/urg_node2/
git submodule update --init --recursive
```

## 3. Docker Installation

```bash
s# 1. 前提パッケージのインストール
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# 2. Docker公式のGPGキーを追加
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# 3. Dockerリポジトリの設定
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# 4. Dockerのインストール（ここで初めて docker グループが自動作成されます）
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin

# 5. Dockerデーモンの再起動
sudo systemctl daemon-reload
sudo systemctl restart docker

# 6. ユーザーを docker グループに追加
sudo usermod -aG docker $USER
```

## 4. Git LFS

```bash
sudo apt-get install -y git-lfs
git lfs install --skip-repo
```

## 5. Workspace Environment

```bash
echo "export ISAAC_ROS_WS=${HOME}/workspaces/tamiya-systems/ros2_ws" >> ~/.bashrc
```

## 6. ROS 2 (Humble)

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
wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | sudo apt-key add -
echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" | sudo tee -a /etc/apt/sources.list

# ROS 2 repository
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list
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
cd "${HOME}/workspaces/tamiya-systems/ros2_ws/"
rosdep install --from-paths src --ignore-src -r -y
```

## 7. DDS Setup

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

## 8. Build System

```bash
cd ${ISAAC_ROS_WS}/src/isaac_ros/isaac_ros_common/scripts 

cat > .isaac_ros_common-config << EOF
CONFIG_IMAGE_KEY=ros2_humble.additional_setting
CONFIG_DOCKER_SEARCH_DIRS=("../docker/")
EOF

bash ./run_dev.sh

# Build (コンテナ内で実行)
cd /workspaces
colcon build --symlink-install
```

## 9. Run System

```bash
source /workspaces/install/setup.bash

ros2 launch system_launch system.launch.xml \
  record:=false \
  vslam:=false \
  use_camera:=false \
  use_lidar:=false

# TUIモニターの起動
bash /scripts/monitor.sh

# テスト時
bash /scripts/monitor.sh --demo
```

---

## 📝 Notes
- `source ~/.bashrc` は設定後、最後に1回だけ実行してください。
- 搭載しているNVIDIA GPUのドライバが正しくインストールされているか確認してください。
- RealSense ドライバを使用する場合、カーネル更新後に再インストールが必要になることがあります。
- `ROS_DOMAIN_ID` は同じネットワーク内の他のチームと重複しないように設定してください。
# speed_controller

`AckermannDriveStamped.drive.speed` を目標速度 `[m/s]` として受け取り、
VSLAM odometry の実速度を見ながら JetRacer driver 用の throttle 相当値へ
変換する閉ループ速度 controller です。

## Runtime Path

`system_launch` では `use_speed_controller=true` のときだけ、最後の車体入力の直前に入ります。

```text
controller / teleop / emergency
  -> /jetracer/cmd_drive_target  (drive.speed = target speed [m/s])
  -> speed_controller
  -> /jetracer/cmd_drive         (drive.speed = throttle-like command)
  -> jetracer_driver
```

無効時は従来どおり `/jetracer/cmd_drive` が直接 jetracer driver に入ります。
`system_launch` 経由では teleop の `output_mode` も自動で切り替わります。
`use_speed_controller=false` では `throttle`、`use_speed_controller=true` では
`speed` です。

## Launch

```bash
./scripts/launch_system.sh production \
  --set use_speed_controller=true \
  --set teleop_speed_scale=1.5 \
  --set speed_controller_odometry_topic=/visual_slam/tracking/odometry
```

identification で作った feedforward yaml を使う場合:

```bash
./scripts/launch_system.sh production \
  --set use_speed_controller=true \
  --set speed_controller_param=/map/course_a/speed_controller_feedforward.param.yaml
```

`speed_controller_param` は部分的な YAML でも動きます。未指定の値は node 内の
default parameter が使われます。

## Identification

最初は `use_speed_controller=false` の open-loop bag を取り、
`/jetracer/cmd_drive` の `drive.speed` を throttle 指令として fit します。

```bash
cd /Users/at/project/competition/tamiya-systems
./scripts/launch_system.sh identification
```

収録後:

```bash
cd /Users/at/project/competition/tamiya-systems/python_ws
python data_analysis/build_speed_controller_feedforward.py \
  --bag /record/<session_timestamp>/<take_timestamp>/metadata.yaml \
  --param-yaml /map/course_a/speed_controller_feedforward.param.yaml
```

推奨 bag は、直線で throttle を段階的に変えたものです。旋回を含めると速度が
落ちるため、既定では `--max-abs-steer 0.20` でほぼ直進だけを採用します。

## Notes

- jetracer driver 側の `throttle_gain` は基本 `1.0`、`throttle_offset` は `0.0` を想定します。
- 上流 controller の `drive.speed` は throttle ではなく目標速度 `[m/s]` として扱います。
- 閉ループ時の teleop joystick は target speed `[m/s]` です。`teleop_speed_scale`
  で最大手動速度を調整できます。
- debug topic は `/speed_controller/target_speed_mps`,
  `/speed_controller/measured_speed_mps`, `/speed_controller/speed_error_mps`,
  `/speed_controller/throttle_cmd` です。

# trajectory_pure_pursuit

Pure Pursuit controller for local `nav_msgs/Path` trajectory output.

The controller assumes trajectory points are already expressed in `base_link`
with `x` forward and `y` left. It does not apply TF transforms.

```bash
ros2 launch trajectory_pure_pursuit trajectory_pure_pursuit.launch.xml
```

Default topics:

- Input: `/autonomous/trajectory`
- Output: `/autonomous/cmd_drive`

Steering uses the standard Pure Pursuit curvature:

```text
curvature = 2 * y / (x^2 + y^2)
steer = atan(wheelbase * curvature)
```

Speed is reduced from `max_speed` based on curvature and steering angle.

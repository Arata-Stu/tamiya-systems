import jax.numpy as jnp


class LocalTrajectoryPurePursuit:
    """Convert normalized Bezier trajectory actions into steer/speed commands."""

    def __init__(
        self,
        num_points=20,
        x_anchors=(0.4, 1.0, 1.8),
        x_offset_scale=0.25,
        y_scale=0.8,
        min_control_dx=0.05,
        wheelbase=0.3302,
        lookahead_base=0.35,
        lookahead_gain=0.35,
        lookahead_min=0.35,
        lookahead_max=1.2,
        min_forward_distance=0.05,
        steering_limit=0.4,
        min_speed=0.0,
        max_speed=5.0,
        curvature_speed_gain=1.5,
        steering_speed_gain=1.0,
        short_path_speed_scale=0.6,
    ):
        self.num_points = int(num_points)
        self.x_anchors = jnp.asarray(x_anchors, dtype=jnp.float32)
        if self.x_anchors.shape != (3,):
            raise ValueError(f"x_anchors must have shape (3,), got {self.x_anchors.shape}")
        self.x_offset_scale = float(x_offset_scale)
        self.y_scale = float(y_scale)
        self.min_control_dx = float(min_control_dx)
        self.wheelbase = float(wheelbase)
        self.lookahead_base = float(lookahead_base)
        self.lookahead_gain = float(lookahead_gain)
        self.lookahead_min = float(lookahead_min)
        self.lookahead_max = float(lookahead_max)
        self.min_forward_distance = float(min_forward_distance)
        self.steering_limit = float(steering_limit)
        self.min_speed = float(min_speed)
        self.max_speed = float(max_speed)
        self.curvature_speed_gain = float(curvature_speed_gain)
        self.steering_speed_gain = float(steering_speed_gain)
        self.short_path_speed_scale = float(short_path_speed_scale)
        self.t = jnp.linspace(1.0 / self.num_points, 1.0, self.num_points, dtype=jnp.float32)

    @property
    def action_dim(self):
        return 6

    def action_to_control_points(self, action_normalized):
        action = jnp.asarray(action_normalized, dtype=jnp.float32)
        if action.ndim != 2 or action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Local trajectory action must have shape (batch, {self.action_dim}), "
                f"got {action.shape}"
            )
        raw = action.reshape((action.shape[0], 3, 2))
        x = self.x_anchors[None, :] + raw[:, :, 0] * self.x_offset_scale
        x1 = jnp.maximum(x[:, 0], self.min_forward_distance)
        x2 = jnp.maximum(x[:, 1], x1 + self.min_control_dx)
        x3 = jnp.maximum(x[:, 2], x2 + self.min_control_dx)
        y = raw[:, :, 1] * self.y_scale
        return jnp.stack(
            [
                jnp.stack([x1, y[:, 0]], axis=1),
                jnp.stack([x2, y[:, 1]], axis=1),
                jnp.stack([x3, y[:, 2]], axis=1),
            ],
            axis=1,
        )

    def sample_bezier(self, control_points):
        p0 = jnp.zeros((control_points.shape[0], 1, 2), dtype=jnp.float32)
        p1 = control_points[:, 0:1, :]
        p2 = control_points[:, 1:2, :]
        p3 = control_points[:, 2:3, :]
        t = self.t[None, :, None]
        omt = 1.0 - t
        return (
            omt**3 * p0
            + 3.0 * omt**2 * t * p1
            + 3.0 * omt * t**2 * p2
            + t**3 * p3
        )

    def action_to_trajectory(self, action_normalized):
        return self.sample_bezier(self.action_to_control_points(action_normalized))

    def trajectory_to_control(self, trajectory, current_speed):
        traj = jnp.asarray(trajectory, dtype=jnp.float32)
        speed = jnp.maximum(jnp.asarray(current_speed, dtype=jnp.float32), 0.0)
        lookahead = jnp.clip(
            self.lookahead_base + self.lookahead_gain * speed,
            self.lookahead_min,
            self.lookahead_max,
        )

        origin = jnp.zeros((traj.shape[0], 1, 2), dtype=jnp.float32)
        prev = jnp.concatenate([origin, traj[:, :-1, :]], axis=1)
        segment_lengths = jnp.linalg.norm(traj - prev, axis=-1)
        cumulative = jnp.cumsum(segment_lengths, axis=1)
        euclidean = jnp.linalg.norm(traj, axis=-1)
        valid = traj[:, :, 0] >= self.min_forward_distance
        hit = valid & ((cumulative >= lookahead[:, None]) | (euclidean >= lookahead[:, None]))
        first_hit = jnp.argmax(hit.astype(jnp.int32), axis=1)
        valid_count = jnp.sum(valid.astype(jnp.int32), axis=1)
        fallback = jnp.maximum(valid_count - 1, 0)
        target_idx = jnp.where(jnp.any(hit, axis=1), first_hit, fallback)
        batch_idx = jnp.arange(traj.shape[0])
        target = traj[batch_idx, target_idx, :]

        x = target[:, 0]
        y = target[:, 1]
        distance_sq = jnp.maximum(1.0e-6, x * x + y * y)
        curvature = 2.0 * y / distance_sq
        steer = jnp.clip(jnp.arctan(self.wheelbase * curvature), -self.steering_limit, self.steering_limit)

        abs_curvature = jnp.abs(curvature)
        speed_cmd = self.max_speed / (1.0 + self.curvature_speed_gain * abs_curvature)
        steer_ratio = jnp.minimum(1.0, jnp.abs(steer) / jnp.maximum(1.0e-6, self.steering_limit))
        speed_cmd = speed_cmd / (1.0 + self.steering_speed_gain * steer_ratio)
        speed_cmd = jnp.where(
            target_idx + 1 >= traj.shape[1],
            speed_cmd * self.short_path_speed_scale,
            speed_cmd,
        )
        speed_cmd = jnp.clip(speed_cmd, self.min_speed, self.max_speed)
        return jnp.stack([steer, speed_cmd], axis=1)

    def act(self, action_normalized, current_speed):
        trajectory = self.action_to_trajectory(action_normalized)
        return self.trajectory_to_control(trajectory, current_speed)

    def smoothness_penalty(self, action_normalized):
        traj = self.action_to_trajectory(action_normalized)
        curvature = traj[:, 2:, :] - 2.0 * traj[:, 1:-1, :] + traj[:, :-2, :]
        return jnp.mean(jnp.sum(curvature**2, axis=-1), axis=1)

    def tail_smoothness_penalty(self, action_normalized, power=2.0):
        traj = self.action_to_trajectory(action_normalized)
        curvature = traj[:, 2:, :] - 2.0 * traj[:, 1:-1, :] + traj[:, :-2, :]
        weights = jnp.linspace(0.0, 1.0, curvature.shape[1], dtype=jnp.float32) ** float(power)
        weights = weights / jnp.maximum(jnp.mean(weights), 1.0e-6)
        return jnp.mean(jnp.sum(curvature**2, axis=-1) * weights[None, :], axis=1)

    def lateral_penalty(self, action_normalized):
        traj = self.action_to_trajectory(action_normalized)
        return jnp.mean(traj[:, :, 1] ** 2, axis=1)

    def terminal_lateral_penalty(self, action_normalized):
        traj = self.action_to_trajectory(action_normalized)
        return traj[:, -1, 1] ** 2

    def terminal_heading_penalty(self, action_normalized):
        traj = self.action_to_trajectory(action_normalized)
        terminal_delta = traj[:, -1, :] - traj[:, -2, :]
        heading = jnp.arctan2(terminal_delta[:, 1], jnp.maximum(terminal_delta[:, 0], 1.0e-6))
        return heading**2

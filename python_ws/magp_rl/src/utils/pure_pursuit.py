import jax
import jax.numpy as jnp


class PurePursuitTeacher:
    """JAX-native pure pursuit teacher for TAL-style reward shaping."""

    def __init__(
        self,
        waypoints_xy,
        waypoints_s,
        waypoints_speed,
        lookahead_distance,
        lookahead_gain,
        wheelbase,
        vgain,
    ):
        self.waypoints_xy = jnp.asarray(waypoints_xy, dtype=jnp.float32)
        self.waypoints_s = jnp.asarray(waypoints_s, dtype=jnp.float32)
        self.waypoints_speed = jnp.asarray(waypoints_speed, dtype=jnp.float32)
        self.lookahead_distance = float(lookahead_distance)
        self.lookahead_gain = float(lookahead_gain)
        self.wheelbase = float(wheelbase)
        self.vgain = float(vgain)
        self.track_length = float(self.waypoints_s[-1])

        if self.track_length <= 0.0:
            raise ValueError("Invalid track length for PurePursuitTeacher")

        self._act_jit = jax.jit(self._act_impl)

    def _act_impl(self, poses_x, poses_y, poses_theta, current_speed):
        points = jnp.stack([poses_x, poses_y], axis=1)
        speed_nonneg = jnp.maximum(current_speed, 0.0)
        lookahead = jnp.maximum(
            self.lookahead_distance + self.lookahead_gain * speed_nonneg,
            1e-3,
        )

        # Nearest waypoint index by Euclidean distance.
        d2 = jnp.sum((points[:, None, :] - self.waypoints_xy[None, :, :]) ** 2, axis=-1)
        nearest_idx = jnp.argmin(d2, axis=1)
        nearest_s = self.waypoints_s[nearest_idx]

        target_s = jnp.mod(nearest_s + lookahead, self.track_length)
        target_idx = jnp.searchsorted(self.waypoints_s, target_s, side="left")
        target_idx = jnp.where(target_idx >= self.waypoints_s.shape[0], 0, target_idx)

        target_xy = self.waypoints_xy[target_idx]
        target_speed = self.waypoints_speed[target_idx] * self.vgain

        dx = target_xy[:, 0] - poses_x
        dy = target_xy[:, 1] - poses_y
        waypoint_y = jnp.sin(-poses_theta) * dx + jnp.cos(-poses_theta) * dy

        eps = 1e-6
        radius = jnp.where(
            jnp.abs(waypoint_y) < eps,
            1e6,
            (lookahead**2) / (2.0 * waypoint_y),
        )
        steer = jnp.where(
            jnp.abs(waypoint_y) < eps,
            0.0,
            jnp.arctan(self.wheelbase / radius),
        )

        return jnp.stack([steer, target_speed], axis=1)

    def act(self, poses_x, poses_y, poses_theta, current_speed=None):
        if current_speed is None:
            current_speed = jnp.zeros_like(poses_x, dtype=jnp.float32)
        return self._act_jit(poses_x, poses_y, poses_theta, current_speed)

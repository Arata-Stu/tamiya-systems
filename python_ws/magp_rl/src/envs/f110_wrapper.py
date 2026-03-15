import jax.numpy as jnp

from src.utils.track import (
    compute_progress_delta,
    load_waypoints,
    project_to_centerline_s,
)


class F110EnvWrapper:
    """Single simulator wrapper (race mode: multi-agent in one world)."""

    def __init__(self, simulator, config, waypoints_path=None):
        self.sim = simulator
        self.num_envs = int(simulator.num_agents)
        self.max_steer = config.max_steer
        self.min_steer = config.min_steer
        self.max_speed = config.max_speed
        self.min_speed = config.min_speed
        self.max_lidar_range = config.max_lidar_range

        self.action_scale = jnp.array(
            [
                (self.max_steer - self.min_steer) / 2.0,
                (self.max_speed - self.min_speed) / 2.0,
            ]
        )
        self.action_bias = jnp.array(
            [
                (self.max_steer + self.min_steer) / 2.0,
                (self.max_speed + self.min_speed) / 2.0,
            ]
        )

        self.base_reward_scale = float(config.reward.base_reward_scale)
        self.progress_coef = float(config.reward.progress_coef)
        self.speed_coef = float(config.reward.speed_coef)
        self.collision_penalty = float(config.reward.collision_penalty)
        self.progress_clip = float(config.reward.progress_clip)

        waypoint_source = waypoints_path if waypoints_path is not None else config.waypoints_path
        waypoints_xy, waypoints_s = load_waypoints(waypoint_source)
        self.waypoints_xy = jnp.asarray(waypoints_xy, dtype=jnp.float32)
        self.waypoints_s = jnp.asarray(waypoints_s, dtype=jnp.float32)
        self.track_length = float(waypoints_s[-1])
        if self.track_length <= 0.0:
            raise ValueError("Invalid waypoint track length.")

        self.prev_s = None
        self.last_obs = None

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"]
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def _project_to_centerline_s(self, poses_x, poses_y):
        points = jnp.stack([poses_x, poses_y], axis=1)
        return project_to_centerline_s(points, self.waypoints_xy, self.waypoints_s, self.track_length)

    def compute_progress_delta(self, current_s, prev_s):
        return compute_progress_delta(current_s, prev_s, self.track_length, self.progress_clip)

    def get_current_progress_s(self):
        state = self.sim.sim_state["state"]
        return self._project_to_centerline_s(state[:, 0], state[:, 1])

    def get_positions(self):
        state = self.sim.sim_state["state"]
        return state[:, 0], state[:, 1]

    def get_speeds(self):
        return jnp.abs(self.sim.sim_state["state"][:, 3])

    def get_collisions(self):
        return self.sim.sim_state["collisions"]

    def reset(self, poses):
        obs_dict, _, _, _ = self.sim.reset(poses)
        self.prev_s = self._project_to_centerline_s(obs_dict["poses_x"], obs_dict["poses_y"])
        self.last_obs = self._normalize_obs(obs_dict)
        return self.last_obs

    def reset_done(self, done_mask, poses):
        # race modeでは独立部分リセット不可のため、doneが1つでも立ったら全体をリセット。
        if bool(jnp.any(done_mask)):
            obs = self.reset(poses)
            return obs, self.prev_s
        return self.last_obs, self.prev_s

    def step(self, action_normalized):
        action_physical = self._scale_action(action_normalized)
        next_obs_dict, reward, done, info = self.sim.step(action_physical)

        next_obs = self._normalize_obs(next_obs_dict)
        current_s = self._project_to_centerline_s(next_obs_dict["poses_x"], next_obs_dict["poses_y"])
        progress = self.compute_progress_delta(current_s, self.prev_s)
        self.prev_s = current_s

        collision = next_obs_dict["collisions"]
        shaped_reward = (
            self.base_reward_scale * reward
            + self.progress_coef * progress
            + self.speed_coef * action_physical[:, 1]
            - self.collision_penalty * collision
        )
        self.last_obs = next_obs

        done_agents = jnp.full((action_normalized.shape[0],), done, dtype=jnp.float32)
        info_out = dict(info)
        if "checkpoint_done" in info_out:
            info_out["checkpoint_done"] = jnp.asarray(info_out["checkpoint_done"], dtype=jnp.float32)
        return next_obs, shaped_reward, done_agents, info_out

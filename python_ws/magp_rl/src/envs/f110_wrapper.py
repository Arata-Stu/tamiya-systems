import jax.numpy as jnp

from src.utils.pure_pursuit import PurePursuitTeacher
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

        tal_cfg = config.reward.get("tal", None)
        tal_default_speed = 2.0
        tal_speed_mode = "file"
        tal_speed_min = float(config.min_speed)
        tal_speed_max = float(config.max_speed)
        tal_speed_lat_accel = 6.0
        tal_speed_smoothing = 9
        if tal_cfg is not None:
            tal_default_speed = float(tal_cfg.get("default_speed_mps", tal_default_speed))
            speed_profile_cfg = tal_cfg.get("speed_profile", {})
            tal_speed_mode = str(speed_profile_cfg.get("mode", tal_speed_mode))
            min_speed_cfg = speed_profile_cfg.get("min_speed_mps", None)
            max_speed_cfg = speed_profile_cfg.get("max_speed_mps", None)
            tal_speed_min = float(config.min_speed if min_speed_cfg is None else min_speed_cfg)
            tal_speed_max = float(config.max_speed if max_speed_cfg is None else max_speed_cfg)
            tal_speed_lat_accel = float(
                speed_profile_cfg.get("max_lateral_accel_mps2", tal_speed_lat_accel)
            )
            tal_speed_smoothing = int(speed_profile_cfg.get("smoothing_window", tal_speed_smoothing))

        waypoint_source = waypoints_path if waypoints_path is not None else config.waypoints_path
        waypoints_xy, waypoints_s, waypoints_speed = load_waypoints(
            waypoint_source,
            default_speed_mps=tal_default_speed,
            speed_mode=tal_speed_mode,
            min_speed_mps=tal_speed_min,
            max_speed_mps=tal_speed_max,
            max_lateral_accel_mps2=tal_speed_lat_accel,
            smoothing_window=tal_speed_smoothing,
        )
        self.waypoints_xy = jnp.asarray(waypoints_xy, dtype=jnp.float32)
        self.waypoints_s = jnp.asarray(waypoints_s, dtype=jnp.float32)
        self.waypoints_speed = jnp.asarray(waypoints_speed, dtype=jnp.float32)
        self.track_length = float(waypoints_s[-1])
        if self.track_length <= 0.0:
            raise ValueError("Invalid waypoint track length.")

        self.prev_s = None
        self.last_obs = None
        self._setup_tal(tal_cfg)

    def _setup_tal(self, tal_cfg):
        self.tal_enabled = False
        self.tal_coef = 0.0
        self.tal_steer_weight = 1.0
        self.tal_speed_weight = 1.0
        self.pp_teacher = None

        if tal_cfg is None:
            return
        if not bool(tal_cfg.get("enabled", False)):
            return

        self.tal_enabled = True
        self.tal_coef = float(tal_cfg.get("coef", 0.0))
        self.tal_steer_weight = float(tal_cfg.get("steer_weight", 1.0))
        self.tal_speed_weight = float(tal_cfg.get("speed_weight", 1.0))

        lookahead = float(tal_cfg.get("lookahead_distance", 0.5))
        lookahead_gain = float(tal_cfg.get("lookahead_gain", 0.3))
        wheelbase = float(tal_cfg.get("wheelbase", 0.17145 + 0.15875))
        vgain = float(tal_cfg.get("vgain", 1.0))
        self.pp_teacher = PurePursuitTeacher(
            self.waypoints_xy,
            self.waypoints_s,
            self.waypoints_speed,
            lookahead_distance=lookahead,
            lookahead_gain=lookahead_gain,
            wheelbase=wheelbase,
            vgain=vgain,
        )

    def set_tal_coef(self, coef):
        if not self.tal_enabled:
            return
        self.tal_coef = float(max(coef, 0.0))

    def get_tal_coef(self):
        return float(self.tal_coef)

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"]
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def _to_normalized_action(self, action_physical):
        action_normalized = (action_physical - self.action_bias) / self.action_scale
        return jnp.clip(action_normalized, -1.0, 1.0)

    def _compute_tal_reward(self, action_normalized):
        if not self.tal_enabled:
            return jnp.zeros((action_normalized.shape[0],), dtype=jnp.float32)

        state = self.sim.sim_state["state"]
        teacher_action = self.pp_teacher.act(state[:, 0], state[:, 1], state[:, 4], state[:, 3])
        teacher_action = teacher_action.at[:, 0].set(
            jnp.clip(teacher_action[:, 0], self.min_steer, self.max_steer)
        )
        teacher_action = teacher_action.at[:, 1].set(
            jnp.clip(teacher_action[:, 1], self.min_speed, self.max_speed)
        )
        teacher_norm = self._to_normalized_action(teacher_action)
        diff = action_normalized - teacher_norm
        imitation_error = (
            self.tal_steer_weight * (diff[:, 0] ** 2)
            + self.tal_speed_weight * (diff[:, 1] ** 2)
        )
        return -self.tal_coef * imitation_error

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
        tal_reward = self._compute_tal_reward(action_normalized)
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
            + tal_reward
        )
        self.last_obs = next_obs

        done_agents = jnp.full((action_normalized.shape[0],), done, dtype=jnp.float32)
        info_out = dict(info)
        if "checkpoint_done" in info_out:
            info_out["checkpoint_done"] = jnp.asarray(info_out["checkpoint_done"], dtype=jnp.float32)
        return next_obs, shaped_reward, done_agents, info_out

import jax
import jax.numpy as jnp

from src.utils.pure_pursuit import PurePursuitTeacher
from src.utils.local_trajectory import LocalTrajectoryPurePursuit
from src.utils.track import (
    compute_progress_delta,
    load_waypoints,
    project_to_centerline_s,
)


class F110IndependentVecEnv:
    """Independent vectorized environments on one GPU.

    Each environment contains a single car (`num_agents=1` in simulator) and is
    batched with `jax.vmap` for parallel stepping.
    """

    def __init__(self, simulator, config, num_envs, waypoints_path=None, seed=0):
        if int(simulator.num_agents) != 1:
            raise ValueError("F110IndependentVecEnv requires simulator.num_agents == 1")

        self.sim = simulator
        self.num_envs = int(num_envs)
        self.max_steer = config.max_steer
        self.min_steer = config.min_steer
        self.max_speed = config.max_speed
        self.min_speed = config.min_speed
        self.max_lidar_range = config.max_lidar_range
        self.control_interface = str(config.get("control_interface", "direct")).lower()
        self.policy_action_dim = int(config.get("action_dim", 2))

        self.action_scale = jnp.array(
            [
                (self.max_steer - self.min_steer) / 2.0,
                (self.max_speed - self.min_speed) / 2.0,
            ],
            dtype=jnp.float32,
        )
        self.action_bias = jnp.array(
            [
                (self.max_steer + self.min_steer) / 2.0,
                (self.max_speed + self.min_speed) / 2.0,
            ],
            dtype=jnp.float32,
        )

        self.base_reward_scale = float(config.reward.base_reward_scale)
        self.progress_coef = float(config.reward.progress_coef)
        self.speed_coef = float(config.reward.speed_coef)
        self.collision_penalty = float(config.reward.collision_penalty)
        self.progress_clip = float(config.reward.progress_clip)
        self._setup_local_trajectory_controller(config)

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

        self.master_rng = jax.random.PRNGKey(int(seed))
        self.sim_state = None
        self.prev_s = None
        self.last_obs = None
        self._setup_tal(tal_cfg)

        self._v_init_state = jax.jit(jax.vmap(self._init_single_state, in_axes=(0, 0)))
        self._v_step = jax.jit(jax.vmap(self._step_single_env, in_axes=(0, 0)))

    def _setup_tal(self, tal_cfg):
        self.tal_enabled = False
        self.tal_coef = 0.0
        self.tal_steer_weight = 1.0
        self.tal_speed_weight = 1.0
        self.pp_teacher = None

        if tal_cfg is None:
            return
        if self.control_interface != "direct":
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

    def _setup_local_trajectory_controller(self, config):
        self.local_trajectory_controller = None
        self.trajectory_smoothness_coef = 0.0
        self.trajectory_lateral_coef = 0.0
        if self.control_interface == "direct":
            if self.policy_action_dim != 2:
                raise ValueError(
                    f"env.action_dim must be 2 when env.control_interface=direct, "
                    f"got {self.policy_action_dim}"
                )
            return
        if self.control_interface != "local_trajectory_pp":
            raise ValueError(
                f"Unsupported env.control_interface: {self.control_interface}. "
                "Use direct | local_trajectory_pp."
            )

        traj_cfg = config.get("trajectory", {})
        traj_max_speed = traj_cfg.get("max_speed", self.max_speed)
        if traj_max_speed is None:
            traj_max_speed = self.max_speed

        self.local_trajectory_controller = LocalTrajectoryPurePursuit(
            num_points=int(traj_cfg.get("num_points", 20)),
            x_anchors=tuple(traj_cfg.get("x_anchors", [0.4, 1.0, 1.8])),
            x_offset_scale=float(traj_cfg.get("x_offset_scale", 0.25)),
            y_scale=float(traj_cfg.get("y_scale", 0.8)),
            min_control_dx=float(traj_cfg.get("min_control_dx", 0.05)),
            wheelbase=float(traj_cfg.get("wheelbase", 0.3302)),
            lookahead_base=float(traj_cfg.get("lookahead_base", 0.35)),
            lookahead_gain=float(traj_cfg.get("lookahead_gain", 0.35)),
            lookahead_min=float(traj_cfg.get("lookahead_min", 0.35)),
            lookahead_max=float(traj_cfg.get("lookahead_max", 1.2)),
            min_forward_distance=float(traj_cfg.get("min_forward_distance", 0.05)),
            steering_limit=float(traj_cfg.get("steering_limit", self.max_steer)),
            min_speed=float(traj_cfg.get("min_speed", self.min_speed)),
            max_speed=float(traj_max_speed),
            curvature_speed_gain=float(traj_cfg.get("curvature_speed_gain", 1.5)),
            steering_speed_gain=float(traj_cfg.get("steering_speed_gain", 1.0)),
            short_path_speed_scale=float(traj_cfg.get("short_path_speed_scale", 0.6)),
        )
        if self.policy_action_dim != self.local_trajectory_controller.action_dim:
            raise ValueError(
                "env.action_dim must match the local trajectory action size "
                f"({self.local_trajectory_controller.action_dim}) when "
                f"env.control_interface=local_trajectory_pp, got {self.policy_action_dim}"
            )
        reward_cfg = traj_cfg.get("reward", {})
        self.trajectory_smoothness_coef = float(reward_cfg.get("smoothness_coef", 0.0))
        self.trajectory_lateral_coef = float(reward_cfg.get("lateral_coef", 0.0))

    def set_tal_coef(self, coef):
        if not self.tal_enabled:
            return
        self.tal_coef = float(max(coef, 0.0))

    def get_tal_coef(self):
        return float(self.tal_coef)

    def _next_rng_keys(self, n):
        self.master_rng, sub = jax.random.split(self.master_rng)
        return jax.random.split(sub, n)

    def _init_single_state(self, pose, rng_key):
        pose = jnp.asarray(pose, dtype=jnp.float32)

        state = jnp.zeros((1, 7), dtype=jnp.float32)
        state = state.at[0, 0].set(pose[0])
        state = state.at[0, 1].set(pose[1])
        state = state.at[0, 4].set(pose[2])

        theta = pose[2]
        cos_th = jnp.cos(-theta)
        sin_th = jnp.sin(-theta)
        start_rot = jnp.array([[cos_th, -sin_th], [sin_th, cos_th]], dtype=jnp.float32)

        return {
            "state": state,
            "collisions": jnp.zeros((1,), dtype=jnp.float32),
            "collision_idx": -1 * jnp.ones((1,), dtype=jnp.float32),
            "steer_buffers": jnp.zeros((1, self.sim.steer_buffer_size), dtype=jnp.float32),
            "rng_key": rng_key,
            "current_time": jnp.float32(0.0),
            "lap_times": jnp.zeros((1,), dtype=jnp.float32),
            "lap_counts": jnp.zeros((1,), dtype=jnp.float32),
            "near_starts": jnp.ones((1,), dtype=jnp.bool_),
            "toggle_list": jnp.zeros((1,), dtype=jnp.float32),
            "start_xs": pose[0:1],
            "start_ys": pose[1:2],
            "start_rot": start_rot,
        }

    def _step_single_env(self, sim_state, action):
        action_2d = jnp.asarray(action, dtype=jnp.float32)[None, :]
        return self.sim._compiled_step(sim_state, action_2d)

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"][:, 0, :]
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def _action_to_physical(self, action_normalized):
        if self.control_interface == "direct":
            return self._scale_action(action_normalized)
        state = self.sim_state["state"][:, 0, :]
        current_speed = jnp.abs(state[:, 3])
        action_physical = self.local_trajectory_controller.act(action_normalized, current_speed)
        steer = jnp.clip(action_physical[:, 0], self.min_steer, self.max_steer)
        speed = jnp.clip(action_physical[:, 1], self.min_speed, self.max_speed)
        return jnp.stack([steer, speed], axis=1)

    def _compute_trajectory_reward(self, action_normalized):
        if self.local_trajectory_controller is None:
            return jnp.zeros((action_normalized.shape[0],), dtype=jnp.float32)
        reward = jnp.zeros((action_normalized.shape[0],), dtype=jnp.float32)
        if self.trajectory_smoothness_coef > 0.0:
            reward = (
                reward
                - self.trajectory_smoothness_coef
                * self.local_trajectory_controller.smoothness_penalty(action_normalized)
            )
        if self.trajectory_lateral_coef > 0.0:
            reward = (
                reward
                - self.trajectory_lateral_coef
                * self.local_trajectory_controller.lateral_penalty(action_normalized)
            )
        return reward

    def _to_normalized_action(self, action_physical):
        action_normalized = (action_physical - self.action_bias) / self.action_scale
        return jnp.clip(action_normalized, -1.0, 1.0)

    def _compute_tal_reward(self, action_normalized):
        if not self.tal_enabled:
            return jnp.zeros((action_normalized.shape[0],), dtype=jnp.float32)

        state = self.sim_state["state"][:, 0, :]
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
        state = self.sim_state["state"][:, 0, :]
        return self._project_to_centerline_s(state[:, 0], state[:, 1])

    def get_positions(self):
        state = self.sim_state["state"][:, 0, :]
        return state[:, 0], state[:, 1]

    def get_speeds(self):
        return jnp.abs(self.sim_state["state"][:, 0, 3])

    def get_collisions(self):
        return self.sim_state["collisions"][:, 0]

    def _merge_by_mask(self, original, replacement, done_mask):
        def merge_leaf(old, new):
            expand_shape = (done_mask.shape[0],) + (1,) * (old.ndim - 1)
            mask = done_mask.reshape(expand_shape)
            return jnp.where(mask, new, old)

        return jax.tree_util.tree_map(merge_leaf, original, replacement)

    def reset(self, poses):
        poses = jnp.asarray(poses, dtype=jnp.float32)
        if poses.shape != (self.num_envs, 3):
            raise ValueError(f"Invalid poses shape: expected {(self.num_envs, 3)}, got {poses.shape}")

        keys = self._next_rng_keys(self.num_envs)
        reset_states = self._v_init_state(poses, keys)
        zero_actions = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        self.sim_state, obs_dict, _, _, _ = self._v_step(reset_states, zero_actions)

        poses_x = obs_dict["poses_x"][:, 0]
        poses_y = obs_dict["poses_y"][:, 0]
        self.prev_s = self._project_to_centerline_s(poses_x, poses_y)
        self.last_obs = self._normalize_obs(obs_dict)
        return self.last_obs

    def reset_done(self, done_mask, poses):
        done_mask = jnp.asarray(done_mask, dtype=jnp.bool_)
        if not bool(jax.device_get(jnp.any(done_mask))):
            return self.last_obs, self.prev_s

        poses = jnp.asarray(poses, dtype=jnp.float32)
        if poses.shape != (self.num_envs, 3):
            raise ValueError(f"Invalid poses shape: expected {(self.num_envs, 3)}, got {poses.shape}")

        keys = self._next_rng_keys(self.num_envs)
        reset_states = self._v_init_state(poses, keys)
        zero_actions = jnp.zeros((self.num_envs, 2), dtype=jnp.float32)
        reset_states, reset_obs_dict, _, _, _ = self._v_step(reset_states, zero_actions)

        self.sim_state = self._merge_by_mask(self.sim_state, reset_states, done_mask)

        reset_obs = self._normalize_obs(reset_obs_dict)
        reset_prev_s = self._project_to_centerline_s(
            reset_obs_dict["poses_x"][:, 0],
            reset_obs_dict["poses_y"][:, 0],
        )

        self.last_obs = jnp.where(done_mask[:, None], reset_obs, self.last_obs)
        self.prev_s = jnp.where(done_mask, reset_prev_s, self.prev_s)
        return self.last_obs, self.prev_s

    def step(self, action_normalized):
        tal_reward = self._compute_tal_reward(action_normalized)
        trajectory_reward = self._compute_trajectory_reward(action_normalized)
        action_physical = self._action_to_physical(action_normalized)
        self.sim_state, next_obs_dict, reward, done, info = self._v_step(self.sim_state, action_physical)

        next_obs = self._normalize_obs(next_obs_dict)
        poses_x = next_obs_dict["poses_x"][:, 0]
        poses_y = next_obs_dict["poses_y"][:, 0]
        current_s = self._project_to_centerline_s(poses_x, poses_y)

        progress = self.compute_progress_delta(current_s, self.prev_s)
        self.prev_s = current_s

        collision = next_obs_dict["collisions"][:, 0]
        shaped_reward = (
            self.base_reward_scale * reward
            + self.progress_coef * progress
            + self.speed_coef * action_physical[:, 1]
            - self.collision_penalty * collision
            + tal_reward
            + trajectory_reward
        )

        done_agents = done.astype(jnp.float32)
        info_out = {
            "checkpoint_done": jnp.asarray(info["checkpoint_done"][:, 0], dtype=jnp.float32)
        }
        self.last_obs = next_obs
        return next_obs, shaped_reward, done_agents, info_out

import jax
import jax.numpy as jnp

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

        waypoint_source = waypoints_path if waypoints_path is not None else config.waypoints_path
        waypoints_xy, waypoints_s = load_waypoints(waypoint_source)
        self.waypoints_xy = jnp.asarray(waypoints_xy, dtype=jnp.float32)
        self.waypoints_s = jnp.asarray(waypoints_s, dtype=jnp.float32)
        self.track_length = float(waypoints_s[-1])
        if self.track_length <= 0.0:
            raise ValueError("Invalid waypoint track length.")

        self.master_rng = jax.random.PRNGKey(int(seed))
        self.sim_state = None
        self.prev_s = None
        self.last_obs = None

        self._v_init_state = jax.jit(jax.vmap(self._init_single_state, in_axes=(0, 0)))
        self._v_step = jax.jit(jax.vmap(self._step_single_env, in_axes=(0, 0)))

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
        action_physical = self._scale_action(action_normalized)
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
        )

        done_agents = done.astype(jnp.float32)
        info_out = {
            "checkpoint_done": jnp.asarray(info["checkpoint_done"][:, 0], dtype=jnp.float32)
        }
        self.last_obs = next_obs
        return next_obs, shaped_reward, done_agents, info_out

import jax.numpy as jnp

from src.utils.pure_pursuit import PurePursuitTeacher
from src.utils.track import (
    compute_progress_delta,
    load_waypoints,
    project_to_centerline_s,
)


class F110EnvWrapper:
    """Single simulator wrapper (race mode: multi-agent in one world)."""

    def __init__(
        self,
        simulator,
        config,
        waypoints_path=None,
        control_mode="selfplay",
        npc_controller="pure_pursuit",
        ego_idx=0,
    ):
        self.sim = simulator
        self.num_agents = int(simulator.num_agents)
        self.control_mode = str(control_mode).lower()
        self.npc_controller_name = str(npc_controller).lower()
        self.ego_idx = int(ego_idx)
        if self.control_mode not in {"selfplay", "npc"}:
            raise ValueError(
                f"Unsupported race control mode: {self.control_mode}. "
                "Use selfplay | npc."
            )
        if not (0 <= self.ego_idx < self.num_agents):
            raise ValueError(
                f"Invalid ego_idx={self.ego_idx}. It must be in [0, {self.num_agents - 1}]"
            )
        self.sim.ego_idx = self.ego_idx
        self.num_envs = self.num_agents if self.control_mode == "selfplay" else 1
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
        race_cfg = config.get("race", {})
        npc_cfg = race_cfg.get("npc", {})
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
        self._setup_npc_controller(tal_cfg, npc_cfg)

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

    def _setup_npc_controller(self, tal_cfg, npc_cfg):
        self.npc_teacher = None
        self.npc_speed_scale = 1.0
        if self.control_mode != "npc":
            return
        if self.npc_controller_name != "pure_pursuit":
            raise ValueError(
                f"Unsupported npc controller: {self.npc_controller_name}. "
                "Only pure_pursuit is supported."
            )

        lookahead = 0.5
        lookahead_gain = 0.3
        wheelbase = 0.17145 + 0.15875
        vgain = 1.0
        if tal_cfg is not None:
            lookahead = float(tal_cfg.get("lookahead_distance", lookahead))
            lookahead_gain = float(tal_cfg.get("lookahead_gain", lookahead_gain))
            wheelbase = float(tal_cfg.get("wheelbase", wheelbase))
            vgain = float(tal_cfg.get("vgain", vgain))
        if npc_cfg is not None:
            lookahead = float(npc_cfg.get("lookahead_distance", lookahead))
            lookahead_gain = float(npc_cfg.get("lookahead_gain", lookahead_gain))
            wheelbase = float(npc_cfg.get("wheelbase", wheelbase))
            vgain = float(npc_cfg.get("vgain", vgain))
            self.npc_speed_scale = float(npc_cfg.get("speed_scale", self.npc_speed_scale))
        if self.npc_speed_scale <= 0.0:
            raise ValueError(
                f"env.race.npc.speed_scale must be > 0, got {self.npc_speed_scale}"
            )

        self.npc_teacher = PurePursuitTeacher(
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

    def _select_controlled(self, values):
        if self.control_mode == "npc":
            return values[self.ego_idx : self.ego_idx + 1]
        return values

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"]
        obs = jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range
        return self._select_controlled(obs)

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
        teacher_norm = self._select_controlled(teacher_norm)
        diff = action_normalized - teacher_norm
        imitation_error = (
            self.tal_steer_weight * (diff[:, 0] ** 2)
            + self.tal_speed_weight * (diff[:, 1] ** 2)
        )
        return -self.tal_coef * imitation_error

    def _compute_npc_action_physical(self):
        if self.npc_teacher is None:
            raise RuntimeError("NPC controller is not initialized.")
        state = self.sim.sim_state["state"]
        npc_action = self.npc_teacher.act(state[:, 0], state[:, 1], state[:, 4], state[:, 3])
        npc_action = npc_action.at[:, 0].set(
            jnp.clip(npc_action[:, 0], self.min_steer, self.max_steer)
        )
        npc_action = npc_action.at[:, 1].set(
            jnp.clip(npc_action[:, 1], self.min_speed, self.max_speed)
        )
        if self.npc_speed_scale != 1.0:
            scales = jnp.full((self.num_agents,), self.npc_speed_scale, dtype=jnp.float32)
            scales = scales.at[self.ego_idx].set(1.0)
            npc_action = npc_action.at[:, 1].set(
                jnp.clip(npc_action[:, 1] * scales, self.min_speed, self.max_speed)
            )
        return npc_action

    def _compose_action_physical(self, action_normalized):
        action_normalized = jnp.asarray(action_normalized, dtype=jnp.float32)
        if self.control_mode == "selfplay":
            expected_shape = (self.num_agents, 2)
            if action_normalized.shape != expected_shape:
                raise ValueError(
                    f"Invalid action shape for selfplay: expected {expected_shape}, "
                    f"got {action_normalized.shape}"
                )
            return self._scale_action(action_normalized)

        expected_shape = (1, 2)
        if action_normalized.shape != expected_shape:
            raise ValueError(
                f"Invalid action shape for npc mode: expected {expected_shape}, "
                f"got {action_normalized.shape}"
            )
        action_all = self._compute_npc_action_physical()
        ego_action = self._scale_action(action_normalized)[0]
        action_all = action_all.at[self.ego_idx].set(ego_action)
        return action_all

    def _compute_done_any(self, done_sim, collisions, info):
        done_sim = jnp.asarray(done_sim, dtype=jnp.bool_)
        collision_any = jnp.any(jnp.asarray(collisions) > 0.5)
        checkpoint_done = info.get("checkpoint_done", None)
        checkpoint_any = (
            jnp.any(jnp.asarray(checkpoint_done) > 0.5)
            if checkpoint_done is not None
            else jnp.bool_(False)
        )
        return jnp.logical_or(done_sim, jnp.logical_or(collision_any, checkpoint_any))

    def _project_to_centerline_s(self, poses_x, poses_y):
        points = jnp.stack([poses_x, poses_y], axis=1)
        return project_to_centerline_s(points, self.waypoints_xy, self.waypoints_s, self.track_length)

    def compute_progress_delta(self, current_s, prev_s):
        return compute_progress_delta(current_s, prev_s, self.track_length, self.progress_clip)

    def get_current_progress_s(self):
        state = self.sim.sim_state["state"]
        s_all = self._project_to_centerline_s(state[:, 0], state[:, 1])
        return self._select_controlled(s_all)

    def get_positions(self):
        state = self.sim.sim_state["state"]
        poses_x = self._select_controlled(state[:, 0])
        poses_y = self._select_controlled(state[:, 1])
        return poses_x, poses_y

    def get_speeds(self):
        speed_all = jnp.abs(self.sim.sim_state["state"][:, 3])
        return self._select_controlled(speed_all)

    def get_collisions(self):
        return self.sim.sim_state["collisions"]

    def reset(self, poses):
        obs_dict, _, _, _ = self.sim.reset(poses)
        prev_s_all = self._project_to_centerline_s(obs_dict["poses_x"], obs_dict["poses_y"])
        self.prev_s = self._select_controlled(prev_s_all)
        self.last_obs = self._normalize_obs(obs_dict)
        return self.last_obs

    def reset_done(self, done_mask, poses):
        # race modeでは全体リセットのみをサポートする。
        if bool(jnp.any(done_mask)):
            obs = self.reset(poses)
            return obs, self.prev_s
        return self.last_obs, self.prev_s

    def step(self, action_normalized):
        tal_reward = self._compute_tal_reward(action_normalized)
        action_physical_all = self._compose_action_physical(action_normalized)
        next_obs_dict, reward_all, done, info = self.sim.step(action_physical_all)

        next_obs = self._normalize_obs(next_obs_dict)
        current_s_all = self._project_to_centerline_s(
            next_obs_dict["poses_x"],
            next_obs_dict["poses_y"],
        )
        current_s = self._select_controlled(current_s_all)
        progress = self.compute_progress_delta(current_s, self.prev_s)
        self.prev_s = current_s

        if self.control_mode == "npc":
            reward = reward_all[self.ego_idx : self.ego_idx + 1]
            speed_term = action_physical_all[self.ego_idx : self.ego_idx + 1, 1]
            collision = next_obs_dict["collisions"][self.ego_idx : self.ego_idx + 1]
        else:
            reward = reward_all
            speed_term = action_physical_all[:, 1]
            collision = next_obs_dict["collisions"]
        shaped_reward = (
            self.base_reward_scale * reward
            + self.progress_coef * progress
            + self.speed_coef * speed_term
            - self.collision_penalty * collision
            + tal_reward
        )
        self.last_obs = next_obs

        done_any = self._compute_done_any(done, next_obs_dict["collisions"], info)
        done_agents = jnp.full((self.num_envs,), done_any, dtype=jnp.float32)
        info_out = dict(info)
        if "checkpoint_done" in info_out:
            checkpoint_done = jnp.asarray(info_out["checkpoint_done"], dtype=jnp.float32)
            info_out["checkpoint_done"] = self._select_controlled(checkpoint_done)
        info_out["done_any"] = jnp.asarray(done_any, dtype=jnp.float32)
        return next_obs, shaped_reward, done_agents, info_out

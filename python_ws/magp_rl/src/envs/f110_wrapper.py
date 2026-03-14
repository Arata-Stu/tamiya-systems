from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


class F110EnvWrapper:
    def __init__(self, simulator, config, waypoints_path=None):
        self.sim = simulator
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
        waypoints_xy, waypoints_s = self._load_waypoints(waypoint_source)
        self.waypoints_xy = jnp.asarray(waypoints_xy, dtype=jnp.float32)
        self.waypoints_s = jnp.asarray(waypoints_s, dtype=jnp.float32)
        self.track_length = float(waypoints_s[-1])
        if self.track_length <= 0.0:
            raise ValueError("Invalid waypoint track length.")

        self.prev_s = None

    def _resolve_waypoints_path(self, path_str):
        path = Path(path_str)
        if path.is_absolute() and path.exists():
            return path

        cwd_path = (Path.cwd() / path).resolve()
        if cwd_path.exists():
            return cwd_path

        project_root = Path(__file__).resolve().parents[3]
        root_path = (project_root / path).resolve()
        if root_path.exists():
            return root_path

        raise FileNotFoundError(f"Waypoints file not found: {path_str}")

    def _load_waypoints(self, path_str):
        waypoint_path = self._resolve_waypoints_path(path_str)
        lines = waypoint_path.read_text(encoding="utf-8").splitlines()
        comment_lines = [ln.strip().lstrip("#").strip() for ln in lines if ln.strip().startswith("#")]
        header_line = next((ln for ln in comment_lines if "x_m" in ln and "y_m" in ln), "")

        first_data_line = next((ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")), "")
        delimiter = ";" if (";" in header_line or ";" in first_data_line) else ","

        data = np.genfromtxt(
            str(waypoint_path),
            delimiter=delimiter,
            comments="#",
            autostrip=True,
            dtype=np.float32,
        )

        if data.ndim == 1:
            data = data[None, :]
        data = data[~np.isnan(data).any(axis=1)]
        if data.shape[1] < 2:
            raise ValueError(f"Invalid waypoint format in: {waypoint_path}")

        columns = [c.strip() for c in header_line.split(delimiter)] if header_line else []
        has_s = "s_m" in columns

        if has_s:
            s_idx = columns.index("s_m")
            x_idx = columns.index("x_m")
            y_idx = columns.index("y_m")
            s = data[:, s_idx]
            xy = data[:, [x_idx, y_idx]]
        else:
            xy = data[:, :2]
            diffs = np.diff(xy, axis=0)
            ds = np.linalg.norm(diffs, axis=1)
            s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)

        return xy, s

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"]
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def _project_to_centerline_s(self, poses_x, poses_y):
        points = jnp.stack([poses_x, poses_y], axis=1)  # (A, 2)

        seg_a = self.waypoints_xy
        seg_b = jnp.roll(self.waypoints_xy, shift=-1, axis=0)
        seg_ab = seg_b - seg_a  # (W, 2)
        seg_len_sq = jnp.sum(seg_ab * seg_ab, axis=1) + 1e-8

        s_a = self.waypoints_s
        s_b = jnp.roll(self.waypoints_s, shift=-1, axis=0)
        seg_ds = jnp.where(s_b >= s_a, s_b - s_a, (s_b + self.track_length) - s_a)

        def project_point(p):
            ap = p[None, :] - seg_a
            t = jnp.clip(jnp.sum(ap * seg_ab, axis=1) / seg_len_sq, 0.0, 1.0)
            proj = seg_a + t[:, None] * seg_ab
            dist_sq = jnp.sum((p[None, :] - proj) ** 2, axis=1)
            idx = jnp.argmin(dist_sq)
            return s_a[idx] + t[idx] * seg_ds[idx]

        return jax.vmap(project_point)(points)

    def reset(self, poses):
        """シミュレータのAPIに合わせてposesを受け取りリセット"""
        obs_dict, _, _, _ = self.sim.reset(poses)
        self.prev_s = self._project_to_centerline_s(obs_dict["poses_x"], obs_dict["poses_y"])
        return self._normalize_obs(obs_dict)

    def step(self, action_normalized):
        """シミュレータを1ステップ進める"""
        action_physical = self._scale_action(action_normalized)
        next_obs_dict, reward, done, info = self.sim.step(action_physical)

        next_obs = self._normalize_obs(next_obs_dict)
        current_s = self._project_to_centerline_s(next_obs_dict["poses_x"], next_obs_dict["poses_y"])
        raw_progress = current_s - self.prev_s
        half_track = self.track_length * 0.5
        progress = jnp.where(raw_progress > half_track, raw_progress - self.track_length, raw_progress)
        progress = jnp.where(progress < -half_track, progress + self.track_length, progress)
        progress = jnp.clip(progress, -self.progress_clip, self.progress_clip)
        self.prev_s = current_s

        collision = next_obs_dict["collisions"]
        shaped_reward = (
            self.base_reward_scale * reward
            + self.progress_coef * progress
            + self.speed_coef * action_physical[:, 1]
            - self.collision_penalty * collision
        )

        # PPO/GAEが扱いやすいよう done を agent 次元に展開
        done_agents = jnp.full((action_normalized.shape[0],), done, dtype=jnp.float32)
        return next_obs, shaped_reward, done_agents, info

import jax.numpy as jnp


class F110EnvWrapper:
    def __init__(self, simulator, config):
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

    def _normalize_obs(self, obs_dict):
        scans = obs_dict["scans"]
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def reset(self, poses):
        """シミュレータのAPIに合わせてposesを受け取りリセット"""
        obs_dict, _, _, _ = self.sim.reset(poses)
        return self._normalize_obs(obs_dict)

    def step(self, action_normalized):
        """シミュレータを1ステップ進める"""
        action_physical = self._scale_action(action_normalized)
        next_obs_dict, reward, done, info = self.sim.step(action_physical)

        next_obs = self._normalize_obs(next_obs_dict)
        shaped_reward = reward + (action_physical[:, 1] * 0.1)

        # PPO/GAEが扱いやすいよう done を agent 次元に展開
        done_agents = jnp.full((action_normalized.shape[0],), done, dtype=jnp.float32)
        return next_obs, shaped_reward, done_agents, info

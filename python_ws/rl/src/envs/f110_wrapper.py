import jax
import jax.numpy as jnp

class F110EnvWrapper:
    def __init__(self, simulator, config):
        self.sim = simulator
        self.max_steer = config.max_steer
        self.min_steer = config.min_steer
        self.max_speed = config.max_speed
        self.min_speed = config.min_speed
        self.max_lidar_range = config.max_lidar_range

        self.action_scale = jnp.array([
            (self.max_steer - self.min_steer) / 2.0,
            (self.max_speed - self.min_speed) / 2.0
        ])
        self.action_bias = jnp.array([
            (self.max_steer + self.min_steer) / 2.0,
            (self.max_speed + self.min_speed) / 2.0
        ])

    def _normalize_obs(self, obs_dict):
        scans = obs_dict['scans']
        return jnp.clip(scans, 0.0, self.max_lidar_range) / self.max_lidar_range

    def _scale_action(self, action_normalized):
        return action_normalized * self.action_scale + self.action_bias

    def reset(self, poses):
        """シミュレータのAPIに合わせてposesを受け取りリセット"""
        obs_dict, reward, done, info = self.sim.reset(poses)
        return self._normalize_obs(obs_dict)

    def step(self, action_normalized):
        """内部状態を持つシミュレータなので、stateの受け渡しは不要"""
        action_physical = self._scale_action(action_normalized)
        
        # 物理スケールのアクションでシミュレータを進める
        next_obs_dict, reward, done, info = self.sim.step(action_physical)
        
        # 観測の正規化
        next_obs = self._normalize_obs(next_obs_dict)
        
        # ==========================================
        # 報酬の成形 (Reward Shaping)
        # ==========================================
        # シミュレータの生報酬だけでなく、PPOが学習しやすいように報酬を加工します。
        # 例: 出力した目標速度に比例した報酬を与える (前に進むことを推奨)
        shaped_reward = reward + (action_physical[:, 1] * 0.1)
        
        return next_obs, shaped_reward, done, info
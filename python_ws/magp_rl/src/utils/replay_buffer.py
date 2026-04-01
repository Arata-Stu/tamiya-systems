import numpy as np
import jax.numpy as jnp


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim, seed=0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((self.capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add_batch(self, obs, actions, rewards, next_obs, terminated):
        obs_np = np.asarray(obs, dtype=np.float32)
        actions_np = np.asarray(actions, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        next_obs_np = np.asarray(next_obs, dtype=np.float32)
        term_np = np.asarray(terminated, dtype=np.float32)

        batch_n = int(obs_np.shape[0])
        if batch_n == 0:
            return

        start = self.ptr
        if batch_n >= self.capacity:
            drop = batch_n - self.capacity
            obs_np = obs_np[drop:]
            actions_np = actions_np[drop:]
            rewards_np = rewards_np[drop:]
            next_obs_np = next_obs_np[drop:]
            term_np = term_np[drop:]
            write_n = self.capacity
            start = (self.ptr + drop) % self.capacity
        else:
            write_n = batch_n

        end = start + write_n
        if end <= self.capacity:
            self.obs[start:end] = obs_np
            self.actions[start:end] = actions_np
            self.rewards[start:end] = rewards_np
            self.next_obs[start:end] = next_obs_np
            self.terminated[start:end] = term_np
        else:
            first = self.capacity - start
            second = write_n - first

            self.obs[start:] = obs_np[:first]
            self.obs[:second] = obs_np[first:]
            self.actions[start:] = actions_np[:first]
            self.actions[:second] = actions_np[first:]
            self.rewards[start:] = rewards_np[:first]
            self.rewards[:second] = rewards_np[first:]
            self.next_obs[start:] = next_obs_np[:first]
            self.next_obs[:second] = next_obs_np[first:]
            self.terminated[start:] = term_np[:first]
            self.terminated[:second] = term_np[first:]

        self.ptr = (self.ptr + batch_n) % self.capacity
        self.size = min(self.size + batch_n, self.capacity)

    def can_sample(self, batch_size):
        return self.size >= batch_size

    def sample(self, batch_size):
        idx = self.rng.integers(0, self.size, size=batch_size)
        batch = {
            "obs": jnp.asarray(self.obs[idx]),
            "actions": jnp.asarray(self.actions[idx]),
            "rewards": jnp.asarray(self.rewards[idx]),
            "next_obs": jnp.asarray(self.next_obs[idx]),
            "terminated": jnp.asarray(self.terminated[idx]),
        }
        return batch

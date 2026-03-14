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

        batch_n = obs_np.shape[0]
        for i in range(batch_n):
            self.obs[self.ptr] = obs_np[i]
            self.actions[self.ptr] = actions_np[i]
            self.rewards[self.ptr] = rewards_np[i]
            self.next_obs[self.ptr] = next_obs_np[i]
            self.terminated[self.ptr] = term_np[i]

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

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

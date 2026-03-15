import numpy as np


def generate_initial_poses(num_agents):
    poses = np.zeros((num_agents, 3), dtype=np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = np.linspace(0.0, 0.4 * max(num_agents - 1, 0), num_agents)
    poses[:, 2] = 0.0
    return poses


def generate_independent_poses(num_envs, base_pose=None):
    if base_pose is None:
        base_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    else:
        base_pose = np.asarray(base_pose, dtype=np.float32)
    return np.tile(base_pose[None, :], (int(num_envs), 1))

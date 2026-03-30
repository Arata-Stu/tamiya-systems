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


def resolve_lidar_sim_params(env_cfg):
    """Resolve LiDAR parameters for F110JaxSimulator from Hydra env config.

    Priority:
    1) `scan_fov` if provided
    2) `scan_angle_max - scan_angle_min` if both provided
    3) fallback to simulator default 4.7

    `scan_beams` defaults to `obs_dim` when omitted.
    """
    num_beams = int(env_cfg.get("scan_beams", env_cfg.get("obs_dim", 1080)))
    if num_beams <= 0:
        raise ValueError(f"env.scan_beams must be > 0, got {num_beams}")

    scan_fov = env_cfg.get("scan_fov", None)
    scan_angle_min = env_cfg.get("scan_angle_min", None)
    scan_angle_max = env_cfg.get("scan_angle_max", None)

    if scan_fov is None and (scan_angle_min is not None) and (scan_angle_max is not None):
        scan_angle_min = float(scan_angle_min)
        scan_angle_max = float(scan_angle_max)
        scan_center = 0.5 * (scan_angle_min + scan_angle_max)
        # Current simulator assumes symmetric FOV around heading.
        if abs(scan_center) > 1e-3:
            raise ValueError(
                "Current simulator expects symmetric scan around 0 rad. "
                f"Got angle_min={scan_angle_min}, angle_max={scan_angle_max} "
                f"(center={scan_center})."
            )
        scan_fov = scan_angle_max - scan_angle_min

    if scan_fov is None:
        scan_fov = 4.7
    scan_fov = float(scan_fov)
    if scan_fov <= 0.0:
        raise ValueError(f"LiDAR FOV must be > 0, got {scan_fov}")

    obs_dim = env_cfg.get("obs_dim", None)
    if obs_dim is not None and int(obs_dim) != num_beams:
        raise ValueError(
            "env.obs_dim and env.scan_beams must match. "
            f"Got obs_dim={int(obs_dim)}, scan_beams={num_beams}."
        )

    return num_beams, scan_fov

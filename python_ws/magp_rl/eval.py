from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from omegaconf import DictConfig

from src.agents.ppo import create_train_states, select_action_deterministic
from src.envs.f110_wrapper import F110EnvWrapper
from src.utils.env_assets import resolve_env_assets

from f110_jax.simulator import F110JaxSimulator, Integrator


def generate_initial_poses(num_agents):
    poses = np.zeros((num_agents, 3), dtype=np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = np.linspace(0.0, 0.4 * max(num_agents - 1, 0), num_agents)
    poses[:, 2] = 0.0
    return poses


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    print("=== JAX F1TENTH PPO Evaluation Start ===")

    rng = jax.random.PRNGKey(cfg.train.seed)
    project_root = Path(__file__).resolve().parents[1]
    map_path, map_ext, waypoints_path = resolve_env_assets(cfg.env, project_root)
    print(f"Map: {map_path}")
    print(f"Waypoints: {waypoints_path}")

    sim = F110JaxSimulator(
        map_path=map_path,
        map_ext=map_ext,
        num_agents=cfg.eval.num_agents,
        integrator=Integrator.RK4,
    )
    env = F110EnvWrapper(sim, cfg.env, waypoints_path=waypoints_path)

    obs_shape = (cfg.env.obs_dim,)
    actor_state, _ = create_train_states(
        rng,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.train.actor_lr,
        critic_lr=cfg.train.critic_lr,
    )

    ckpt_dir = Path(cfg.eval.checkpoint_dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = Path.cwd() / ckpt_dir

    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target={"actor_state": actor_state, "critic_state": None, "update": 0},
    )
    actor_state = restored["actor_state"]
    print(f"Loaded checkpoint update={restored['update']} from: {ckpt_dir}")

    poses = generate_initial_poses(cfg.eval.num_agents)

    returns = []
    lengths = []
    collision_rates = []

    for ep in range(cfg.eval.episodes):
        obs = env.reset(poses)
        episode_return = 0.0
        episode_steps = 0
        collided = False

        for _ in range(cfg.eval.max_steps):
            action = select_action_deterministic(actor_state, obs)
            obs, reward, done, info = env.step(action)

            episode_return += float(jax.device_get(jnp.mean(reward)))
            episode_steps += 1

            if "checkpoint_done" in info:
                lap_done = bool(jax.device_get(jnp.any(info["checkpoint_done"])))
            else:
                lap_done = False

            done_flag = bool(jax.device_get(jnp.any(done))) or lap_done
            if done_flag:
                scans = sim.sim_state["collisions"]
                collided = bool(jax.device_get(jnp.any(scans > 0.0)))
                break

        returns.append(episode_return)
        lengths.append(episode_steps)
        collision_rates.append(1.0 if collided else 0.0)
        print(
            f"Episode {ep + 1}/{cfg.eval.episodes} | Return: {episode_return:.3f} "
            f"| Length: {episode_steps} | Collided: {collided}"
        )

    print("=== Eval Summary ===")
    print(f"Average Return: {np.mean(returns):.3f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Collision Rate: {np.mean(collision_rates):.3f}")


if __name__ == "__main__":
    main()

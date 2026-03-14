from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from f110_jax.simulator import F110JaxSimulator, Integrator
from src.agents.ppo import create_train_states, select_action_deterministic
from src.agents.sac import create_sac_states, sac_act_deterministic
from src.envs.f110_wrapper import F110EnvWrapper
from src.utils.common import generate_initial_poses
from src.utils.env_assets import resolve_env_assets


def _restore_ppo_actor(cfg: DictConfig, rng, obs_shape):
    actor_state, _ = create_train_states(
        rng,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
    )
    target = {"actor_state": actor_state, "critic_state": None, "update": 0}
    return actor_state, target, "update"


def _restore_sac_actor(cfg: DictConfig, rng, obs_shape):
    (
        actor_state,
        critic1_state,
        critic2_state,
        target_critic1_params,
        target_critic2_params,
        alpha_state,
    ) = create_sac_states(
        rng,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
        alpha_lr=cfg.agent.alpha_lr,
        init_temperature=cfg.agent.init_temperature,
    )

    target = {
        "actor_state": actor_state,
        "critic1_state": critic1_state,
        "critic2_state": critic2_state,
        "target_critic1_params": target_critic1_params,
        "target_critic2_params": target_critic2_params,
        "alpha_state": alpha_state,
        "global_step": 0,
    }
    return actor_state, target, "global_step"


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig):
    print("=== JAX F1TENTH RL Evaluation Start ===")
    print(f"Algorithm: {cfg.agent.name}")

    rng = jax.random.PRNGKey(cfg.eval.seed)
    project_root = Path(get_original_cwd())
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
    if cfg.agent.name == "ppo":
        actor_state, restore_target, step_key = _restore_ppo_actor(cfg, rng, obs_shape)
        act_fn = lambda actor, obs: select_action_deterministic(actor, obs)
    elif cfg.agent.name == "sac":
        actor_state, restore_target, step_key = _restore_sac_actor(cfg, rng, obs_shape)
        act_fn = lambda actor, obs: sac_act_deterministic(actor, obs)
    else:
        raise ValueError(f"Unsupported agent for eval: {cfg.agent.name}")

    ckpt_dir = Path(cfg.eval.checkpoint_dir)
    if cfg.eval.checkpoint_dir is None:
        raise ValueError("Set eval.checkpoint_dir to a checkpoint run directory.")
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir

    restored = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=restore_target)
    actor_state = restored["actor_state"]
    print(f"Loaded checkpoint {step_key}={restored[step_key]} from: {ckpt_dir}")

    poses = generate_initial_poses(cfg.eval.num_agents)

    returns = []
    lengths = []
    collision_rates = []
    progresses = []
    avg_speeds = []
    completion_flags = []

    for ep in range(cfg.eval.episodes):
        obs = env.reset(poses)
        prev_s = env.get_current_progress_s()
        episode_return = 0.0
        episode_steps = 0
        episode_progress = 0.0
        speed_sum = 0.0
        speed_count = 0
        collided = False
        completed = False

        for _ in range(cfg.eval.max_steps):
            action = act_fn(actor_state, obs)
            obs, reward, done, info = env.step(action)

            episode_return += float(jax.device_get(jnp.mean(reward)))
            episode_steps += 1

            current_s = env.get_current_progress_s()
            progress_delta = env.compute_progress_delta(current_s, prev_s)
            episode_progress += float(jax.device_get(jnp.mean(progress_delta)))
            prev_s = current_s

            speeds = jnp.abs(sim.sim_state["state"][:, 3])
            speed_sum += float(jax.device_get(jnp.mean(speeds)))
            speed_count += 1

            if "checkpoint_done" in info:
                lap_done = bool(jax.device_get(jnp.any(info["checkpoint_done"])))
            else:
                lap_done = False
            if lap_done:
                completed = True

            done_flag = bool(jax.device_get(jnp.any(done))) or lap_done
            if done_flag:
                scans = sim.sim_state["collisions"]
                collided = bool(jax.device_get(jnp.any(scans > 0.0)))
                break

        returns.append(episode_return)
        lengths.append(episode_steps)
        collision_rates.append(1.0 if collided else 0.0)
        progresses.append(episode_progress)
        avg_speed = speed_sum / max(speed_count, 1)
        avg_speeds.append(avg_speed)
        completion_flags.append(1.0 if completed else 0.0)
        print(
            f"Episode {ep + 1}/{cfg.eval.episodes} | Return: {episode_return:.3f} "
            f"| Length: {episode_steps} | Progress(m): {episode_progress:.3f} "
            f"| AvgSpeed(m/s): {avg_speed:.3f} | Completed: {completed} | Collided: {collided}"
        )

    print("=== Eval Summary ===")
    print(f"Average Return: {np.mean(returns):.3f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Average Progress (m): {np.mean(progresses):.3f}")
    print(f"Average Speed (m/s): {np.mean(avg_speeds):.3f}")
    print(f"Completion Rate: {np.mean(completion_flags):.3f}")
    print(f"Collision Rate: {np.mean(collision_rates):.3f}")


if __name__ == "__main__":
    main()

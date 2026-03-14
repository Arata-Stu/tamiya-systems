from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
from flax.training import checkpoints
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from f110_jax.simulator import F110JaxSimulator, Integrator
from src.agents.ppo import create_train_states, select_action, update_step
from src.agents.sac import (
    create_sac_states,
    sac_act,
    sac_update_step,
)
from src.envs.f110_wrapper import F110EnvWrapper
from src.utils.buffer import RolloutBuffer, compute_gae
from src.utils.common import generate_initial_poses
from src.utils.env_assets import resolve_env_assets
from src.utils.replay_buffer import ReplayBuffer

def maybe_make_writer(cfg: DictConfig, base_dir: Path):
    if not cfg.train.tensorboard.enabled:
        return None

    log_dir = Path(cfg.train.tensorboard.log_dir)
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def maybe_restore_checkpoint(cfg: DictConfig, ckpt_dir: Path, target):
    if not cfg.train.checkpoint.resume:
        return target

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    restored = checkpoints.restore_checkpoint(ckpt_dir=str(ckpt_dir), target=target)

    step_key = "update" if "update" in restored else "global_step"
    if int(restored.get(step_key, 0)) == 0:
        print(f"No checkpoint found in: {ckpt_dir}. Start from scratch.")
    else:
        print(f"Resumed from checkpoint {step_key}={restored[step_key]} in: {ckpt_dir}")
    return restored


def train_ppo(cfg, env, writer, ckpt_dir, rng):
    rng, rng_agent = jax.random.split(rng, 2)

    obs_shape = (cfg.env.obs_dim,)
    actor_state, critic_state = create_train_states(
        rng_agent,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
    )

    restore_target = {"actor_state": actor_state, "critic_state": critic_state, "update": 0}
    restored = maybe_restore_checkpoint(cfg, ckpt_dir, restore_target)
    actor_state = restored["actor_state"]
    critic_state = restored["critic_state"]
    start_update = int(restored["update"])

    buffer = RolloutBuffer()
    num_updates = cfg.train.total_timesteps // (cfg.agent.num_steps * cfg.train.num_agents)

    poses = generate_initial_poses(cfg.train.num_agents)
    obs = env.reset(poses)

    episode_return = jnp.array(0.0, dtype=jnp.float32)
    episode_len = 0
    num_episodes = 0

    for update in range(start_update, num_updates):
        buffer.reset()

        for _ in tqdm(range(cfg.agent.num_steps), desc=f"Update {update + 1}/{num_updates}"):
            rng, rng_action = jax.random.split(rng)

            action, log_prob = select_action(actor_state, obs, rng_action)
            value = critic_state.apply_fn(critic_state.params, obs).squeeze()

            next_obs, reward, terminated, _ = env.step(action)

            episode_len += 1
            timeout = float(episode_len >= cfg.train.max_episode_steps)
            timeout_vec = jnp.full_like(terminated, timeout)
            done_for_gae = jnp.maximum(terminated, timeout_vec)

            buffer.add(obs, action, reward, done_for_gae, value, log_prob)
            episode_return = episode_return + jnp.mean(reward)
            obs = next_obs

            done_flag = bool(jax.device_get(jnp.any(terminated))) or bool(timeout)
            if done_flag:
                num_episodes += 1
                ep_ret = float(jax.device_get(episode_return))
                if writer is not None:
                    writer.add_scalar("episode/return", ep_ret, num_episodes)
                    writer.add_scalar("episode/length", episode_len, num_episodes)

                episode_return = jnp.array(0.0, dtype=jnp.float32)
                episode_len = 0
                obs = env.reset(poses)

        last_value = critic_state.apply_fn(critic_state.params, obs).squeeze()
        data = buffer.get_stacked()
        advantages, returns = compute_gae(
            data["rewards"],
            data["values"],
            data["dones"],
            last_value,
            gamma=cfg.agent.gamma,
            gae_lambda=cfg.agent.gae_lambda,
        )

        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        b_obs = flatten(data["obs"])
        b_actions = flatten(data["actions"])
        b_log_probs = flatten(data["log_probs"])
        b_returns = flatten(returns)
        b_advantages = flatten(advantages)

        for _ in range(cfg.agent.update_epochs):
            actor_state, critic_state, metrics = update_step(
                actor_state,
                critic_state,
                b_obs,
                b_actions,
                b_log_probs,
                b_returns,
                b_advantages,
                clip_eps=cfg.agent.clip_eps,
                entropy_coef=cfg.agent.entropy_coef,
            )

        if writer is not None and (update + 1) % cfg.train.tensorboard.log_every_updates == 0:
            metrics_host = jax.device_get(metrics)
            writer.add_scalar("loss/actor", float(metrics_host["actor_loss"]), update + 1)
            writer.add_scalar("loss/critic", float(metrics_host["critic_loss"]), update + 1)
            writer.add_scalar("policy/entropy", float(metrics_host["entropy"]), update + 1)

        if (update + 1) % cfg.train.checkpoint.save_every_updates == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=str(ckpt_dir),
                target={
                    "actor_state": actor_state,
                    "critic_state": critic_state,
                    "update": update + 1,
                },
                step=update + 1,
                overwrite=True,
                keep=cfg.train.checkpoint.keep,
            )

        metrics_host = jax.device_get(metrics)
        print(
            f"[PPO] Update {update + 1} | Actor Loss: {float(metrics_host['actor_loss']):.4f} "
            f"| Critic Loss: {float(metrics_host['critic_loss']):.4f} | Episodes: {num_episodes}"
        )


def train_sac(cfg, env, writer, ckpt_dir, rng):
    rng, rng_agent = jax.random.split(rng, 2)

    obs_shape = (cfg.env.obs_dim,)
    (
        actor_state,
        critic1_state,
        critic2_state,
        target_critic1_params,
        target_critic2_params,
        alpha_state,
    ) = create_sac_states(
        rng_agent,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
        alpha_lr=cfg.agent.alpha_lr,
        init_temperature=cfg.agent.init_temperature,
    )

    restore_target = {
        "actor_state": actor_state,
        "critic1_state": critic1_state,
        "critic2_state": critic2_state,
        "target_critic1_params": target_critic1_params,
        "target_critic2_params": target_critic2_params,
        "alpha_state": alpha_state,
        "global_step": 0,
    }
    restored = maybe_restore_checkpoint(cfg, ckpt_dir, restore_target)
    actor_state = restored["actor_state"]
    critic1_state = restored["critic1_state"]
    critic2_state = restored["critic2_state"]
    target_critic1_params = restored["target_critic1_params"]
    target_critic2_params = restored["target_critic2_params"]
    alpha_state = restored["alpha_state"]
    global_step = int(restored["global_step"])

    replay = ReplayBuffer(
        capacity=cfg.agent.replay_size,
        obs_dim=cfg.env.obs_dim,
        action_dim=cfg.env.action_dim,
        seed=cfg.train.seed,
    )

    poses = generate_initial_poses(cfg.train.num_agents)
    obs = env.reset(poses)

    episode_return = jnp.array(0.0, dtype=jnp.float32)
    episode_len = 0
    num_episodes = 0
    last_metrics = None

    pbar = tqdm(total=cfg.train.total_timesteps, initial=global_step, desc="SAC steps")
    target_entropy = -float(cfg.env.action_dim) * cfg.agent.target_entropy_scale

    while global_step < cfg.train.total_timesteps:
        if global_step < cfg.agent.start_steps:
            rng, rng_random = jax.random.split(rng)
            action = jax.random.uniform(
                rng_random,
                shape=(cfg.train.num_agents, cfg.env.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            rng, rng_action = jax.random.split(rng)
            action = sac_act(actor_state, obs, rng_action)

        next_obs, reward, terminated, _ = env.step(action)

        episode_len += 1
        timeout = float(episode_len >= cfg.train.max_episode_steps)

        replay.add_batch(
            jax.device_get(obs),
            jax.device_get(action),
            jax.device_get(reward),
            jax.device_get(next_obs),
            jax.device_get(terminated),
        )

        obs = next_obs
        episode_return = episode_return + jnp.mean(reward)

        if global_step >= cfg.agent.update_after and replay.can_sample(cfg.agent.batch_size):
            for _ in range(cfg.agent.updates_per_step):
                batch = replay.sample(cfg.agent.batch_size)
                rng, rng_update = jax.random.split(rng)
                (
                    actor_state,
                    critic1_state,
                    critic2_state,
                    target_critic1_params,
                    target_critic2_params,
                    alpha_state,
                    last_metrics,
                ) = sac_update_step(
                    actor_state,
                    critic1_state,
                    critic2_state,
                    target_critic1_params,
                    target_critic2_params,
                    alpha_state,
                    batch["obs"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_obs"],
                    batch["terminated"],
                    rng_update,
                    gamma=cfg.agent.gamma,
                    tau=cfg.agent.tau,
                    target_entropy=target_entropy,
                )

        done_flag = bool(jax.device_get(jnp.any(terminated))) or bool(timeout)
        if done_flag:
            num_episodes += 1
            ep_ret = float(jax.device_get(episode_return))
            if writer is not None:
                writer.add_scalar("episode/return", ep_ret, num_episodes)
                writer.add_scalar("episode/length", episode_len, num_episodes)

            episode_return = jnp.array(0.0, dtype=jnp.float32)
            episode_len = 0
            obs = env.reset(poses)

        global_step += cfg.train.num_agents
        pbar.update(cfg.train.num_agents)

        if writer is not None and last_metrics is not None and global_step % cfg.train.tensorboard.log_every_updates == 0:
            metrics_host = jax.device_get(last_metrics)
            writer.add_scalar("sac/actor_loss", float(metrics_host["actor_loss"]), global_step)
            writer.add_scalar("sac/critic1_loss", float(metrics_host["critic1_loss"]), global_step)
            writer.add_scalar("sac/critic2_loss", float(metrics_host["critic2_loss"]), global_step)
            writer.add_scalar("sac/alpha_loss", float(metrics_host["alpha_loss"]), global_step)
            writer.add_scalar("sac/alpha", float(metrics_host["alpha"]), global_step)
            writer.add_scalar("sac/q_target_mean", float(metrics_host["q_target_mean"]), global_step)

        if global_step % cfg.train.checkpoint.save_every_updates == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=str(ckpt_dir),
                target={
                    "actor_state": actor_state,
                    "critic1_state": critic1_state,
                    "critic2_state": critic2_state,
                    "target_critic1_params": target_critic1_params,
                    "target_critic2_params": target_critic2_params,
                    "alpha_state": alpha_state,
                    "global_step": global_step,
                },
                step=global_step,
                overwrite=True,
                keep=cfg.train.checkpoint.keep,
            )

        if global_step % max(cfg.train.tensorboard.log_every_updates, cfg.train.num_agents) == 0 and last_metrics is not None:
            metrics_host = jax.device_get(last_metrics)
            print(
                f"[SAC] Step {global_step} | Actor: {float(metrics_host['actor_loss']):.4f} "
                f"| Critic1: {float(metrics_host['critic1_loss']):.4f} "
                f"| Alpha: {float(metrics_host['alpha']):.4f} | Episodes: {num_episodes}"
            )

    pbar.close()


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    print("=== JAX F1TENTH RL Training Start ===")
    print(f"Algorithm: {cfg.agent.name}")
    print(f"JAX backend: {jax.default_backend()}")

    project_root = Path(get_original_cwd())

    map_path, map_ext, waypoints_path = resolve_env_assets(cfg.env, project_root)
    print(f"Map: {map_path}")
    print(f"Waypoints: {waypoints_path}")

    rng = jax.random.PRNGKey(cfg.train.seed)

    sim = F110JaxSimulator(
        map_path=map_path,
        map_ext=map_ext,
        num_agents=cfg.train.num_agents,
        integrator=Integrator.RK4,
    )
    env = F110EnvWrapper(sim, cfg.env, waypoints_path=waypoints_path)

    writer = maybe_make_writer(cfg, project_root)

    ckpt_dir = Path(cfg.train.checkpoint.dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.agent.name == "ppo":
        train_ppo(cfg, env, writer, ckpt_dir, rng)
    elif cfg.agent.name == "sac":
        train_sac(cfg, env, writer, ckpt_dir, rng)
    else:
        raise ValueError(f"Unsupported agent: {cfg.agent.name}")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.agents.ppo import create_train_states, select_action, update_step
from src.envs.f110_wrapper import F110EnvWrapper
from src.utils.buffer import RolloutBuffer, compute_gae

from f110_jax.simulator import F110JaxSimulator, Integrator


def generate_initial_poses(num_agents):
    """車両同士の重なりを避けて初期姿勢を生成"""
    poses = np.zeros((num_agents, 3), dtype=np.float32)
    poses[:, 0] = 0.0
    poses[:, 1] = np.linspace(0.0, 0.4 * max(num_agents - 1, 0), num_agents)
    poses[:, 2] = 0.0
    return poses


def maybe_make_writer(cfg: DictConfig, run_dir: Path):
    if not cfg.train.tensorboard.enabled:
        return None

    log_dir = Path(cfg.train.tensorboard.log_dir)
    if not log_dir.is_absolute():
        log_dir = run_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def maybe_restore_checkpoint(cfg: DictConfig, ckpt_dir: Path, actor_state, critic_state):
    if not cfg.train.checkpoint.resume:
        return actor_state, critic_state, 0

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    restored = checkpoints.restore_checkpoint(
        ckpt_dir=str(ckpt_dir),
        target={"actor_state": actor_state, "critic_state": critic_state, "update": 0},
    )
    if restored["update"] == 0:
        print(f"No checkpoint found in: {ckpt_dir}. Start from scratch.")
    else:
        print(f"Resumed from checkpoint update={restored['update']} in: {ckpt_dir}")

    return restored["actor_state"], restored["critic_state"], int(restored["update"])


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    print("=== JAX F1TENTH PPO Training Start ===")
    print(f"JAX backend: {jax.default_backend()}")

    run_dir = Path(os.getcwd())

    rng = jax.random.PRNGKey(cfg.train.seed)
    rng, rng_agent = jax.random.split(rng, 2)

    sim = F110JaxSimulator(
        map_path=cfg.env.map_path,
        map_ext=cfg.env.map_ext,
        num_agents=cfg.train.num_agents,
        integrator=Integrator.RK4,
    )
    env = F110EnvWrapper(sim, cfg.env)

    obs_shape = (cfg.env.obs_dim,)
    actor_state, critic_state = create_train_states(
        rng_agent,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.train.actor_lr,
        critic_lr=cfg.train.critic_lr,
    )

    writer = maybe_make_writer(cfg, run_dir)

    ckpt_dir = Path(cfg.train.checkpoint.dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = run_dir / ckpt_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    actor_state, critic_state, start_update = maybe_restore_checkpoint(
        cfg, ckpt_dir, actor_state, critic_state
    )

    buffer = RolloutBuffer()
    num_updates = cfg.train.total_timesteps // (cfg.train.num_steps * cfg.train.num_agents)

    poses = generate_initial_poses(cfg.train.num_agents)
    obs = env.reset(poses)

    # episode stats are tracked on device and only transferred on episode end.
    episode_return = jnp.array(0.0, dtype=jnp.float32)
    episode_len = 0
    num_episodes = 0

    for update in range(start_update, num_updates):
        buffer.reset()

        for _ in tqdm(range(cfg.train.num_steps), desc=f"Update {update + 1}/{num_updates}"):
            rng, rng_action = jax.random.split(rng)

            action, log_prob = select_action(actor_state, obs, rng_action)
            value = critic_state.apply_fn(critic_state.params, obs).squeeze()

            next_obs, reward, done, _ = env.step(action)
            done_float = done.astype(jnp.float32)

            buffer.add(obs, action, reward, done_float, value, log_prob)

            # reduce reward vector on device to avoid frequent host transfer.
            episode_return = episode_return + jnp.mean(reward)
            episode_len += 1
            obs = next_obs

            done_flag = bool(jax.device_get(jnp.any(done)))
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
            gamma=cfg.train.gamma,
            gae_lambda=cfg.train.gae_lambda,
        )

        flatten = lambda x: x.reshape(-1, *x.shape[2:])
        b_obs = flatten(data["obs"])
        b_actions = flatten(data["actions"])
        b_log_probs = flatten(data["log_probs"])
        b_returns = flatten(returns)
        b_advantages = flatten(advantages)

        for _ in range(cfg.train.update_epochs):
            actor_state, critic_state, metrics = update_step(
                actor_state,
                critic_state,
                b_obs,
                b_actions,
                b_log_probs,
                b_returns,
                b_advantages,
                clip_eps=cfg.train.clip_eps,
                entropy_coef=cfg.train.entropy_coef,
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
            f"Update {update + 1} | Actor Loss: {float(metrics_host['actor_loss']):.4f} "
            f"| Critic Loss: {float(metrics_host['critic_loss']):.4f} "
            f"| Episodes: {num_episodes}"
        )

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

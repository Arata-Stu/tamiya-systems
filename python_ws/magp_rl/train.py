from pathlib import Path
from datetime import datetime
import logging
import math
import re
import shutil

import hydra
import jax
import jax.numpy as jnp
import numpy as np
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
from src.agents.td3 import (
    create_td3_states,
    td3_act,
    td3_soft_update_target_critic2,
    td3_update_actor_and_targets,
    td3_update_critics,
)
from src.envs.f110_independent_vec import F110IndependentVecEnv
from src.envs.f110_wrapper import F110EnvWrapper
from src.utils.buffer import RolloutBuffer, compute_gae
from src.utils.common import (
    generate_independent_poses,
    generate_initial_poses,
    resolve_lidar_sim_params,
)
from src.utils.env_assets import resolve_env_assets
from src.utils.replay_buffer import ReplayBuffer
from src.utils.vehicle import resolve_vehicle_params


def maybe_make_writer(cfg: DictConfig, base_dir: Path):
    if not cfg.train.tensorboard.enabled:
        return None

    log_dir = Path(cfg.train.tensorboard.log_dir)
    if not log_dir.is_absolute():
        log_dir = base_dir / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))


def _sanitize_path_token(text: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return token if token else "unknown"


def _unique_child_dir(parent: Path, base_name: str) -> Path:
    candidate = parent / base_name
    if not candidate.exists():
        return candidate

    idx = 1
    while True:
        suffixed = parent / f"{base_name}_{idx:02d}"
        if not suffixed.exists():
            return suffixed
        idx += 1


def maybe_autofork_checkpoint_dir(cfg: DictConfig, ckpt_dir: Path) -> Path:
    ckpt_cfg = cfg.train.checkpoint
    if not bool(ckpt_cfg.resume):
        return ckpt_dir
    if not bool(ckpt_cfg.get("auto_fork_on_resume", False)):
        return ckpt_dir

    latest = checkpoints.latest_checkpoint(str(ckpt_dir))
    if latest is None:
        print(
            "Checkpoint auto-fork enabled, but source has no checkpoint yet. "
            f"Continue in-place: {ckpt_dir}"
        )
        return ckpt_dir

    map_name = _sanitize_path_token(cfg.env.track.get("name", "map"))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fork_name = f"{ckpt_dir.name}__fork_{map_name}_{stamp}"
    fork_dir = _unique_child_dir(ckpt_dir.parent, fork_name)
    shutil.copytree(ckpt_dir, fork_dir)
    print(
        "Checkpoint auto-fork enabled. "
        f"Copied source checkpoint dir:\n  from: {ckpt_dir}\n  to  : {fork_dir}"
    )
    return fork_dir


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


def get_parallel_mode(env_cfg: DictConfig) -> str:
    parallel_cfg = env_cfg.get("parallel", None)
    if parallel_cfg is None:
        return "independent"
    return str(parallel_cfg.get("mode", "independent")).lower()


def get_race_control_mode(env_cfg: DictConfig) -> str:
    race_cfg = env_cfg.get("race", None)
    if race_cfg is None:
        return "selfplay"
    return str(race_cfg.get("control_mode", "selfplay")).lower()


def get_race_num_agents(env_cfg: DictConfig, fallback: int) -> int:
    race_cfg = env_cfg.get("race", None)
    if race_cfg is None:
        return int(fallback)
    configured = race_cfg.get("num_agents", None)
    if configured is None:
        return int(fallback)
    return int(configured)


def get_race_ego_idx(env_cfg: DictConfig) -> int:
    race_cfg = env_cfg.get("race", None)
    if race_cfg is None:
        return 0
    return int(race_cfg.get("ego_idx", 0))


def get_race_npc_controller(env_cfg: DictConfig) -> str:
    race_cfg = env_cfg.get("race", None)
    if race_cfg is None:
        return "pure_pursuit"
    npc_cfg = race_cfg.get("npc", None)
    if npc_cfg is None:
        return "pure_pursuit"
    return str(npc_cfg.get("controller", "pure_pursuit"))


def get_train_env_count(cfg: DictConfig) -> int:
    legacy = cfg.train.get("num_agents", None)
    if legacy is not None:
        return int(legacy)
    return int(cfg.train.num_envs)


def build_env(
    cfg: DictConfig,
    map_path: str,
    map_ext: str,
    waypoints_path: str,
    num_envs: int,
    vehicle_params,
    scan_beams: int,
    scan_fov: float,
    max_lidar_range: float,
):
    parallel_mode = get_parallel_mode(cfg.env)
    if parallel_mode == "independent":
        sim = F110JaxSimulator(
            map_path=map_path,
            map_ext=map_ext,
            num_agents=1,
            params=vehicle_params,
            seed=cfg.train.seed,
            integrator=Integrator.RK4,
            num_beams=scan_beams,
            fov=scan_fov,
            max_range=max_lidar_range,
        )
        env = F110IndependentVecEnv(
            sim,
            cfg.env,
            num_envs=num_envs,
            waypoints_path=waypoints_path,
            seed=cfg.train.seed,
        )
    elif parallel_mode == "race":
        race_control_mode = get_race_control_mode(cfg.env)
        race_num_agents = get_race_num_agents(cfg.env, fallback=num_envs)
        race_ego_idx = get_race_ego_idx(cfg.env)
        race_npc_controller = get_race_npc_controller(cfg.env)
        if race_control_mode == "npc" and race_num_agents < 2:
            raise ValueError("env.race.num_agents must be >= 2 when env.race.control_mode=npc.")
        sim = F110JaxSimulator(
            map_path=map_path,
            map_ext=map_ext,
            num_agents=race_num_agents,
            params=vehicle_params,
            seed=cfg.train.seed,
            integrator=Integrator.RK4,
            num_beams=scan_beams,
            fov=scan_fov,
            max_range=max_lidar_range,
        )
        env = F110EnvWrapper(
            sim,
            cfg.env,
            waypoints_path=waypoints_path,
            control_mode=race_control_mode,
            npc_controller=race_npc_controller,
            ego_idx=race_ego_idx,
        )
    else:
        raise ValueError(f"Unsupported env.parallel.mode: {parallel_mode}")

    return env, parallel_mode


def make_start_poses(parallel_mode: str, num_envs: int, num_sim_agents: int | None = None):
    if parallel_mode == "independent":
        return generate_independent_poses(num_envs)
    agent_count = int(num_envs if num_sim_agents is None else num_sim_agents)
    return generate_initial_poses(agent_count)


def _compute_tal_coef(cfg: DictConfig, global_step: int):
    tal_cfg = cfg.env.reward.get("tal", None)
    if tal_cfg is None or (not bool(tal_cfg.get("enabled", False))):
        return None

    base_coef = float(tal_cfg.get("coef", 0.0))
    schedule_cfg = tal_cfg.get("schedule", None)
    if schedule_cfg is None or (not bool(schedule_cfg.get("enabled", False))):
        return base_coef

    start_step = int(schedule_cfg.get("start_step", 0))
    decay_steps = int(schedule_cfg.get("decay_steps", 1))
    decay_steps = max(1, decay_steps)
    coef_min = float(schedule_cfg.get("coef_min", 0.0))
    mode = str(schedule_cfg.get("mode", "linear")).lower()

    if global_step <= start_step:
        return base_coef

    progress = min(max((global_step - start_step) / decay_steps, 0.0), 1.0)
    if mode == "linear":
        coef = base_coef + (coef_min - base_coef) * progress
    elif mode == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        coef = coef_min + (base_coef - coef_min) * cosine
    else:
        coef = base_coef + (coef_min - base_coef) * progress

    return float(max(coef, 0.0))


def _apply_tal_schedule(env, cfg: DictConfig, global_step: int):
    coef = _compute_tal_coef(cfg, global_step)
    if coef is None:
        return None
    if hasattr(env, "set_tal_coef"):
        env.set_tal_coef(coef)
    return coef


def _log_finished_episodes(
    writer,
    env,
    done_mask,
    episode_return,
    episode_progress,
    episode_len,
    num_episodes,
    global_step,
    first_progress_100_state,
):
    done_host = np.asarray(jax.device_get(done_mask))
    if not np.any(done_host):
        return num_episodes

    ret_host = np.asarray(jax.device_get(episode_return))
    prog_host = np.asarray(jax.device_get(episode_progress))
    len_host = np.asarray(jax.device_get(episode_len))

    for idx in np.where(done_host)[0]:
        num_episodes += 1
        ep_ret = float(ret_host[idx])
        ep_prog = float(prog_host[idx])
        ep_len = int(len_host[idx])
        ep_prog_pct = (ep_prog / max(env.track_length, 1e-6)) * 100.0

        if writer is not None:
            writer.add_scalar("episode/return", ep_ret, num_episodes)
            writer.add_scalar("episode/length", ep_len, num_episodes)
            writer.add_scalar("episode/progress_m", ep_prog, num_episodes)
            writer.add_scalar("episode/progress_pct", ep_prog_pct, num_episodes)

        if (not first_progress_100_state["logged"]) and ep_prog_pct >= 100.0:
            first_progress_100_state["logged"] = True
            first_progress_100_state["global_step"] = int(global_step)
            first_progress_100_state["episode"] = int(num_episodes)
            first_progress_100_state["progress_pct"] = float(ep_prog_pct)
            first_progress_100_state["progress_m"] = float(ep_prog)
            print(
                "[Milestone] First progress >=100% reached | "
                f"step={int(global_step)} | episode={num_episodes} | "
                f"progress={ep_prog_pct:.2f}% ({ep_prog:.3f} m)"
            )
            if writer is not None:
                writer.add_scalar("milestone/first_progress_100_step", float(global_step), 1)
                writer.add_scalar("milestone/first_progress_100_episode", float(num_episodes), 1)
                writer.add_scalar("milestone/first_progress_100_pct", float(ep_prog_pct), 1)
                writer.add_scalar("milestone/first_progress_100_m", float(ep_prog), 1)

    return num_episodes


def train_ppo(cfg, env, writer, ckpt_dir, rng, num_envs, parallel_mode):
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
    denom = cfg.agent.num_steps * num_envs
    num_updates = max(1, int(np.ceil(cfg.train.total_timesteps / max(denom, 1))))

    race_agents = int(getattr(env, "num_agents", num_envs))
    poses = make_start_poses(parallel_mode, num_envs, num_sim_agents=race_agents)
    obs = env.reset(poses)
    prev_s = env.get_current_progress_s()

    episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_progress = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_len = jnp.zeros((num_envs,), dtype=jnp.int32)
    num_episodes = 0
    global_step = start_update * cfg.agent.num_steps * num_envs
    first_progress_100_state = {
        "logged": False,
        "global_step": None,
        "episode": None,
        "progress_pct": None,
        "progress_m": None,
    }

    for update in range(start_update, num_updates):
        buffer.reset()
        tal_coef = _compute_tal_coef(cfg, global_step)

        for _ in tqdm(range(cfg.agent.num_steps), desc=f"Update {update + 1}/{num_updates}"):
            rng, rng_action = jax.random.split(rng)
            step_after_transition = global_step + num_envs
            tal_coef = _apply_tal_schedule(env, cfg, global_step)

            action, log_prob = select_action(actor_state, obs, rng_action)
            value = critic_state.apply_fn(critic_state.params, obs).squeeze(-1)

            next_obs, reward, terminated, _ = env.step(action)
            current_s = env.get_current_progress_s()
            progress_delta = env.compute_progress_delta(current_s, prev_s)

            episode_len = episode_len + 1
            timeout_mask = episode_len >= int(cfg.train.max_episode_steps)
            terminated_mask = terminated > 0.5
            done_mask = jnp.logical_or(terminated_mask, timeout_mask)
            done_for_gae = done_mask.astype(jnp.float32)

            buffer.add(obs, action, reward, done_for_gae, value, log_prob)
            episode_return = episode_return + reward
            episode_progress = episode_progress + progress_delta

            obs = next_obs
            prev_s = current_s

            if bool(jax.device_get(jnp.any(done_mask))):
                num_episodes = _log_finished_episodes(
                    writer,
                    env,
                    done_mask,
                    episode_return,
                    episode_progress,
                    episode_len,
                    num_episodes,
                    global_step=step_after_transition,
                    first_progress_100_state=first_progress_100_state,
                )
                obs, prev_s = env.reset_done(done_mask, poses)
                episode_return = jnp.where(done_mask, 0.0, episode_return)
                episode_progress = jnp.where(done_mask, 0.0, episode_progress)
                episode_len = jnp.where(done_mask, jnp.zeros_like(episode_len), episode_len)

            global_step = step_after_transition

        last_value = critic_state.apply_fn(critic_state.params, obs).squeeze(-1)
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
            if tal_coef is not None:
                writer.add_scalar("reward/tal_coef", float(tal_coef), global_step)

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


def train_sac(cfg, env, writer, ckpt_dir, rng, num_envs, parallel_mode):
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

    race_agents = int(getattr(env, "num_agents", num_envs))
    poses = make_start_poses(parallel_mode, num_envs, num_sim_agents=race_agents)
    obs = env.reset(poses)
    prev_s = env.get_current_progress_s()

    episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_progress = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_len = jnp.zeros((num_envs,), dtype=jnp.int32)
    num_episodes = 0
    last_metrics = None
    first_progress_100_state = {
        "logged": False,
        "global_step": None,
        "episode": None,
        "progress_pct": None,
        "progress_m": None,
    }

    pbar = tqdm(total=cfg.train.total_timesteps, initial=global_step, desc="SAC steps")
    target_entropy = -float(cfg.env.action_dim) * cfg.agent.target_entropy_scale
    print_every_steps = int(cfg.agent.get("print_every_steps", 1000))
    tb_log_every_steps = int(cfg.agent.get("tb_log_every_steps", print_every_steps))
    checkpoint_every_steps = int(cfg.agent.get("checkpoint_every_steps", 5000))

    while global_step < cfg.train.total_timesteps:
        step_after_transition = global_step + num_envs
        tal_coef = _apply_tal_schedule(env, cfg, global_step)
        if global_step < cfg.agent.start_steps:
            rng, rng_random = jax.random.split(rng)
            action = jax.random.uniform(
                rng_random,
                shape=(num_envs, cfg.env.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            rng, rng_action = jax.random.split(rng)
            action = sac_act(actor_state, obs, rng_action)

        next_obs, reward, terminated, _ = env.step(action)
        current_s = env.get_current_progress_s()
        progress_delta = env.compute_progress_delta(current_s, prev_s)

        episode_len = episode_len + 1
        timeout_mask = episode_len >= int(cfg.train.max_episode_steps)
        terminated_mask = terminated > 0.5
        done_mask = jnp.logical_or(terminated_mask, timeout_mask)

        replay.add_batch(
            jax.device_get(obs),
            jax.device_get(action),
            jax.device_get(reward),
            jax.device_get(next_obs),
            jax.device_get(done_mask.astype(jnp.float32)),
        )

        obs = next_obs
        episode_return = episode_return + reward
        episode_progress = episode_progress + progress_delta
        prev_s = current_s

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

        if bool(jax.device_get(jnp.any(done_mask))):
            num_episodes = _log_finished_episodes(
                writer,
                env,
                done_mask,
                episode_return,
                episode_progress,
                episode_len,
                num_episodes,
                global_step=step_after_transition,
                first_progress_100_state=first_progress_100_state,
            )
            obs, prev_s = env.reset_done(done_mask, poses)
            episode_return = jnp.where(done_mask, 0.0, episode_return)
            episode_progress = jnp.where(done_mask, 0.0, episode_progress)
            episode_len = jnp.where(done_mask, jnp.zeros_like(episode_len), episode_len)

        global_step = step_after_transition
        pbar.update(num_envs)

        if writer is not None and last_metrics is not None and global_step % tb_log_every_steps == 0:
            metrics_host = jax.device_get(last_metrics)
            writer.add_scalar("sac/actor_loss", float(metrics_host["actor_loss"]), global_step)
            writer.add_scalar("sac/critic1_loss", float(metrics_host["critic1_loss"]), global_step)
            writer.add_scalar("sac/critic2_loss", float(metrics_host["critic2_loss"]), global_step)
            writer.add_scalar("sac/alpha_loss", float(metrics_host["alpha_loss"]), global_step)
            writer.add_scalar("sac/alpha", float(metrics_host["alpha"]), global_step)
            writer.add_scalar("sac/q_target_mean", float(metrics_host["q_target_mean"]), global_step)
            if tal_coef is not None:
                writer.add_scalar("reward/tal_coef", float(tal_coef), global_step)

        if last_metrics is not None and global_step % max(print_every_steps, num_envs) == 0:
            metrics_host = jax.device_get(last_metrics)
            pbar.set_postfix(
                actor=f"{float(metrics_host['actor_loss']):.3f}",
                critic=f"{float(metrics_host['critic1_loss']):.3f}",
                alpha=f"{float(metrics_host['alpha']):.4f}",
                tal=f"{float(tal_coef):.4f}" if tal_coef is not None else "off",
                ep=num_episodes,
            )

        if global_step % checkpoint_every_steps == 0:
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
            pbar.write(f"checkpoint saved: step={global_step}")

    pbar.close()


def train_td3(cfg, env, writer, ckpt_dir, rng, num_envs, parallel_mode):
    rng, rng_agent = jax.random.split(rng, 2)

    obs_shape = (cfg.env.obs_dim,)
    (
        actor_state,
        critic1_state,
        critic2_state,
        target_actor_params,
        target_critic1_params,
        target_critic2_params,
    ) = create_td3_states(
        rng_agent,
        obs_shape,
        cfg.env.action_dim,
        actor_lr=cfg.agent.actor_lr,
        critic_lr=cfg.agent.critic_lr,
    )

    restore_target = {
        "actor_state": actor_state,
        "critic1_state": critic1_state,
        "critic2_state": critic2_state,
        "target_actor_params": target_actor_params,
        "target_critic1_params": target_critic1_params,
        "target_critic2_params": target_critic2_params,
        "global_step": 0,
        "update_step": 0,
    }
    restored = maybe_restore_checkpoint(cfg, ckpt_dir, restore_target)
    actor_state = restored["actor_state"]
    critic1_state = restored["critic1_state"]
    critic2_state = restored["critic2_state"]
    target_actor_params = restored["target_actor_params"]
    target_critic1_params = restored["target_critic1_params"]
    target_critic2_params = restored["target_critic2_params"]
    global_step = int(restored["global_step"])
    update_step = int(restored.get("update_step", 0))

    replay = ReplayBuffer(
        capacity=cfg.agent.replay_size,
        obs_dim=cfg.env.obs_dim,
        action_dim=cfg.env.action_dim,
        seed=cfg.train.seed,
    )

    race_agents = int(getattr(env, "num_agents", num_envs))
    poses = make_start_poses(parallel_mode, num_envs, num_sim_agents=race_agents)
    obs = env.reset(poses)
    prev_s = env.get_current_progress_s()

    episode_return = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_progress = jnp.zeros((num_envs,), dtype=jnp.float32)
    episode_len = jnp.zeros((num_envs,), dtype=jnp.int32)
    num_episodes = 0
    last_metrics = None
    last_actor_loss = jnp.array(0.0, dtype=jnp.float32)
    first_progress_100_state = {
        "logged": False,
        "global_step": None,
        "episode": None,
        "progress_pct": None,
        "progress_m": None,
    }

    pbar = tqdm(total=cfg.train.total_timesteps, initial=global_step, desc="TD3 steps")
    print_every_steps = int(cfg.agent.get("print_every_steps", 1000))
    tb_log_every_steps = int(cfg.agent.get("tb_log_every_steps", print_every_steps))
    checkpoint_every_steps = int(cfg.agent.get("checkpoint_every_steps", 5000))
    policy_delay = int(cfg.agent.policy_delay)

    while global_step < cfg.train.total_timesteps:
        step_after_transition = global_step + num_envs
        tal_coef = _apply_tal_schedule(env, cfg, global_step)

        if global_step < cfg.agent.start_steps:
            rng, rng_random = jax.random.split(rng)
            action = jax.random.uniform(
                rng_random,
                shape=(num_envs, cfg.env.action_dim),
                minval=-1.0,
                maxval=1.0,
            )
        else:
            rng, rng_action = jax.random.split(rng)
            action = td3_act(
                actor_state,
                obs,
                rng_action,
                exploration_noise=cfg.agent.exploration_noise,
            )

        next_obs, reward, terminated, _ = env.step(action)
        current_s = env.get_current_progress_s()
        progress_delta = env.compute_progress_delta(current_s, prev_s)

        episode_len = episode_len + 1
        timeout_mask = episode_len >= int(cfg.train.max_episode_steps)
        terminated_mask = terminated > 0.5
        done_mask = jnp.logical_or(terminated_mask, timeout_mask)

        replay.add_batch(
            jax.device_get(obs),
            jax.device_get(action),
            jax.device_get(reward),
            jax.device_get(next_obs),
            jax.device_get(done_mask.astype(jnp.float32)),
        )

        obs = next_obs
        episode_return = episode_return + reward
        episode_progress = episode_progress + progress_delta
        prev_s = current_s

        if global_step >= cfg.agent.update_after and replay.can_sample(cfg.agent.batch_size):
            for _ in range(cfg.agent.updates_per_step):
                batch = replay.sample(cfg.agent.batch_size)
                rng, rng_update = jax.random.split(rng)

                critic1_state, critic2_state, critic_metrics = td3_update_critics(
                    actor_state,
                    critic1_state,
                    critic2_state,
                    target_actor_params,
                    target_critic1_params,
                    target_critic2_params,
                    batch["obs"],
                    batch["actions"],
                    batch["rewards"],
                    batch["next_obs"],
                    batch["terminated"],
                    rng_update,
                    gamma=cfg.agent.gamma,
                    target_policy_noise=cfg.agent.target_policy_noise,
                    target_noise_clip=cfg.agent.target_noise_clip,
                )

                update_step += 1
                actor_updated = (update_step % policy_delay) == 0
                if actor_updated:
                    (
                        actor_state,
                        target_actor_params,
                        target_critic1_params,
                        actor_loss,
                    ) = td3_update_actor_and_targets(
                        actor_state,
                        critic1_state,
                        target_actor_params,
                        target_critic1_params,
                        batch["obs"],
                        tau=cfg.agent.tau,
                    )
                    target_critic2_params = td3_soft_update_target_critic2(
                        target_critic2_params,
                        critic2_state.params,
                        tau=cfg.agent.tau,
                    )
                    last_actor_loss = actor_loss

                last_metrics = {
                    "actor_loss": last_actor_loss,
                    "critic1_loss": critic_metrics["critic1_loss"],
                    "critic2_loss": critic_metrics["critic2_loss"],
                    "q_target_mean": critic_metrics["q_target_mean"],
                    "actor_updated": jnp.array(1.0 if actor_updated else 0.0, dtype=jnp.float32),
                }

        if bool(jax.device_get(jnp.any(done_mask))):
            num_episodes = _log_finished_episodes(
                writer,
                env,
                done_mask,
                episode_return,
                episode_progress,
                episode_len,
                num_episodes,
                global_step=step_after_transition,
                first_progress_100_state=first_progress_100_state,
            )
            obs, prev_s = env.reset_done(done_mask, poses)
            episode_return = jnp.where(done_mask, 0.0, episode_return)
            episode_progress = jnp.where(done_mask, 0.0, episode_progress)
            episode_len = jnp.where(done_mask, jnp.zeros_like(episode_len), episode_len)

        global_step = step_after_transition
        pbar.update(num_envs)

        if writer is not None and last_metrics is not None and global_step % tb_log_every_steps == 0:
            metrics_host = jax.device_get(last_metrics)
            writer.add_scalar("td3/actor_loss", float(metrics_host["actor_loss"]), global_step)
            writer.add_scalar("td3/critic1_loss", float(metrics_host["critic1_loss"]), global_step)
            writer.add_scalar("td3/critic2_loss", float(metrics_host["critic2_loss"]), global_step)
            writer.add_scalar("td3/q_target_mean", float(metrics_host["q_target_mean"]), global_step)
            writer.add_scalar("td3/actor_updated", float(metrics_host["actor_updated"]), global_step)
            if tal_coef is not None:
                writer.add_scalar("reward/tal_coef", float(tal_coef), global_step)

        if last_metrics is not None and global_step % max(print_every_steps, num_envs) == 0:
            metrics_host = jax.device_get(last_metrics)
            pbar.set_postfix(
                actor=f"{float(metrics_host['actor_loss']):.3f}",
                critic=f"{float(metrics_host['critic1_loss']):.3f}",
                tal=f"{float(tal_coef):.4f}" if tal_coef is not None else "off",
                au=int(float(metrics_host["actor_updated"])),
                ep=num_episodes,
            )

        if global_step % checkpoint_every_steps == 0:
            checkpoints.save_checkpoint(
                ckpt_dir=str(ckpt_dir),
                target={
                    "actor_state": actor_state,
                    "critic1_state": critic1_state,
                    "critic2_state": critic2_state,
                    "target_actor_params": target_actor_params,
                    "target_critic1_params": target_critic1_params,
                    "target_critic2_params": target_critic2_params,
                    "global_step": global_step,
                    "update_step": update_step,
                },
                step=global_step,
                overwrite=True,
                keep=cfg.train.checkpoint.keep,
            )
            pbar.write(f"checkpoint saved: step={global_step}")

    pbar.close()


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    if cfg.train.get("quiet_absl", False):
        logging.getLogger("absl").setLevel(logging.WARNING)

    print("=== JAX F1TENTH RL Training Start ===")
    print(f"Algorithm: {cfg.agent.name}")
    print(f"JAX backend: {jax.default_backend()}")

    project_root = Path(get_original_cwd())

    map_path, map_ext, waypoints_path = resolve_env_assets(cfg.env, project_root)
    print(f"Map: {map_path}")
    print(f"Waypoints: {waypoints_path}")
    vehicle_params, vehicle_source = resolve_vehicle_params(cfg, project_root)
    print(f"Vehicle Params: {vehicle_source}")

    num_envs = get_train_env_count(cfg)
    scan_beams, scan_fov = resolve_lidar_sim_params(cfg.env)
    max_lidar_range = float(cfg.env.get("max_lidar_range", 30.0))
    rng = jax.random.PRNGKey(cfg.train.seed)

    env, parallel_mode = build_env(
        cfg,
        map_path=map_path,
        map_ext=map_ext,
        waypoints_path=waypoints_path,
        num_envs=num_envs,
        vehicle_params=vehicle_params,
        scan_beams=scan_beams,
        scan_fov=scan_fov,
        max_lidar_range=max_lidar_range,
    )
    print(f"Parallel Mode: {parallel_mode} | Vector Size: {env.num_envs}")
    if parallel_mode == "race":
        print(
            "Race Mode Details: "
            f"control_mode={env.control_mode}, "
            f"num_agents={env.num_agents}, "
            f"learned_agents={env.num_envs}, "
            f"ego_idx={env.ego_idx}"
        )
    print(f"LiDAR: beams={scan_beams}, fov={scan_fov:.6f} rad, max_range={max_lidar_range:.3f} m")

    writer = maybe_make_writer(cfg, project_root)

    ckpt_dir = Path(cfg.train.checkpoint.dir)
    if not ckpt_dir.is_absolute():
        ckpt_dir = project_root / ckpt_dir
    ckpt_dir = maybe_autofork_checkpoint_dir(cfg, ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.agent.name == "ppo":
        train_ppo(cfg, env, writer, ckpt_dir, rng, num_envs=env.num_envs, parallel_mode=parallel_mode)
    elif cfg.agent.name == "sac":
        train_sac(cfg, env, writer, ckpt_dir, rng, num_envs=env.num_envs, parallel_mode=parallel_mode)
    elif cfg.agent.name == "td3":
        train_td3(cfg, env, writer, ckpt_dir, rng, num_envs=env.num_envs, parallel_mode=parallel_mode)
    else:
        raise ValueError(f"Unsupported agent: {cfg.agent.name}")

    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()

from pathlib import Path

import cv2
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


class TrajectoryVideoRecorder:
    def __init__(self, output_path: Path, centerline_xy, fps=20, size=900, margin=20.0):
        self.output_path = output_path
        self.fps = int(fps)
        self.size = int(size)
        self.margin = float(margin)

        self.centerline_xy = np.asarray(centerline_xy, dtype=np.float32)
        self.min_xy = self.centerline_xy.min(axis=0) - self.margin
        self.max_xy = self.centerline_xy.max(axis=0) + self.margin
        span = np.maximum(self.max_xy - self.min_xy, 1e-3)
        self.scale = (self.size - 20) / span

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (self.size, self.size))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.output_path}")

        self.base = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        self._draw_centerline()
        self.traj = []

    def _world_to_pixel(self, x, y):
        px = int((x - self.min_xy[0]) * self.scale[0]) + 10
        py = int((y - self.min_xy[1]) * self.scale[1]) + 10
        py = self.size - py
        return px, py

    def _draw_centerline(self):
        pts = np.array([self._world_to_pixel(x, y) for x, y in self.centerline_xy], dtype=np.int32)
        cv2.polylines(self.base, [pts], isClosed=True, color=(160, 160, 160), thickness=1, lineType=cv2.LINE_AA)

    def add_frame(self, x, y, episode_idx, collided=False):
        frame = self.base.copy()
        self.traj.append((x, y))
        if len(self.traj) > 1:
            traj_pts = np.array([self._world_to_pixel(px, py) for px, py in self.traj], dtype=np.int32)
            cv2.polylines(frame, [traj_pts], isClosed=False, color=(30, 144, 255), thickness=2, lineType=cv2.LINE_AA)

        car_pt = self._world_to_pixel(x, y)
        car_color = (0, 0, 255) if collided else (0, 180, 0)
        cv2.circle(frame, car_pt, 5, car_color, -1, lineType=cv2.LINE_AA)
        cv2.putText(frame, f"Episode {episode_idx}", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
        self.writer.write(frame)

    def close(self):
        self.writer.release()


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
    progress_pcts = []
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
        video_recorder = None

        if cfg.eval.video.enabled:
            output_dir = Path(cfg.eval.video.output_dir)
            if not output_dir.is_absolute():
                output_dir = project_root / output_dir
            output_dir.mkdir(parents=True, exist_ok=True)
            video_path = output_dir / f"{cfg.eval.video.filename_prefix}_ep{ep + 1:03d}.mp4"
            video_recorder = TrajectoryVideoRecorder(
                output_path=video_path,
                centerline_xy=jax.device_get(env.waypoints_xy),
                fps=cfg.eval.video.fps,
                size=cfg.eval.video.size,
                margin=cfg.eval.video.margin_m,
            )

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

            if video_recorder is not None:
                ego_x = float(jax.device_get(sim.sim_state["state"][0, 0]))
                ego_y = float(jax.device_get(sim.sim_state["state"][0, 1]))
                video_recorder.add_frame(ego_x, ego_y, ep + 1, collided=False)

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
                if video_recorder is not None:
                    ego_x = float(jax.device_get(sim.sim_state["state"][0, 0]))
                    ego_y = float(jax.device_get(sim.sim_state["state"][0, 1]))
                    video_recorder.add_frame(ego_x, ego_y, ep + 1, collided=collided)
                break

        if video_recorder is not None:
            video_recorder.close()

        returns.append(episode_return)
        lengths.append(episode_steps)
        collision_rates.append(1.0 if collided else 0.0)
        progresses.append(episode_progress)
        progress_pct = (episode_progress / max(env.track_length, 1e-6)) * 100.0
        progress_pcts.append(progress_pct)
        avg_speed = speed_sum / max(speed_count, 1)
        avg_speeds.append(avg_speed)
        completion_flags.append(1.0 if completed else 0.0)
        print(
            f"Episode {ep + 1}/{cfg.eval.episodes} | Return: {episode_return:.3f} "
            f"| Length: {episode_steps} | Progress(m): {episode_progress:.3f} "
            f"| Progress(%): {progress_pct:.2f} "
            f"| AvgSpeed(m/s): {avg_speed:.3f} | Completed: {completed} | Collided: {collided}"
        )

    print("=== Eval Summary ===")
    print(f"Average Return: {np.mean(returns):.3f}")
    print(f"Average Length: {np.mean(lengths):.2f}")
    print(f"Average Progress (m): {np.mean(progresses):.3f}")
    print(f"Average Progress (%): {np.mean(progress_pcts):.2f}")
    print(f"Average Speed (m/s): {np.mean(avg_speeds):.3f}")
    print(f"Completion Rate: {np.mean(completion_flags):.3f}")
    print(f"Collision Rate: {np.mean(collision_rates):.3f}")


if __name__ == "__main__":
    main()

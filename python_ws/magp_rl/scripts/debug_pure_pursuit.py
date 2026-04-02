#!/usr/bin/env python3
"""Pure Pursuit teacher debug and parameter sweep utility.

Outputs:
- waypoint speed profile image(s) with color-by-speed
- optional rollout video(s) with color-by-speed trajectory
- CSV summary for parameter comparison
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import math
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from f110_jax.simulator import F110JaxSimulator, Integrator
from src.envs.f110_independent_vec import F110IndependentVecEnv
from src.utils.common import resolve_lidar_sim_params
from src.utils.env_assets import resolve_env_assets
from src.utils.pure_pursuit import PurePursuitTeacher
from src.utils.vehicle import resolve_vehicle_params


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _speed_to_color(speed_mps: float, speed_min: float, speed_max: float) -> Tuple[int, int, int]:
    speed_max = max(float(speed_max), float(speed_min) + 1e-6)
    norm = (float(speed_mps) - float(speed_min)) / (speed_max - speed_min)
    norm = float(np.clip(norm, 0.0, 1.0))
    idx = int(norm * 255.0)
    bgr = cv2.applyColorMap(np.array([[idx]], dtype=np.uint8), cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


class CanvasProjector:
    def __init__(self, points_xy: np.ndarray, size: int = 900, margin_m: float = 20.0):
        self.size = int(size)
        self.margin_m = float(margin_m)
        pts = np.asarray(points_xy, dtype=np.float32)
        self.min_xy = pts.min(axis=0) - self.margin_m
        self.max_xy = pts.max(axis=0) + self.margin_m
        span = np.maximum(self.max_xy - self.min_xy, 1e-3)
        self.scale = (self.size - 20) / span

    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        px = int((x - self.min_xy[0]) * self.scale[0]) + 10
        py = int((y - self.min_xy[1]) * self.scale[1]) + 10
        py = self.size - py
        return px, py

    def make_base(self, centerline_xy: np.ndarray) -> np.ndarray:
        frame = np.full((self.size, self.size, 3), 255, dtype=np.uint8)
        centerline_xy = np.asarray(centerline_xy, dtype=np.float32)
        pts = np.array([self.world_to_pixel(x, y) for x, y in centerline_xy], dtype=np.int32)
        cv2.polylines(
            frame,
            [pts],
            isClosed=True,
            color=(180, 180, 180),
            thickness=1,
            lineType=cv2.LINE_AA,
        )
        return frame


def save_speed_profile_image(
    output_path: Path,
    waypoints_xy: np.ndarray,
    waypoints_speed: np.ndarray,
    speed_min: float,
    speed_max: float,
    size: int = 1000,
    margin_m: float = 20.0,
) -> None:
    projector = CanvasProjector(waypoints_xy, size=size, margin_m=margin_m)
    frame = projector.make_base(waypoints_xy)
    for i in range(1, len(waypoints_xy)):
        p0 = projector.world_to_pixel(float(waypoints_xy[i - 1, 0]), float(waypoints_xy[i - 1, 1]))
        p1 = projector.world_to_pixel(float(waypoints_xy[i, 0]), float(waypoints_xy[i, 1]))
        color = _speed_to_color(float(waypoints_speed[i]), speed_min, speed_max)
        cv2.line(frame, p0, p1, color, 3, lineType=cv2.LINE_AA)

    # Close segment for closed tracks.
    p0 = projector.world_to_pixel(float(waypoints_xy[-1, 0]), float(waypoints_xy[-1, 1]))
    p1 = projector.world_to_pixel(float(waypoints_xy[0, 0]), float(waypoints_xy[0, 1]))
    color = _speed_to_color(float(waypoints_speed[0]), speed_min, speed_max)
    cv2.line(frame, p0, p1, color, 3, lineType=cv2.LINE_AA)

    start_pt = projector.world_to_pixel(float(waypoints_xy[0, 0]), float(waypoints_xy[0, 1]))
    cv2.circle(frame, start_pt, 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Waypoint speed profile [{speed_min:.2f}, {speed_max:.2f}] m/s",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (25, 25, 25),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(output_path), frame)


def _build_env_with_vehicle(
    env_cfg: DictConfig,
    map_path: str,
    map_ext: str,
    waypoints_path: str,
    seed: int,
    vehicle_params,
):
    num_beams, scan_fov = resolve_lidar_sim_params(env_cfg)
    max_lidar_range = float(env_cfg.get("max_lidar_range", 30.0))
    sim = F110JaxSimulator(
        map_path=map_path,
        map_ext=map_ext,
        num_agents=1,
        params=vehicle_params,
        seed=seed,
        integrator=Integrator.RK4,
        num_beams=num_beams,
        fov=scan_fov,
        max_range=max_lidar_range,
    )
    env = F110IndependentVecEnv(
        sim,
        env_cfg,
        num_envs=1,
        waypoints_path=waypoints_path,
        seed=seed,
    )
    return env


def _make_start_pose_from_waypoints(waypoints_xy: np.ndarray) -> np.ndarray:
    if len(waypoints_xy) < 2:
        raise ValueError("Need at least two waypoints to infer start heading.")
    p0 = waypoints_xy[0]
    p1 = waypoints_xy[1]
    theta = math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0]))
    return np.array([[float(p0[0]), float(p0[1]), float(theta)]], dtype=np.float32)


def _teacher_action_normalized(env: F110IndependentVecEnv):
    state = env.sim_state["state"][:, 0, :]
    action = env.pp_teacher.act(state[:, 0], state[:, 1], state[:, 4], state[:, 3])
    action = action.at[:, 0].set(jnp.clip(action[:, 0], env.min_steer, env.max_steer))
    action = action.at[:, 1].set(jnp.clip(action[:, 1], env.min_speed, env.max_speed))
    return env._to_normalized_action(action)


def run_pure_pursuit_episode(
    env: F110IndependentVecEnv,
    max_steps: int,
    speed_min: float,
    speed_max: float,
    video_path: Path | None = None,
    video_fps: int = 20,
    video_size: int = 900,
    video_margin_m: float = 20.0,
    done_check_interval: int = 25,
    progress_eval_interval: int = 5,
):
    waypoints_xy = np.asarray(jax.device_get(env.waypoints_xy), dtype=np.float32)
    pose = _make_start_pose_from_waypoints(waypoints_xy)
    env.reset(pose)
    prev_s = env.get_current_progress_s()

    # Fast path (no video): minimize host-device sync by keeping most accumulators on device.
    if video_path is None:
        done_seen = jnp.array(False, dtype=jnp.bool_)
        completed = jnp.array(False, dtype=jnp.bool_)
        collided = jnp.array(False, dtype=jnp.bool_)
        episode_progress = jnp.array(0.0, dtype=jnp.float32)
        speed_sum = jnp.array(0.0, dtype=jnp.float32)
        speed_count = jnp.array(0, dtype=jnp.int32)
        done_step = jnp.array(int(max_steps), dtype=jnp.int32)

        steps = int(max_steps)
        check_every = max(int(done_check_interval), 1)
        progress_every = max(int(progress_eval_interval), 1)
        for step in range(1, int(max_steps) + 1):
            action = _teacher_action_normalized(env)
            _, _, done_arr, info = env.step(action)

            state = env.sim_state["state"][:, 0, :]
            if step % progress_every == 0:
                current_s = env._project_to_centerline_s(state[:, 0], state[:, 1])
                progress_delta = env.compute_progress_delta(current_s, prev_s)
                prev_s = current_s
            else:
                progress_delta = jnp.zeros_like(prev_s, dtype=jnp.float32)

            speed_now = jnp.abs(state[0, 3])
            checkpoint_done = info.get("checkpoint_done", jnp.zeros((env.num_envs,), dtype=jnp.float32))
            completed_now = jnp.any(checkpoint_done > 0.5)
            collision_now = jnp.any(env.get_collisions() > 0.0)
            done_now = jnp.any(done_arr > 0.5) | completed_now

            active = ~done_seen
            episode_progress = episode_progress + jnp.where(active, progress_delta[0], 0.0)
            speed_sum = speed_sum + jnp.where(active, speed_now, 0.0)
            speed_count = speed_count + jnp.where(active, jnp.int32(1), jnp.int32(0))
            done_step = jnp.where((~done_seen) & done_now, jnp.int32(step), done_step)

            completed = completed | completed_now
            collided = collided | collision_now
            done_seen = done_seen | done_now

            if step % check_every == 0:
                if bool(jax.device_get(done_seen)):
                    steps = step
                    break

        (
            done_seen_h,
            completed_h,
            collided_h,
            episode_progress_h,
            speed_sum_h,
            speed_count_h,
            done_step_h,
        ) = jax.device_get(
            (
                done_seen,
                completed,
                collided,
                episode_progress,
                speed_sum,
                speed_count,
                done_step,
            )
        )

        if bool(done_seen_h):
            steps = int(done_step_h)
        progress_pct = (float(episode_progress_h) / max(float(env.track_length), 1e-6)) * 100.0
        avg_speed = float(speed_sum_h) / max(int(speed_count_h), 1)
        lap_time_sec = (steps * float(env.sim.time_step)) if bool(completed_h) else float("inf")

        return {
            "steps": int(steps),
            "completed": bool(completed_h),
            "collided": bool(collided_h),
            "progress_m": float(episode_progress_h),
            "progress_pct": float(progress_pct),
            "avg_speed_mps": float(avg_speed),
            "lap_time_sec": float(lap_time_sec),
        }

    done = False
    completed = False
    collided = False
    episode_progress = 0.0
    speed_sum = 0.0
    speed_count = 0
    positions = []
    speeds = []

    writer = None
    projector = None
    base = None
    if video_path is not None:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, int(video_fps), (int(video_size), int(video_size)))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")
        projector = CanvasProjector(waypoints_xy, size=video_size, margin_m=video_margin_m)
        base = projector.make_base(waypoints_xy)

    steps = 0
    while steps < int(max_steps):
        action = _teacher_action_normalized(env)
        _, _, done_arr, info = env.step(action)
        steps += 1

        current_s = env.get_current_progress_s()
        progress_delta = env.compute_progress_delta(current_s, prev_s)
        episode_progress += float(jax.device_get(progress_delta[0]))
        prev_s = current_s

        pos_x, pos_y = env.get_positions()
        speed_now = env.get_speeds()
        x = float(jax.device_get(pos_x[0]))
        y = float(jax.device_get(pos_y[0]))
        v = float(jax.device_get(speed_now[0]))
        positions.append((x, y))
        speeds.append(v)
        speed_sum += v
        speed_count += 1

        checkpoint_done = info.get("checkpoint_done", jnp.zeros((env.num_envs,), dtype=jnp.float32))
        completed = bool(jax.device_get(jnp.any(checkpoint_done > 0.5)))
        done = bool(jax.device_get(jnp.any(done_arr > 0.5))) or completed
        if done:
            collisions = env.get_collisions()
            collided = bool(jax.device_get(jnp.any(collisions > 0.0)))

        if writer is not None:
            frame = base.copy()
            if len(positions) > 1:
                for i in range(1, len(positions)):
                    p0 = projector.world_to_pixel(positions[i - 1][0], positions[i - 1][1])
                    p1 = projector.world_to_pixel(positions[i][0], positions[i][1])
                    col = _speed_to_color(speeds[i], speed_min, speed_max)
                    cv2.line(frame, p0, p1, col, 2, lineType=cv2.LINE_AA)
            car_pt = projector.world_to_pixel(x, y)
            cv2.circle(
                frame,
                car_pt,
                5,
                (0, 0, 255) if collided else (0, 180, 0),
                -1,
                lineType=cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"step={steps} speed={v:.2f} m/s",
                (14, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (25, 25, 25),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)

        if done:
            break

    if writer is not None:
        writer.release()

    progress_pct = (episode_progress / max(float(env.track_length), 1e-6)) * 100.0
    avg_speed = speed_sum / max(speed_count, 1)
    lap_time_sec = (steps * float(env.sim.time_step)) if completed else float("inf")

    return {
        "steps": int(steps),
        "completed": bool(completed),
        "collided": bool(collided),
        "progress_m": float(episode_progress),
        "progress_pct": float(progress_pct),
        "avg_speed_mps": float(avg_speed),
        "lap_time_sec": float(lap_time_sec),
    }


def main():
    parser = argparse.ArgumentParser(description="Debug Pure Pursuit settings before RL/TAL training.")
    parser.add_argument("--track-name", type=str, default="Austin")
    parser.add_argument("--line-type", type=str, default="centerline", choices=["centerline", "raceline"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lookahead-list", type=str, default="0.5")
    parser.add_argument("--vgain-list", type=str, default="1.0")
    parser.add_argument("--fixed-lookahead", type=float, default=0.5)
    parser.add_argument("--fixed-vgain", type=float, default=1.0)
    parser.add_argument(
        "--sweep-lookahead-vgain",
        action="store_true",
        help="Enable lookahead/vgain sweep. Default behavior keeps them fixed.",
    )
    parser.add_argument("--speed-mode-list", type=str, default="file_or_curvature")
    parser.add_argument("--lat-accel-list", type=str, default="2.0,2.5,3.0,3.5")
    parser.add_argument("--smoothing-list", type=str, default="9")
    parser.add_argument("--min-speed", type=float, default=0.5)
    parser.add_argument("--max-speed", type=float, default=5.0)
    parser.add_argument("--save-video-top-k", type=int, default=3)
    parser.add_argument(
        "--save-rollout-videos",
        action="store_true",
        help="Save rollout videos for top-k settings. Default: disabled for faster sweep.",
    )
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-size", type=int, default=900)
    parser.add_argument("--video-margin-m", type=float, default=20.0)
    parser.add_argument(
        "--done-check-interval",
        type=int,
        default=25,
        help="In no-video mode, check done/collision on host every N steps to reduce sync overhead.",
    )
    parser.add_argument(
        "--progress-eval-interval",
        type=int,
        default=5,
        help="In no-video mode, evaluate centerline progress every N steps (larger is faster).",
    )
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    lookahead_list = _parse_float_list(args.lookahead_list)
    vgain_list = _parse_float_list(args.vgain_list)
    speed_mode_list = [x.strip() for x in args.speed_mode_list.split(",") if x.strip()]
    lat_accel_list = _parse_float_list(args.lat_accel_list)
    smoothing_list = _parse_int_list(args.smoothing_list)

    if not speed_mode_list or not lat_accel_list or not smoothing_list:
        raise ValueError("speed-mode-list / lat-accel-list / smoothing-list must not be empty.")
    if args.sweep_lookahead_vgain:
        if not lookahead_list or not vgain_list:
            raise ValueError("lookahead-list and vgain-list must not be empty.")
        teacher_combos = list(itertools.product(lookahead_list, vgain_list))
    else:
        teacher_combos = [(float(args.fixed_lookahead), float(args.fixed_vgain))]

    run_ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else (PROJECT_ROOT / "records" / "pure_pursuit_debug" / f"{args.track_name}_{args.line_type}_{run_ts}")
    )
    if not out_dir.is_absolute():
        out_dir = (PROJECT_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(PROJECT_ROOT / "config" / "eval.yaml")
    cfg.env.parallel.mode = "independent"
    cfg.env.track.name = args.track_name
    cfg.env.track.line_type = args.line_type
    cfg.env.reward.tal.enabled = True
    cfg.env.reward.tal.coef = 0.0  # TAL reward is irrelevant for pure pursuit rollout itself.
    cfg.env.reward.tal.speed_profile.min_speed_mps = float(args.min_speed)
    cfg.env.reward.tal.speed_profile.max_speed_mps = float(args.max_speed)

    map_path, map_ext, waypoints_path = resolve_env_assets(cfg.env, PROJECT_ROOT)
    vehicle_params, vehicle_source = resolve_vehicle_params(cfg, PROJECT_ROOT)

    print("=== Pure Pursuit Debug Sweep ===")
    print(f"Track: {args.track_name} ({args.line_type})")
    print(f"Map: {map_path}")
    print(f"Waypoints: {waypoints_path}")
    print(f"Vehicle Params: {vehicle_source}")
    print(f"Output Dir: {out_dir}")
    if args.sweep_lookahead_vgain:
        print(f"Lookahead/Vgain sweep: ON ({len(teacher_combos)} combos)")
    else:
        print(
            "Lookahead/Vgain sweep: OFF "
            f"(lookahead={teacher_combos[0][0]:.3f}, vgain={teacher_combos[0][1]:.3f})"
        )
    if args.save_rollout_videos:
        print(f"Rollout video save: ON (top-k={max(0, int(args.save_video_top_k))})")
    else:
        print(
            "Rollout video save: OFF "
            f"(done check interval={max(int(args.done_check_interval), 1)}, "
            f"progress eval interval={max(int(args.progress_eval_interval), 1)})"
        )

    records = []
    combo_id = 0

    speed_profile_combos = list(itertools.product(speed_mode_list, lat_accel_list, smoothing_list))
    for speed_mode, lat_accel, smoothing in speed_profile_combos:
        cfg.env.reward.tal.speed_profile.mode = str(speed_mode)
        cfg.env.reward.tal.speed_profile.max_lateral_accel_mps2 = float(lat_accel)
        cfg.env.reward.tal.speed_profile.smoothing_window = int(smoothing)
        cfg.env.reward.tal.lookahead_distance = float(teacher_combos[0][0])
        cfg.env.reward.tal.vgain = float(teacher_combos[0][1])

        env = _build_env_with_vehicle(
            cfg.env,
            map_path=map_path,
            map_ext=map_ext,
            waypoints_path=waypoints_path,
            seed=args.seed,
            vehicle_params=vehicle_params,
        )

        wp_xy = np.asarray(jax.device_get(env.waypoints_xy), dtype=np.float32)
        wp_speed = np.asarray(jax.device_get(env.waypoints_speed), dtype=np.float32)
        profile_tag = f"mode-{speed_mode}_alat-{lat_accel:.2f}_sm-{int(smoothing)}"
        save_speed_profile_image(
            output_path=out_dir / f"waypoint_speed_{profile_tag}.png",
            waypoints_xy=wp_xy,
            waypoints_speed=wp_speed,
            speed_min=float(args.min_speed),
            speed_max=float(args.max_speed),
            size=max(int(args.video_size), 900),
            margin_m=float(args.video_margin_m),
        )

        wheelbase = float(cfg.env.reward.tal.wheelbase)
        for lookahead, vgain in teacher_combos:
            combo_id += 1
            env.pp_teacher = PurePursuitTeacher(
                env.waypoints_xy,
                env.waypoints_s,
                env.waypoints_speed,
                lookahead_distance=float(lookahead),
                lookahead_gain=float(cfg.env.reward.tal.get("lookahead_gain", 0.3)),
                wheelbase=wheelbase,
                vgain=float(vgain),
            )

            ep_results = []
            for ep in range(int(args.episodes)):
                result = run_pure_pursuit_episode(
                    env,
                    max_steps=args.max_steps,
                    speed_min=float(args.min_speed),
                    speed_max=float(args.max_speed),
                    video_path=None,
                    done_check_interval=args.done_check_interval,
                    progress_eval_interval=args.progress_eval_interval,
                )
                ep_results.append(result)

            completed_rate = float(np.mean([1.0 if r["completed"] else 0.0 for r in ep_results]))
            collision_rate = float(np.mean([1.0 if r["collided"] else 0.0 for r in ep_results]))
            mean_progress_pct = float(np.mean([r["progress_pct"] for r in ep_results]))
            mean_progress_m = float(np.mean([r["progress_m"] for r in ep_results]))
            mean_avg_speed = float(np.mean([r["avg_speed_mps"] for r in ep_results]))
            mean_steps = float(np.mean([r["steps"] for r in ep_results]))
            lap_times = [r["lap_time_sec"] for r in ep_results if np.isfinite(r["lap_time_sec"])]
            mean_lap_time = float(np.mean(lap_times)) if lap_times else float("inf")

            records.append(
                {
                    "combo_id": combo_id,
                    "speed_mode": speed_mode,
                    "max_lateral_accel_mps2": float(lat_accel),
                    "smoothing_window": int(smoothing),
                    "lookahead_distance": float(lookahead),
                    "vgain": float(vgain),
                    "completed_rate": completed_rate,
                    "collision_rate": collision_rate,
                    "progress_pct": mean_progress_pct,
                    "progress_m": mean_progress_m,
                    "avg_speed_mps": mean_avg_speed,
                    "steps": mean_steps,
                    "lap_time_sec": mean_lap_time,
                    "profile_min_speed_mps": float(np.min(wp_speed)),
                    "profile_max_speed_mps": float(np.max(wp_speed)),
                }
            )

            lap_str = f"{mean_lap_time:.2f}" if np.isfinite(mean_lap_time) else "-"
            print(
                f"[{combo_id:03d}] mode={speed_mode:<17} alat={lat_accel:.2f} sm={int(smoothing):2d} "
                f"lookahead={lookahead:.2f} vgain={vgain:.2f} | "
                f"complete={completed_rate:.2f} collision={collision_rate:.2f} "
                f"progress={mean_progress_pct:.1f}% avg_v={mean_avg_speed:.2f} lap={lap_str}"
            )

    # Ranking policy:
    # 1) Higher completion rate
    # 2) Lower collision rate
    # 3) Higher progress
    # 4) Higher average speed
    # 5) Lower lap time
    records_sorted = sorted(
        records,
        key=lambda r: (
            -r["completed_rate"],
            r["collision_rate"],
            -r["progress_pct"],
            -r["avg_speed_mps"],
            r["lap_time_sec"],
        ),
    )

    csv_path = out_dir / "pure_pursuit_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(records_sorted)

    print("")
    print("=== Top Results ===")
    for rank, rec in enumerate(records_sorted[: max(int(args.save_video_top_k), 1)], start=1):
        lap_str = f"{rec['lap_time_sec']:.2f}s" if np.isfinite(rec["lap_time_sec"]) else "-"
        print(
            f"#{rank} combo={rec['combo_id']} complete={rec['completed_rate']:.2f} "
            f"collision={rec['collision_rate']:.2f} progress={rec['progress_pct']:.1f}% "
            f"avg_v={rec['avg_speed_mps']:.2f} lap={lap_str} | "
            f"mode={rec['speed_mode']} alat={rec['max_lateral_accel_mps2']:.2f} "
            f"sm={rec['smoothing_window']} lookahead={rec['lookahead_distance']:.2f} "
            f"vgain={rec['vgain']:.2f}"
        )

    # Save rollout videos for top-k settings.
    top_k = max(0, int(args.save_video_top_k))
    if args.save_rollout_videos and top_k > 0:
        print("")
        print(f"Saving rollout videos for top {top_k} settings...")
        for rank, rec in enumerate(records_sorted[:top_k], start=1):
            cfg.env.reward.tal.speed_profile.mode = rec["speed_mode"]
            cfg.env.reward.tal.speed_profile.max_lateral_accel_mps2 = rec["max_lateral_accel_mps2"]
            cfg.env.reward.tal.speed_profile.smoothing_window = int(rec["smoothing_window"])
            cfg.env.reward.tal.lookahead_distance = float(rec["lookahead_distance"])
            cfg.env.reward.tal.vgain = float(rec["vgain"])

            env = _build_env_with_vehicle(
                cfg.env,
                map_path=map_path,
                map_ext=map_ext,
                waypoints_path=waypoints_path,
                seed=args.seed,
                vehicle_params=vehicle_params,
            )
            env.pp_teacher = PurePursuitTeacher(
                env.waypoints_xy,
                env.waypoints_s,
                env.waypoints_speed,
                lookahead_distance=float(rec["lookahead_distance"]),
                lookahead_gain=float(cfg.env.reward.tal.get("lookahead_gain", 0.3)),
                wheelbase=float(cfg.env.reward.tal.wheelbase),
                vgain=float(rec["vgain"]),
            )
            video_name = (
                f"rank{rank:02d}_combo{int(rec['combo_id']):03d}"
                f"_mode-{rec['speed_mode']}"
                f"_alat-{rec['max_lateral_accel_mps2']:.2f}"
                f"_la-{rec['lookahead_distance']:.2f}"
                f"_vg-{rec['vgain']:.2f}.mp4"
            )
            run_pure_pursuit_episode(
                env,
                max_steps=args.max_steps,
                speed_min=float(args.min_speed),
                speed_max=float(args.max_speed),
                video_path=out_dir / video_name,
                video_fps=args.video_fps,
                video_size=args.video_size,
                video_margin_m=args.video_margin_m,
            )
    elif top_k > 0:
        print("")
        print("Skipping rollout video export (enable with --save-rollout-videos).")

    print("")
    print(f"Sweep CSV: {csv_path}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()

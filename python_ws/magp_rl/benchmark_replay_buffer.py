#!/usr/bin/env python3
"""
Benchmark ReplayBuffer.add_batch performance.

This script compares:
1) Legacy loop-based add_batch (embedded baseline)
2) Current src.utils.replay_buffer.ReplayBuffer implementation

Run from python_ws/magp_rl:
  PYTHONPATH=. python3 benchmark_replay_buffer.py
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from src.utils.replay_buffer import ReplayBuffer as CurrentReplayBuffer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Failed to import src.utils.replay_buffer.ReplayBuffer. "
        "Run from python_ws/magp_rl with PYTHONPATH=. and activated env."
    ) from exc


class LoopReplayBuffer:
    """Baseline: legacy loop implementation before vectorization."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, seed: int = 0):
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

        batch_n = int(obs_np.shape[0])
        for i in range(batch_n):
            self.obs[self.ptr] = obs_np[i]
            self.actions[self.ptr] = actions_np[i]
            self.rewards[self.ptr] = rewards_np[i]
            self.next_obs[self.ptr] = next_obs_np[i]
            self.terminated[self.ptr] = term_np[i]
            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)


@dataclass
class Row:
    num_envs: int
    loop_us_per_call: float
    current_us_per_call: float
    loop_m_samples_per_s: float
    current_m_samples_per_s: float
    speedup_x: float


def parse_args():
    parser = argparse.ArgumentParser(description="ReplayBuffer.add_batch benchmark")
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[1, 8, 32, 64, 128, 256, 512],
        help="List of num_envs values to benchmark",
    )
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--obs-dim", type=int, default=320)
    parser.add_argument("--action-dim", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-equivalence-check",
        action="store_true",
        help="Skip correctness parity check between baseline and current buffer",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    return parser.parse_args()


def _random_batch(rng: np.random.Generator, num_envs: int, obs_dim: int, action_dim: int):
    obs = rng.standard_normal((num_envs, obs_dim), dtype=np.float32)
    actions = rng.standard_normal((num_envs, action_dim), dtype=np.float32)
    rewards = rng.standard_normal((num_envs,), dtype=np.float32)
    next_obs = rng.standard_normal((num_envs, obs_dim), dtype=np.float32)
    terminated = rng.integers(0, 2, size=(num_envs,), dtype=np.int8).astype(np.float32)
    return obs, actions, rewards, next_obs, terminated


def check_equivalence(seed: int):
    rng = np.random.default_rng(seed)
    capacities = [1, 2, 7, 31, 128]
    obs_dims = [5, 11]
    action_dims = [1, 2, 4]

    for capacity in capacities:
        for obs_dim in obs_dims:
            for action_dim in action_dims:
                loop_rb = LoopReplayBuffer(capacity, obs_dim, action_dim, seed=seed)
                cur_rb = CurrentReplayBuffer(capacity, obs_dim, action_dim, seed=seed)

                for _ in range(300):
                    n = int(rng.integers(0, capacity * 3 + 5))
                    obs, actions, rewards, next_obs, terminated = _random_batch(
                        rng, n, obs_dim, action_dim
                    )
                    loop_rb.add_batch(obs, actions, rewards, next_obs, terminated)
                    cur_rb.add_batch(obs, actions, rewards, next_obs, terminated)

                np.testing.assert_array_equal(loop_rb.obs, cur_rb.obs)
                np.testing.assert_array_equal(loop_rb.actions, cur_rb.actions)
                np.testing.assert_array_equal(loop_rb.rewards, cur_rb.rewards)
                np.testing.assert_array_equal(loop_rb.next_obs, cur_rb.next_obs)
                np.testing.assert_array_equal(loop_rb.terminated, cur_rb.terminated)
                assert loop_rb.ptr == cur_rb.ptr
                assert loop_rb.size == cur_rb.size


def benchmark_one(
    buffer_cls,
    obs,
    actions,
    rewards,
    next_obs,
    terminated,
    capacity: int,
    obs_dim: int,
    action_dim: int,
    warmup: int,
    iters: int,
    repeats: int,
) -> float:
    elapsed_sec = []
    for rep in range(repeats):
        rb = buffer_cls(capacity=capacity, obs_dim=obs_dim, action_dim=action_dim, seed=rep)
        for _ in range(warmup):
            rb.add_batch(obs, actions, rewards, next_obs, terminated)

        start = time.perf_counter()
        for _ in range(iters):
            rb.add_batch(obs, actions, rewards, next_obs, terminated)
        elapsed_sec.append(time.perf_counter() - start)

    return float(np.median(np.asarray(elapsed_sec, dtype=np.float64)))


def run_case(args, num_envs: int) -> Row:
    rng = np.random.default_rng(args.seed + num_envs)
    obs, actions, rewards, next_obs, terminated = _random_batch(
        rng, num_envs, args.obs_dim, args.action_dim
    )

    loop_sec = benchmark_one(
        LoopReplayBuffer,
        obs,
        actions,
        rewards,
        next_obs,
        terminated,
        args.capacity,
        args.obs_dim,
        args.action_dim,
        args.warmup,
        args.iters,
        args.repeats,
    )
    current_sec = benchmark_one(
        CurrentReplayBuffer,
        obs,
        actions,
        rewards,
        next_obs,
        terminated,
        args.capacity,
        args.obs_dim,
        args.action_dim,
        args.warmup,
        args.iters,
        args.repeats,
    )

    loop_us = loop_sec * 1e6 / args.iters
    current_us = current_sec * 1e6 / args.iters
    loop_mps = (args.iters * num_envs / loop_sec) / 1e6
    current_mps = (args.iters * num_envs / current_sec) / 1e6
    speedup = loop_sec / current_sec

    return Row(
        num_envs=num_envs,
        loop_us_per_call=loop_us,
        current_us_per_call=current_us,
        loop_m_samples_per_s=loop_mps,
        current_m_samples_per_s=current_mps,
        speedup_x=speedup,
    )


def print_rows(rows: Iterable[Row]):
    header = (
        "num_envs,loop_us_per_call,current_us_per_call,"
        "loop_Msamples_per_s,current_Msamples_per_s,speedup_x"
    )
    print(header)
    for row in rows:
        print(
            f"{row.num_envs},"
            f"{row.loop_us_per_call:.2f},"
            f"{row.current_us_per_call:.2f},"
            f"{row.loop_m_samples_per_s:.3f},"
            f"{row.current_m_samples_per_s:.3f},"
            f"{row.speedup_x:.2f}"
        )


def write_csv(path: Path, rows: Iterable[Row]):
    row_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "num_envs",
                "loop_us_per_call",
                "current_us_per_call",
                "loop_Msamples_per_s",
                "current_Msamples_per_s",
                "speedup_x",
            ],
        )
        writer.writeheader()
        for row in row_list:
            writer.writerow(asdict(row))


def main():
    args = parse_args()
    if any(n < 0 for n in args.num_envs):
        print("num_envs must be non-negative", file=sys.stderr)
        sys.exit(2)

    if not args.skip_equivalence_check:
        check_equivalence(args.seed)

    rows = [run_case(args, n) for n in args.num_envs]
    print_rows(rows)

    if args.csv_out is not None:
        write_csv(args.csv_out, rows)
        print(f"\nSaved CSV: {args.csv_out}")


if __name__ == "__main__":
    main()

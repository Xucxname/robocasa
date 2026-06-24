"""Collect SONIC-style demonstration data in RoboCasa environments.

This script provides a basic workflow for collecting simulation data using a
random policy. It is intended as a starting point for integrating SONIC or
other control strategies to collect high-quality demonstrations for
learning algorithms.

Usage example:

```bash
python -m robocasa.scripts.collect_sonic_demos \
    --task robocasa/PickPlaceCounterToCabinet \
    --robot Panda \
    --episodes 5 \
    --horizon 200 \
    --split pretrain \
    --output-dir /tmp/sonic_demos
```

The script saves each episode as an `.npz` file containing arrays of
observations and actions. You can extend this script to record additional
information (rewards, dones, timestamps, etc.) or to integrate a control
policy such as SONIC.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import robocasa


def collect_demo(
    task_name: str,
    robot: Optional[str],
    episodes: int,
    horizon: int,
    split: str,
    output_dir: str,
    seed: int,
) -> None:
    """Collect a number of episodes of random-action data and save to disk.

    Args:
        task_name: Gym registration name for the RoboCasa task, e.g.
            ``"robocasa/PickPlaceCounterToCabinet"``.
        robot: Name of the robot model to load, if applicable. Many tasks
            encode the robot in the task name; pass ``None`` to use the
            default.
        episodes: Number of episodes to collect.
        horizon: Maximum number of time steps per episode.
        split: Dataset split; one of ``pretrain`` or ``target``.
        output_dir: Directory to write the collected `.npz` files.
        seed: Base random seed; subsequent episodes will increment this seed.

    The collected files will be named ``<task>_<index>.npz`` within
    ``output_dir``. Each file contains two arrays: ``observations`` and
    ``actions``.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        env_seed = seed + episode
        env = gym.make(task_name, split=split, seed=env_seed)
        if robot is not None and hasattr(env.unwrapped, "set_robot_model"):
            # Some environments may support switching robot models at runtime
            try:
                env.unwrapped.set_robot_model(robot)
            except Exception:
                pass

        obs_list: list[np.ndarray] = []
        action_list: list[np.ndarray] = []

        obs, _ = env.reset()
        for _ in range(horizon):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            action_list.append(action)
            obs = next_obs
            if terminated or truncated:
                break

        file_name = f"{task_name.replace('/', '_')}_{episode}.npz"
        file_path = output_path / file_name
        np.savez(
            file_path,
            observations=np.array(obs_list, dtype=object),
            actions=np.array(action_list, dtype=object),
        )
        env.close()
        print(f"[collect] Saved {file_path}")



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect SONIC-style robot demonstrations")
    parser.add_argument(
        "--task",
        required=True,
        help="Gym registry name for the task (e.g. robocasa/PickPlaceCounterToCabinet)",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help="Optional robot model name to override the default robot of the task",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to collect",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Maximum time steps per episode",
    )
    parser.add_argument(
        "--split",
        default="pretrain",
        choices=["pretrain", "target"],
        help="Dataset split to use when constructing the environment",
    )
    parser.add_argument(
        "--output-dir",
        default="./sonic_demos",
        help="Directory where collected episodes will be saved",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed for reproducibility",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    collect_demo(
        task_name=args.task,
        robot=args.robot,
        episodes=args.episodes,
        horizon=args.horizon,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

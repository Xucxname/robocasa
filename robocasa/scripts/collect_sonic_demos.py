"""Collect SONIC-style demonstration data in RoboCasa environments.

This script collects simulation demonstration data with support for recording
additional metadata such as rewards, done flags, and camera images. It serves
as a flexible starting point for integrating SONIC or other control strategies
to collect high-quality demonstrations for learning algorithms.

Usage example:

```bash
python -m robocasa.scripts.collect_sonic_demos \
    --task robocasa/PickPlaceCounterToCabinet \
    --robot Panda \
    --episodes 5 \
    --horizon 200 \
    --split pretrain \
    --output-dir /tmp/sonic_demos \
    --save-images \
    --camera-name agentview
```

The script saves each episode as an `.npz` file containing arrays of
observations, actions, and optionally images, rewards, and done flags.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, List

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
    save_images: bool = False,
    camera_name: Optional[str] = None,
    camera_width: int = 256,
    camera_height: int = 256,
    save_meta: bool = False,
) -> None:
    """Collect demonstration data for a given task and robot.

    Args:
        task_name: Gym registration name for the RoboCasa task, e.g.
            ``"robocasa/PickPlaceCounterToCabinet"``.
        robot: Name of the robot model to load, if applicable. Many tasks
            encode the robot in the task name; pass ``None`` to use the default.
        episodes: Number of episodes to collect.
        horizon: Maximum number of time steps per episode.
        split: Dataset split; one of ``pretrain`` or ``target``.
        output_dir: Directory to write the collected `.npz` files.
        seed: Base random seed; subsequent episodes will increment this seed.
        save_images: Whether to capture and save rendered RGB images alongside
            observations and actions.
        camera_name: Optional camera name for rendering; defaults to environment
            default camera when ``None``.
        camera_width: Width of rendered images when saving images.
        camera_height: Height of rendered images when saving images.
        save_meta: Whether to save per-step rewards and done flags.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for episode in range(episodes):
        env_seed = seed + episode
        env = gym.make(task_name, split=split, seed=env_seed)
        if robot is not None and hasattr(env.unwrapped, "set_robot_model"):
            try:
                env.unwrapped.set_robot_model(robot)
            except Exception:
                pass

        obs_list: List[np.ndarray] = []
        action_list: List[np.ndarray] = []
        image_list: List[np.ndarray] = [] if save_images else []
        reward_list: List[float] = [] if save_meta else []
        done_list: List[bool] = [] if save_meta else []

        obs, _ = env.reset()
        for _ in range(horizon):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            action_list.append(action)

            if save_images:
                frame = None
                # Try to render a frame using the requested camera and resolution.
                try:
                    if camera_name is not None:
                        frame = env.render(
                            camera_name=camera_name,
                            width=camera_width,
                            height=camera_height,
                        )
                    else:
                        frame = env.render()
                except Exception:
                    # Fallback to default render without arguments.
                    try:
                        frame = env.render()
                    except Exception:
                        frame = None
                if frame is not None:
                    image_list.append(frame)

            if save_meta:
                reward_list.append(float(reward))
                done_list.append(bool(terminated or truncated))

            obs = next_obs
            if terminated or truncated:
                break

        file_name = f"{task_name.replace('/', '_')}_{episode}.npz"
        file_path = output_path / file_name
        data = {
            "observations": np.array(obs_list, dtype=object),
            "actions": np.array(action_list, dtype=object),
        }
        if save_images:
            data["images"] = np.array(image_list, dtype=object)
        if save_meta:
            data["rewards"] = np.array(reward_list, dtype=float)
            data["dones"] = np.array(done_list, dtype=bool)

        np.savez(file_path, **data)
        env.close()
        print(f"[collect] Saved {file_path}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Collect SONIC-style robot demonstrations with optional metadata"
    )
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
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Save images rendered at each step using env.render(). Images will be stored in the npz file as an object array.",
    )
    parser.add_argument(
        "--camera-name",
        default=None,
        help="Optional camera name to use when rendering images",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=256,
        help="Width of rendered images when saving images",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=256,
        help="Height of rendered images when saving images",
    )
    parser.add_argument(
        "--save-meta",
        action="store_true",
        help="Save per-step rewards and done flags in the output file",
    )
    return parser.parse_args(argv)



def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    collect_demo(
        task_name=args.task,
        robot=args.robot,
        episodes=args.episodes,
        horizon=args.horizon,
        split=args.split,
        output_dir=args.output_dir,
        seed=args.seed,
        save_images=args.save_images,
        camera_name=args.camera_name,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        save_meta=args.save_meta,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

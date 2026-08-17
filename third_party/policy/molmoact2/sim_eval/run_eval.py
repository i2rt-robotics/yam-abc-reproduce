#!/usr/bin/env python3
"""
MolmoAct2 closed-loop evaluation on ManiSkill environments.

Usage:

    python -m sim_eval.run_eval --policy-type remote-yam --remote-url http://<host>:8202/act -e BimanualYAMPutEverythingInBox-v1

    python -m sim_eval.run_eval --policy-type remote-droid --remote-url http://<host>:8000/act -e DroidPutEverythingInBox-v1

    # Multiple tasks:
    python -m sim_eval.run_eval --policy-type remote-yam --remote-url http://<host>:8202/act -e BimanualYAMPutEverythingInBox-v1 DroidPutEverythingInBox-v1
"""

import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Optional

import gymnasium as gym
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mani_skill.envs  # noqa: F401
from .tasks import *  # noqa: F401, F403
from .inference.client import DroidClient, YAMClient, MolmoActClientBase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).parent
DEFAULT_OUTPUT_DIR = str(EVAL_DIR / "outputs")

DEFAULT_LANGUAGE_INSTRUCTIONS: dict[str, str] = {
    "DroidPutEverythingInBox-v1":       "put everything into the box",
    "BimanualYAMPutEverythingInBox-v1": "put everything into the box",
}


PolicyType = Literal["remote-droid", "remote-yam"]


@dataclass
class EvalConfig:
    """MolmoAct2 closed-loop policy evaluation on ManiSkill."""

    policy_type: Annotated[PolicyType, tyro.conf.arg(aliases=["-p"])] = "remote-yam"
    """'remote-droid' or 'remote-yam'."""

    remote_url: Optional[str] = None
    """Full /act endpoint URL, e.g. http://<host>:8202/act."""

    remote_request_timeout: float = 60.0

    env_id: Annotated[list[str], tyro.conf.arg(aliases=["-e"])] = dataclasses.field(
        default_factory=list
    )
    """One or more ManiSkill env IDs."""

    language_instruction: Optional[str] = None
    """Language instruction override. If None, uses the per-env default."""

    n_episodes: Annotated[int, tyro.conf.arg(aliases=["-n"])] = 10
    max_episode_steps: int = 2000
    n_action_steps: Optional[int] = None
    """Actions per server chunk to execute. None = consume full chunk."""
    seed: int = 42

    control_freq: int = 30
    sim_freq: int = 150

    shader_pack: str = "rt-fast"
    """SAPIEN shader for sensor cameras (policy input)."""

    output_dir: str = DEFAULT_OUTPUT_DIR
    save_video: bool = True
    max_videos: int = 10
    verbose: bool = True



def _save_video(frames: list, path: Path, fps: int = 30) -> None:
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    processed = []
    for f in frames:
        if isinstance(f, torch.Tensor):
            f = f.cpu().numpy()
        if f.dtype != np.uint8:
            f = (f * 255).astype(np.uint8)
        processed.append(f)
    imageio.mimsave(str(path), processed, fps=fps)
    logger.info("Saved video → %s", path)


def _save_image(frame: np.ndarray, path: Path) -> None:
    import imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    if frame.dtype != np.uint8:
        frame = (frame * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
    imageio.imwrite(str(path), frame)


def _capture_frame(env: gym.Env) -> Optional[np.ndarray]:
    for method in (lambda: env.unwrapped.render_rgb_array(), lambda: env.render()):
        try:
            frame = method()
            if frame is None or hasattr(frame, "window"):
                continue
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if hasattr(frame, "ndim"):
                if frame.ndim == 4 and frame.shape[0] == 1:
                    frame = frame[0]
                return frame.copy()
        except Exception:
            pass
    return None


def _extract_input_frames(obs: dict) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for cam_name, sensor_data in (obs.get("sensor_data") or {}).items():
        img = sensor_data.get("rgb") if isinstance(sensor_data, dict) else sensor_data
        if img is None:
            continue
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img[0]
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        frames[cam_name] = img
    return frames



def _run_episode(
    env: gym.Env,
    client: MolmoActClientBase,
    obs: dict,
    instruction: str,
    max_steps: int,
) -> dict:
    total_reward = 0.0
    success = False
    frames: list[np.ndarray] = []
    input_frames = _extract_input_frames(obs)

    frame = _capture_frame(env)
    if frame is not None:
        frames.append(frame)

    step = 0
    for step in range(max_steps):
        action = client.infer(obs, instruction)
        obs, reward, terminated, truncated, info = env.step(action)

        if isinstance(reward, torch.Tensor):
            reward = reward.item() if reward.numel() == 1 else reward.sum().item()
        total_reward += reward

        sv = info.get("success", False)
        if isinstance(sv, torch.Tensor):
            sv = bool(sv.item() if sv.numel() == 1 else sv.any().item())
        if sv:
            success = True

        terminated = bool(terminated.any()) if isinstance(terminated, torch.Tensor) else bool(terminated)
        truncated  = bool(truncated.any())  if isinstance(truncated,  torch.Tensor) else bool(truncated)

        frame = _capture_frame(env)
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    return {
        "total_reward": total_reward,
        "success": success,
        "steps": step + 1,
        "frames": frames,
        "input_frames": input_frames,
    }


def _evaluate_task(env_id: str, client: MolmoActClientBase, config: EvalConfig) -> dict:
    instruction = config.language_instruction or DEFAULT_LANGUAGE_INSTRUCTIONS.get(env_id, env_id)
    logger.info("─" * 50)
    logger.info("Task: %s  |  instruction: %s", env_id, instruction)

    env = gym.make(
        env_id,
        obs_mode="rgb",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        max_episode_steps=config.max_episode_steps,
        reward_mode="none",
        sensor_configs=dict(shader_pack=config.shader_pack),
        sim_config=dict(sim_freq=config.sim_freq, control_freq=config.control_freq),
    )

    out_dir = Path(config.output_dir)
    episodes, successes, rewards, steps_list = [], [], [], []

    pbar = tqdm(range(config.n_episodes), desc=env_id, leave=False)
    for ep in pbar:
        obs, _ = env.reset(seed=config.seed + ep)
        torch.manual_seed(config.seed + ep)
        np.random.seed(config.seed + ep)

        if ep == 0:
            sensors = list((obs.get("sensor_data") or {}).keys())
            logger.info("Cameras in obs: %s  |  server expects: %s",
                        sensors, list(client.schema.camera_keys))

        result = _run_episode(env, client, obs, instruction, config.max_episode_steps)
        client.reset()

        episodes.append({
            "episode": ep,
            "success": result["success"],
            "reward": float(result["total_reward"]),
            "steps": result["steps"],
        })
        successes.append(result["success"])
        rewards.append(result["total_reward"])
        steps_list.append(result["steps"])

        if config.save_video and ep < config.max_videos and result["frames"]:
            _save_video(result["frames"], out_dir / "videos" / env_id / f"ep{ep:03d}.mp4")

        if ep < config.max_videos and result["input_frames"]:
            for cam, frame in result["input_frames"].items():
                _save_image(frame, out_dir / "frames" / env_id / f"ep{ep:03d}_{cam}.png")

        pbar.set_postfix(success=f"{np.mean(successes)*100:.0f}%")

    env.close()

    summary = {
        "env_id": env_id,
        "success_rate": float(np.mean(successes)),
        "avg_reward":   float(np.mean(rewards)),
        "avg_steps":    float(np.mean(steps_list)),
        "episodes":     episodes,
    }
    logger.info("%s  success=%.0f%%  avg_reward=%.2f  avg_steps=%.0f",
                env_id, summary["success_rate"] * 100,
                summary["avg_reward"], summary["avg_steps"])
    return summary



def main() -> None:
    config = tyro.cli(EvalConfig)

    if not config.env_id:
        raise SystemExit(
            "Specify at least one env with -e, e.g.:\n"
            "  -e BimanualYAMPutEverythingInBox-v1"
        )

    _clients = {"remote-droid": DroidClient, "remote-yam": YAMClient}
    if config.policy_type not in _clients:
        raise SystemExit(f"Unknown --policy-type '{config.policy_type}'. "
                         f"Known: {sorted(_clients.keys())}")
    if not config.remote_url:
        raise SystemExit("--remote-url is required")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    config.output_dir = str(Path(config.output_dir) / timestamp)

    logger.info("MolmoAct2 Eval  |  %s  url=%s", config.policy_type, config.remote_url)
    logger.info("Tasks: %s", config.env_id)
    logger.info("Output: %s", config.output_dir)

    client = _clients[config.policy_type](
        url=config.remote_url,
        n_action_steps=config.n_action_steps,
        request_timeout=config.remote_request_timeout,
    )

    all_results: dict = {"tasks": {}, "overall": {}}
    successes, rewards = [], []

    for env_id in config.env_id:
        try:
            r = _evaluate_task(env_id, client, config)
            all_results["tasks"][env_id] = r
            successes.append(r["success_rate"])
            rewards.append(r["avg_reward"])
        except Exception as e:
            logger.error("Failed on %s: %s", env_id, e)
            if config.verbose:
                import traceback; traceback.print_exc()

    if successes:
        all_results["overall"] = {
            "mean_success_rate": float(np.mean(successes)),
            "mean_reward":       float(np.mean(rewards)),
            "num_tasks":         len(successes),
        }
        logger.info("─" * 50)
        logger.info("Overall  success=%.0f%%  mean_reward=%.2f  tasks=%d",
                    all_results["overall"]["mean_success_rate"] * 100,
                    all_results["overall"]["mean_reward"],
                    all_results["overall"]["num_tasks"])

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(all_results, indent=2))
    logger.info("Results → %s/results.json", out)


if __name__ == "__main__":
    main()

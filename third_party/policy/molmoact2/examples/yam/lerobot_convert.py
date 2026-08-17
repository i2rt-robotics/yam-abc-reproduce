#!/usr/bin/env python
"""
Convert MolmoAct-style dataset to LeRobot v3.0 format in one shot.

This follows the same high-level pipeline and logic as molmoact_to_lerobot_v21.py:
  1. Episode-first layout on disk:
        data_dir/
        ├── 000001/
        │   ├── 000001.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        ├── 000002/
        │   ├── 000002.json
        │   ├── left_rgb/
        │   ├── right_rgb/
        │   └── front_rgb/
        └── ...

  2. Load all episodes into memory (qpos, actions, images).
  3. Stream frames into a LeRobotDataset via `add_frame` + `save_episode`.

Differences vs the v2.1 script:
  - Uses the v3.0 LeRobotDataset API from `lerobot.datasets.lerobot_dataset`.
  - Creates a v3.0 dataset directly (no v2.1→v3.0 conversion step).
  - Calls `dataset.finalize()` at the end to produce a valid v3.0 dataset.
  - Does NOT support resume into an existing dataset directory; the output_dir must
    be new or empty.

Usage example:

    python examples/yam/lerobot_convert.py \
        --data_dir /path/to/molmoact \
        --output_dir /path/to/molmoact_lerobot_v30 \
        --repo_id your-user/molmoact_v30 \
        --fps 30

You can then train with:

    # diffusion policy
    python src/lerobot/scripts/lerobot_train.py \
            --dataset.repo_id=your-user/lerobot_test \
            --policy.type=diffusion \
            --policy.repo_id=your-user/your_policy \
            --output_dir=./outputs/test_lerobot \
            --save_after_step=60000 \
            --steps=100000 \

Note: For local training, `--dataset.repo_id` can be the absolute path to the dataset directory.
"""

#!/usr/bin/env python
"""
Convert MolmoAct-style dataset to LeRobot v3.0 format.
Fixed to avoid OOM by using Lazy Loading of images.
"""

import argparse
import json
import gc
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
import tqdm
from tqdm import trange
from omegaconf import OmegaConf

# LeRobot v3.0 API
from lerobot.datasets.lerobot_dataset import LeRobotDataset


STATE_DIM_NAMES = [
    "left_joint1",
    "left_joint2",
    "left_joint3",
    "left_joint4",
    "left_joint5",
    "left_joint6",
    "left_gripper",
    "right_joint1",
    "right_joint2",
    "right_joint3",
    "right_joint4",
    "right_joint5",
    "right_joint6",
    "right_gripper",
]


def _sorted_image_files(camera_dir: Path) -> List[Path]:
    """Sort images by numeric stem when possible, fallback to lexical order."""
    image_files = [f for f in camera_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]]

    def _key(path: Path):
        stem = path.stem
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    return sorted(image_files, key=_key)


def _extract_task_text(frame: Dict[str, Any]) -> str | None:
    """Extract task text from common MolmoAct keys."""
    for key in ("task", "instruction", "language_instruction"):
        value = frame.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _load_camera_paths_from_dir(ep_dir: Path) -> Dict[str, List[Path]]:
    """Collect sorted image paths from each camera subdir (left_rgb/, right_rgb/, front_rgb/).

    Returns a dict keyed by stripped camera name ("left", "right", "front"), matching
    the keys consumed by ``infer_common_cameras`` and ``create_lerobot_dataset_v30``.
    """
    camera_paths: Dict[str, List[Path]] = {}
    for camera_dir in [ep_dir / "left_rgb", ep_dir / "right_rgb", ep_dir / "front_rgb"]:
        if not camera_dir.exists():
            continue
        cam_name = camera_dir.name.replace("_rgb", "")
        camera_paths[cam_name] = _sorted_image_files(camera_dir)
    return camera_paths


def _load_episode_from_json(ep_dir: Path, json_path: Path) -> Dict[str, Any] | None:
    """Load one episode whose per-frame metadata lives in a JSON list (DataSaver layout)."""
    if not json_path.exists():
        return None
    with open(json_path, "r") as f:
        episode_data = json.load(f)
    if not episode_data:
        return None

    task_description = _extract_task_text(episode_data[0])

    left_joint = np.array(
        [json.loads(fd["left_joint"]) for fd in episode_data], dtype=np.float32
    )
    right_joint = np.array(
        [json.loads(fd["right_joint"]) for fd in episode_data], dtype=np.float32
    )
    has_next_joint = all(
        ("next_left_joint" in fd and "next_right_joint" in fd) for fd in episode_data
    )
    next_qpos = None
    if has_next_joint:
        next_left = np.array(
            [json.loads(fd["next_left_joint"]) for fd in episode_data], dtype=np.float32
        )
        next_right = np.array(
            [json.loads(fd["next_right_joint"]) for fd in episode_data], dtype=np.float32
        )
        next_qpos = np.concatenate([next_left, next_right], axis=1)  # (T, 14)
    qpos = np.concatenate([left_joint, right_joint], axis=1)  # (T, 14)

    return {
        "task_description": task_description,
        "qpos": qpos,
        "next_qpos": next_qpos,
        "episode_length": len(qpos),
        "camera_paths": _load_camera_paths_from_dir(ep_dir),
    }


def _load_episode_from_h5(ep_dir: Path, h5_path: Path) -> Dict[str, Any] | None:
    """Load one episode whose per-frame metadata lives in ``episode.h5`` (eval layout)."""
    if not h5_path.exists():
        return None
    import h5py  # late import — keeps the JSON-only CLI path free of the h5py dep

    with h5py.File(h5_path, "r") as f:
        raw_instr = f.attrs.get("language_instruction", "")
        if isinstance(raw_instr, (bytes, np.bytes_)):
            instruction = raw_instr.decode()
        else:
            instruction = str(raw_instr)
        qpos = np.asarray(f["state"][:], dtype=np.float32)
        next_qpos = (
            np.asarray(f["next_state"][:], dtype=np.float32) if "next_state" in f else None
        )

    return {
        "task_description": instruction if instruction else None,
        "qpos": qpos,
        "next_qpos": next_qpos,
        "episode_length": len(qpos),
        "camera_paths": _load_camera_paths_from_dir(ep_dir),
    }


def load_molmoact_data(data_dir: str) -> List[Dict[str, Any]]:
    """Load all numbered episode subdirs (``NNNNNN/``) from ``data_dir``.

    Each ``NNNNNN/`` is expected to contain ``NNNNNN.json`` plus the per-camera image dirs.
    Image data is stored as path lists, not pixels, to keep memory near zero.
    """
    episodes: List[Dict[str, Any]] = []
    data_path = Path(data_dir)
    episode_dirs = sorted(
        [d for d in data_path.iterdir() if d.is_dir() and d.name.isdigit()]
    )
    print(f"Found {len(episode_dirs)} episodes under {data_dir}")

    for ep_dir in episode_dirs:
        json_path = ep_dir / f"{ep_dir.name}.json"
        ep = _load_episode_from_json(ep_dir, json_path)
        if ep is not None:
            episodes.append(ep)

    print(f"Metadata loaded for {len(episodes)} episodes.")
    return episodes


def load_droid_layout_data(
    base_dir: str | Path | None = None,
    include_eval: bool = False,
    explicit_paths: List[Path] | None = None,
) -> List[Dict[str, Any]]:
    """Load DROID-style episodes written by the eval system.

    Expected on-disk layout::

        base_dir/success/{date}/{ts}/   episode.h5 + left_rgb/ + right_rgb/ + front_rgb/
        base_dir/failure/{date}/{ts}/
        base_dir/eval/{ts}/              (only walked when include_eval=True)

    ``explicit_paths`` overrides the directory walk — useful when the caller
    already tracks the exact rollout dirs from this session.
    """
    episode_dirs: List[Path] = []
    if explicit_paths is not None:
        episode_dirs = [Path(p) for p in explicit_paths]
    else:
        if base_dir is None:
            raise ValueError("Must provide base_dir or explicit_paths.")
        base = Path(base_dir)
        labels = ["success", "failure"] + (["eval"] if include_eval else [])
        for label in labels:
            label_dir = base / label
            if not label_dir.exists():
                continue
            if label == "eval":
                episode_dirs.extend(d for d in label_dir.iterdir() if d.is_dir())
            else:
                for date_dir in label_dir.iterdir():
                    if date_dir.is_dir():
                        episode_dirs.extend(d for d in date_dir.iterdir() if d.is_dir())

    episode_dirs.sort()
    print(f"Found {len(episode_dirs)} DROID-layout rollouts.")

    episodes: List[Dict[str, Any]] = []
    for ep_dir in episode_dirs:
        ep = _load_episode_from_h5(ep_dir, ep_dir / "episode.h5")
        if ep is not None:
            episodes.append(ep)
    print(f"Loaded {len(episodes)} DROID-layout episodes.")
    return episodes


def infer_camera_shapes(episodes: List[Dict[str, Any]], camera_names: List[str]) -> Dict[str, Tuple[int, int, int]]:
    """Inspect one image per camera to determine H, W, C for the schema."""
    camera_shapes: Dict[str, Tuple[int, int, int]] = {}
    for ep in episodes:
        for cam_name in camera_names:
            paths = ep["camera_paths"].get(cam_name, [])
            if cam_name not in camera_shapes and paths:
                with Image.open(paths[0]) as img:
                    w, h = img.size
                    c = len(img.getbands())
                    camera_shapes[cam_name] = (h, w, c)
        if len(camera_shapes) == len(camera_names):
            return camera_shapes

    for cam_name in camera_names:
        camera_shapes.setdefault(cam_name, (360, 640, 3))
    return camera_shapes


def infer_common_cameras(episodes: List[Dict[str, Any]]) -> List[str]:
    """Use only cameras available (with at least 1 frame) in every episode."""
    if not episodes:
        return []

    common = None
    for ep in episodes:
        ep_cams = {cam for cam, paths in ep["camera_paths"].items() if len(paths) > 0}
        common = ep_cams if common is None else common & ep_cams

    return sorted(common) if common else []


def build_actions_from_episode(ep_data: Dict[str, Any], action_mode: str) -> np.ndarray:
    """Build action array aligned with observation.state semantics."""
    qpos = ep_data["qpos"]

    if action_mode == "next_joint_fields":
        next_qpos = ep_data.get("next_qpos")
        if isinstance(next_qpos, np.ndarray):
            if next_qpos.shape != qpos.shape:
                raise ValueError(
                    f"next_qpos shape mismatch: {next_qpos.shape} vs qpos {qpos.shape}"
                )
            return next_qpos.copy()
        # Fallback for datasets that do not have next_* fields.
        action_mode = "next_state"

    if action_mode == "copy_state":
        return qpos.copy()

    if action_mode == "next_state":
        if len(qpos) == 0:
            return qpos.copy()
        actions = np.empty_like(qpos)
        actions[:-1] = qpos[1:]
        actions[-1] = qpos[-1]
        return actions

    raise ValueError(f"Unsupported action_mode: {action_mode}")


def resolve_global_task(episodes: List[Dict[str, Any]], task_instruction: str | None) -> str:
    """Resolve one task string to use across all episodes."""
    if task_instruction and task_instruction.strip():
        return task_instruction.strip()

    for ep in episodes:
        candidate = ep.get("task_description")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return "perform the task"


def sanitize_episode_metadata_for_online_viz(output_dir: str) -> None:
    """Drop quantile-only columns from episode metadata for broader viewer compatibility."""
    try:
        import pandas as pd
    except Exception:
        print("Warning: pandas unavailable; skipping metadata sanitization.")
        return

    episodes_root = Path(output_dir) / "meta" / "episodes"
    if not episodes_root.exists():
        return

    parquet_files = sorted(episodes_root.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        return

    dropped_total = 0
    for p in parquet_files:
        df = pd.read_parquet(p)
        drop_cols = [
            c
            for c in df.columns
            if c.endswith("/q01")
            or c.endswith("/q10")
            or c.endswith("/q50")
            or c.endswith("/q90")
            or c.endswith("/q99")
        ]
        if drop_cols:
            df = df.drop(columns=drop_cols)
            df.to_parquet(p, index=False)
            dropped_total += len(drop_cols)

    if dropped_total > 0:
        print(f"Sanitized episode metadata for online visualizer compatibility (dropped {dropped_total} columns).")


def create_lerobot_dataset_v30(
    episodes,
    output_dir,
    repo_id,
    fps,
    robot_type,
    skip_initial_frames=0,
    action_mode="next_state",
    task_instruction: str | None = None,
    sanitize_online_viz_meta=True,
    vcodec="libsvtav1",
    image_writer_processes=0,
    image_writer_threads=0,
    parallel_encoding=True,
):
    output_path = Path(output_dir)
    if output_path.exists() and any(output_path.iterdir()):
        raise RuntimeError(f"Output directory '{output_dir}' is not empty.")

    camera_names = infer_common_cameras(episodes)
    if not camera_names:
        raise RuntimeError("No common cameras found across episodes with at least one frame.")
    global_task = resolve_global_task(episodes, task_instruction)
    print(f"Using one global task instruction for all episodes: {global_task}")

    camera_shapes = infer_camera_shapes(episodes, camera_names)
    image_dim_names = ["height", "width", "channels"]

    # Feature schema (same keys and shapes as the v2.1 script; v3.0 layout is handled by LeRobot).
    features: Dict[str, Dict[str, Any]] = {
        # Robot joint positions
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": STATE_DIM_NAMES,
        },
        # Actions (joint-space)
        "action": {
            "dtype": "float32",
            "shape": (14,),
            # Keep identical ordering and naming to observation.state for 1:1 comparison in visualizers.
            "names": STATE_DIM_NAMES,
        },
        # Image observations
    }
    for cam_name in camera_names:
        features[f"observation.images.camera_{cam_name}"] = {
            "dtype": "video",
            "shape": camera_shapes[cam_name],
            "names": image_dim_names,
        }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        root=output_dir,
        robot_type=robot_type,
        features=features,
        use_videos=True,
        image_writer_processes=image_writer_processes,
        image_writer_threads=image_writer_threads,
        batch_encoding_size=1,
        vcodec=vcodec,
    )

    for ep_idx, ep_data in enumerate(tqdm.tqdm(episodes, desc="Processing Episodes")):
        qpos = ep_data["qpos"]
        actions = build_actions_from_episode(ep_data, action_mode)
        cam_paths = ep_data["camera_paths"]
        per_cam_lengths = [len(cam_paths[cam]) for cam in camera_names]
        usable_len = min(len(qpos), len(actions), *per_cam_lengths)
        start_idx = max(0, int(skip_initial_frames))

        if usable_len <= start_idx:
            print(
                f"Skipping episode {ep_idx}: usable_len={usable_len}, "
                f"skip_initial_frames={start_idx}"
            )
            continue

        for f_idx in trange(start_idx, usable_len, leave=False, desc=f"Frames (ep {ep_idx})"):

            frame_data = {
                "observation.state": qpos[f_idx],
                "action": actions[f_idx],
                "task": global_task,
            }

            # JUST-IN-TIME IMAGE LOADING
            for cam_name in camera_names:
                # Open only this specific frame's image
                with Image.open(cam_paths[cam_name][f_idx]) as img:
                    frame_data[f"observation.images.camera_{cam_name}"] = img.convert("RGB")

            dataset.add_frame(frame_data)

        # Finalize the episode on disk and clear temp buffers
        dataset.save_episode(parallel_encoding=bool(parallel_encoding))

        # Explicitly trigger garbage collection after each episode
        gc.collect()

    print("Finalizing v3.0 dataset (writing Parquet and MP4 files)...")
    dataset.finalize()
    if sanitize_online_viz_meta:
        sanitize_episode_metadata_for_online_viz(output_dir)
    print(f"Success! Dataset saved to {output_dir}")


def upload_dataset_to_hf(repo_id: str, output_dir: str) -> None:
    """Upload local LeRobot dataset directory to the HF dataset repo."""
    upload_cmd = [
        "hf",
        "upload",
        str(repo_id),
        str(output_dir),
        "--repo-type=dataset",
    ]
    print(f"Uploading dataset to Hugging Face: {' '.join(upload_cmd)}")
    subprocess.run(upload_cmd, check=True)


def add_v30_tag(repo_id: str) -> None:
    """Create v3.0 dataset tag in HF repo if missing."""
    from huggingface_hub import HfApi

    hub_api = HfApi()
    refs = hub_api.list_repo_refs(repo_id, repo_type="dataset")
    existing_tags = {tag.name for tag in refs.tags}
    if "v3.0" in existing_tags:
        print("Tag v3.0 already exists. Skipping tag creation.")
        print(f"Repo tags: {sorted(existing_tags)}")
        return

    hub_api.create_tag(repo_id, tag="v3.0", repo_type="dataset")
    refs = hub_api.list_repo_refs(repo_id, repo_type="dataset")
    print(f"Repo tags: {[tag.name for tag in refs.tags]}")


def cleanup_local_dirs(data_dir: str, output_dir: str) -> None:
    """Delete local source and converted dataset directories."""
    for path in (Path(data_dir), Path(output_dir)):
        if path.exists():
            print(f"Deleting local directory: {path}")
            shutil.rmtree(path)
    print("Local cleanup completed.")


def load_defaults_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load converter defaults from launch config yaml."""
    defaults: Dict[str, Any] = {
        "data_dir": None,
        "output_dir": None,
        "repo_id": "molmoact_v30",
        "fps": 10,
        "robot_type": "molmoact_dual_arm",
        "skip_initial_frames": 0,
        "action_mode": "next_joint_fields",
        "task_instruction": None,
        "sanitize_online_viz_meta": 1,
        "vcodec": "h264",
        "image_writer_processes": 8,
        "image_writer_threads": 8,
        "parallel_encoding": 1,
        "upload_to_hf": 0,
        "delete_local_after_upload": 0,
    }

    cfg_path = Path(config_path).expanduser()
    if not cfg_path.exists():
        return defaults

    cfg = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)
    if not isinstance(cfg, dict):
        return defaults

    storage = cfg.get("storage", {}) or {}
    lerobot = cfg.get("lerobot", {}) or {}

    base_dir = storage.get("base_dir")
    task_directory = storage.get("task_directory")
    if base_dir and task_directory:
        base_dir_path = Path(base_dir).expanduser() / "data"
        defaults["data_dir"] = str(base_dir_path / str(task_directory))
        defaults["output_dir"] = str(base_dir_path / f"{task_directory}_lerobot_v30")

    repo_id = lerobot.get("hf_repo_id", storage.get("hf_repo_id"))
    if repo_id:
        defaults["repo_id"] = str(repo_id)

    defaults["fps"] = int(lerobot.get("fps", storage.get("lerobot_fps", cfg.get("hz", 10))))
    defaults["robot_type"] = str(
        lerobot.get("robot_type", storage.get("lerobot_robot_type", "molmoact_dual_arm"))
    )
    defaults["skip_initial_frames"] = int(
        lerobot.get("skip_initial_frames", storage.get("lerobot_skip_initial_frames", 0))
    )
    defaults["action_mode"] = str(
        lerobot.get("action_mode", storage.get("lerobot_action_mode", "next_joint_fields"))
    )
    defaults["task_instruction"] = storage.get("language_instruction")
    defaults["sanitize_online_viz_meta"] = int(
        bool(lerobot.get("sanitize_online_viz_meta", storage.get("sanitize_online_viz_meta", True)))
    )
    defaults["vcodec"] = str(lerobot.get("vcodec", "h264"))
    defaults["image_writer_processes"] = int(lerobot.get("image_writer_processes", 8))
    defaults["image_writer_threads"] = int(lerobot.get("image_writer_threads", 8))
    defaults["parallel_encoding"] = int(bool(lerobot.get("parallel_encoding", True)))
    defaults["upload_to_hf"] = int(bool(lerobot.get("auto_upload", False)))
    defaults["delete_local_after_upload"] = int(
        bool(lerobot.get("delete_local_after_upload", storage.get("delete_local_after_upload", False)))
    )
    return defaults


def main():
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config_path",
        type=str,
        default="../gello_software/configs/yam_left.yaml",
        help="Path to launch yaml used for default converter settings.",
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_defaults = load_defaults_from_yaml(bootstrap_args.config_path)

    parser = argparse.ArgumentParser(parents=[bootstrap_parser])
    parser.add_argument("--data_dir", type=str, default=config_defaults["data_dir"])
    parser.add_argument("--output_dir", type=str, default=config_defaults["output_dir"])
    parser.add_argument("--repo_id", type=str, default=config_defaults["repo_id"])
    parser.add_argument("--fps", type=int, default=config_defaults["fps"])
    parser.add_argument("--robot_type", type=str, default=config_defaults["robot_type"])
    parser.add_argument("--skip_initial_frames", type=int, default=config_defaults["skip_initial_frames"])
    parser.add_argument(
        "--action_mode",
        type=str,
        default="next_joint_fields",
        choices=["next_joint_fields", "next_state", "copy_state"],
        help=(
            "How to derive action. next_joint_fields uses next_left/right_joint from source JSON "
            "(recommended), next_state uses shifted state[t+1], copy_state uses state[t]."
        ),
    )
    parser.add_argument(
        "--task_instruction",
        type=str,
        default=config_defaults["task_instruction"],
        help="Single task instruction to apply to all episodes (recommended).",
    )
    parser.add_argument(
        "--sanitize_online_viz_meta",
        type=int,
        default=config_defaults["sanitize_online_viz_meta"],
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=config_defaults["vcodec"],
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec for encoding. Use h264 for faster conversion.",
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=config_defaults["image_writer_processes"],
        help="Number of image writer processes used by LeRobotDataset.",
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=config_defaults["image_writer_threads"],
        help="Number of image writer threads used by LeRobotDataset.",
    )
    parser.add_argument(
        "--parallel_encoding",
        type=int,
        default=config_defaults["parallel_encoding"],
        help="Set to 1 to parallelize multi-camera encoding in save_episode.",
    )
    parser.add_argument(
        "--upload_to_hf",
        type=int,
        default=config_defaults["upload_to_hf"],
        help="Set to 1 to upload converted dataset to Hugging Face.",
    )
    parser.add_argument(
        "--delete_local_after_upload",
        type=int,
        default=config_defaults["delete_local_after_upload"],
        help="Set to 1 to delete local source and converted data after successful upload.",
    )
    args = parser.parse_args()

    if not args.data_dir or not args.output_dir:
        raise ValueError(
            "data_dir/output_dir are not set. Provide them directly or ensure "
            "storage.base_dir and storage.task_directory are set in --config_path."
        )

    episodes = load_molmoact_data(args.data_dir)
    create_lerobot_dataset_v30(
        episodes,
        args.output_dir,
        args.repo_id,
        args.fps,
        args.robot_type,
        skip_initial_frames=args.skip_initial_frames,
        action_mode=args.action_mode,
        task_instruction=args.task_instruction,
        sanitize_online_viz_meta=bool(args.sanitize_online_viz_meta),
        vcodec=args.vcodec,
        image_writer_processes=max(0, int(args.image_writer_processes)),
        image_writer_threads=max(0, int(args.image_writer_threads)),
        parallel_encoding=bool(args.parallel_encoding),
    )

    if bool(args.upload_to_hf):
        upload_dataset_to_hf(args.repo_id, args.output_dir)
        add_v30_tag(args.repo_id)
        if bool(args.delete_local_after_upload):
            cleanup_local_dirs(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

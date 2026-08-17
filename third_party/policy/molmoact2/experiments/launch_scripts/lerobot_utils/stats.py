from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

from launch_scripts.data_mixtures import LEROBOT_TAG_PREFIX, is_lerobot_tag, strip_lerobot_tag_prefix
from launch_scripts.lerobot_utils.train_plan import (
    _normalize_registered_lerobot_tag_metadata,
    _require_tag_state_keys,
)
from olmo.data.data_loader import KwargsMixture
from olmo.data.lerobot_wrapper import _parse_repo_spec
from olmo.torch_util import get_world_size, get_global_rank, gather_object

log = logging.getLogger(__name__)

def _resolve_repo_root(repo_id: str, root_base: Optional[str]) -> Optional[str]:
    if not root_base:
        return None
    # Always scope to the requested repo_id to avoid silently reading stats/data
    # from an unrelated dataset root that happens to contain data/meta.
    base_root = os.path.expanduser(root_base)
    return os.path.join(base_root, repo_id)


def _coerce_count(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.size == 0:
        return None
    return float(arr.reshape(-1)[0])


def _normalize_feature_stats(stats: Dict[str, object]) -> Dict[str, object]:
    normalized: Dict[str, object] = {}
    for key, value in stats.items():
        if key == "count":
            normalized[key] = _coerce_count(value)
        else:
            normalized[key] = np.asarray(value, dtype=np.float64)
    return normalized


def _merge_feature_stats(
    accumulated: Optional[Dict[str, object]],
    new_stats: Dict[str, object],
) -> Dict[str, object]:
    new_norm = _normalize_feature_stats(new_stats)
    if accumulated is None:
        return new_norm
    merged = dict(accumulated)

    count_a = accumulated.get("count")
    count_b = new_norm.get("count")
    weight_a = count_a if count_a is not None else 1.0
    weight_b = count_b if count_b is not None else 1.0
    total_weight = weight_a + weight_b

    if "min" in accumulated and "min" in new_norm:
        merged["min"] = np.minimum(accumulated["min"], new_norm["min"])
    elif "min" in new_norm:
        merged["min"] = new_norm["min"]

    if "max" in accumulated and "max" in new_norm:
        merged["max"] = np.maximum(accumulated["max"], new_norm["max"])
    elif "max" in new_norm:
        merged["max"] = new_norm["max"]

    if "mean" in accumulated and "mean" in new_norm:
        merged_mean = (accumulated["mean"] * weight_a + new_norm["mean"] * weight_b) / total_weight
        merged["mean"] = merged_mean
    elif "mean" in new_norm:
        merged_mean = new_norm["mean"]
        merged["mean"] = merged_mean
    else:
        merged_mean = accumulated.get("mean")

    if "std" in accumulated and "std" in new_norm and merged_mean is not None:
        mean_a = accumulated.get("mean")
        mean_b = new_norm.get("mean")
        if mean_a is not None and mean_b is not None:
            var_a = np.square(accumulated["std"])
            var_b = np.square(new_norm["std"])
            var = (
                weight_a * (var_a + np.square(mean_a - merged_mean))
                + weight_b * (var_b + np.square(mean_b - merged_mean))
            ) / total_weight
            merged["std"] = np.sqrt(np.maximum(var, 0.0))
        else:
            merged["std"] = (accumulated["std"] * weight_a + new_norm["std"] * weight_b) / total_weight
    elif "std" in new_norm:
        merged["std"] = new_norm["std"]

    for key, value in new_norm.items():
        if not key.startswith("q"):
            continue
        if key in accumulated:
            merged[key] = (accumulated[key] * weight_a + value * weight_b) / total_weight
        else:
            merged[key] = value

    if count_a is not None or count_b is not None:
        merged["count"] = (count_a or 0.0) + (count_b or 0.0)
    else:
        merged.pop("count", None)

    return merged


def _serialize_stats(stats: Dict[str, object]) -> Dict[str, object]:
    serialized: Dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            serialized[key] = _serialize_stats(value)
        elif key == "count":
            if value is None:
                continue
            serialized[key] = [float(value)]
        elif isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        else:
            serialized[key] = value
    return serialized


def _stat_dim(stats: Dict[str, object]) -> Optional[int]:
    for key in ("min", "max", "mean", "std", "q01", "q99", "q10", "q90"):
        value = stats.get(key)
        if isinstance(value, (list, tuple, np.ndarray)):
            return len(value)
    return None


def _extract_feature_names(feature_spec: object) -> Optional[List[str]]:
    if not isinstance(feature_spec, dict):
        return None
    raw_names = feature_spec.get("names")
    if raw_names is None:
        return None
    if isinstance(raw_names, list):
        names = [str(v) for v in raw_names]
        return names or None
    if isinstance(raw_names, dict):
        names: List[str] = []
        for value in raw_names.values():
            if isinstance(value, (list, tuple)):
                names.extend(str(v) for v in value)
            elif value is not None:
                names.append(str(value))
        return names or None
    return [str(raw_names)]


_SO100_SO101_CANONICAL_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
_SO100_CANONICAL_JOINT_NAMES = _SO100_SO101_CANONICAL_JOINT_NAMES
_SO100_SO101_CANONICAL_TAGS = {
    "so100_so101_molmoact2",
}
_SO100_SO101_NAME_PREFIXES = ("main_", "left_", "right_")
_YAM_DUAL_MOLMOACT2_CANONICAL_JOINT_NAMES = [
    "left_joint_0.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_gripper.pos",
    "right_joint_0.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_gripper.pos",
]
_YAM_DUAL_MOLMOACT2_LEGACY_JOINT_NAME_ALIASES = [
    [
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
    ],
    [
        "left_m1",
        "left_m2",
        "left_m3",
        "left_m4",
        "left_m5",
        "left_m6",
        "left_m7",
        "right_m8",
        "right_m9",
        "right_m3",
        "right_m4",
        "right_m5",
        "right_m6",
        "right_m7",
    ],
]
_YAM_DUAL_MOLMOACT2_CANONICAL_TAGS = {
    "yam_dual_molmoact2",
}


def _canonicalize_feature_names(tag: str, key: str, names: Sequence[str]) -> List[str]:
    normalized = [str(name) for name in names]
    if (
        tag in _YAM_DUAL_MOLMOACT2_CANONICAL_TAGS
        and key in {"action", "observation.state"}
        and any(normalized == alias for alias in _YAM_DUAL_MOLMOACT2_LEGACY_JOINT_NAME_ALIASES)
    ):
        return list(_YAM_DUAL_MOLMOACT2_CANONICAL_JOINT_NAMES)
    if tag not in _SO100_SO101_CANONICAL_TAGS or key not in {"action", "observation.state"}:
        return normalized
    canonical: List[str] = []
    for name in normalized:
        stripped = name
        for prefix in _SO100_SO101_NAME_PREFIXES:
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):]
                break
        canonical.append(stripped)
    if canonical == _SO100_SO101_CANONICAL_JOINT_NAMES:
        return canonical
    return normalized


def _default_feature_mask(
    stats: Dict[str, object],
    *,
    normalize_gripper: bool,
) -> Optional[List[bool]]:
    stat_dim = _stat_dim(stats)
    if stat_dim is None:
        return None
    mask = [True] * stat_dim
    if normalize_gripper:
        return mask
    raw_names = stats.get("names")
    if not isinstance(raw_names, list):
        return mask
    names = [str(v) for v in raw_names]
    if len(names) != stat_dim:
        log.warning(
            "Feature names length %d does not match stat dim %d; leaving default all-true mask.",
            len(names),
            stat_dim,
        )
        return mask
    for idx, name in enumerate(names):
        if "gripper" in name.lower():
            mask[idx] = False
    return mask


_VECTOR_STAT_KEYS = ("min", "max", "mean", "std", "q01", "q99", "q10", "q90", "mask")


def _state_stats_key(state_keys: Sequence[str]) -> str:
    if not state_keys:
        raise ValueError("state_keys must be non-empty.")
    return str(state_keys[0])


def _as_1d_array(value: object) -> np.ndarray:
    return np.asarray(value, dtype=np.float64).reshape(-1)


def _feature_names_for_key(
    key: str,
    stats: Dict[str, object],
    *,
    prefix: bool,
) -> List[str]:
    stat_dim = _stat_dim(stats)
    raw_names = stats.get("names")
    names: Optional[List[str]] = None
    if isinstance(raw_names, list):
        names = [str(value) for value in raw_names]
    if names is None or stat_dim is None or len(names) != stat_dim:
        if stat_dim is None:
            return []
        names = [str(idx) for idx in range(stat_dim)]
    if prefix:
        return [f"{key}:{name}" for name in names]
    return names


def _concatenate_state_stats(
    *,
    tag: str,
    merged: Dict[str, object],
    state_keys: Sequence[str],
) -> Dict[str, object]:
    missing = [
        key
        for key in state_keys
        if key not in merged or not isinstance(merged.get(key), dict)
    ]
    if missing:
        raise ValueError(
            f"Tag '{tag}' is missing LeRobot stats for configured state_keys {missing}."
        )

    state_stats = [merged[str(key)] for key in state_keys]
    combined: Dict[str, object] = {}
    for stat_key in _VECTOR_STAT_KEYS:
        present = [stat_key in stats for stats in state_stats]
        if not any(present):
            continue
        if not all(present):
            raise ValueError(
                f"Tag '{tag}' has incomplete '{stat_key}' stats across state_keys {list(state_keys)}."
            )
        combined[stat_key] = np.concatenate(
            [_as_1d_array(stats[stat_key]) for stats in state_stats],
            axis=0,
        )

    counts = [
        _coerce_count(stats.get("count"))
        for stats in state_stats
        if _coerce_count(stats.get("count")) is not None
    ]
    if counts:
        first_count = counts[0]
        if any(not np.isclose(first_count, count) for count in counts[1:]):
            log.warning(
                "Tag %s has different frame counts across state_keys %s; using %s for combined state stats.",
                tag,
                list(state_keys),
                first_count,
            )
        combined["count"] = first_count

    combined_names: List[str] = []
    prefix_names = len(state_keys) > 1
    for key, stats in zip(state_keys, state_stats, strict=True):
        combined_names.extend(
            _feature_names_for_key(str(key), stats, prefix=prefix_names)
        )
    combined_dim = _stat_dim(combined)
    if combined_dim is not None and len(combined_names) == combined_dim:
        combined["names"] = combined_names
    return combined


def _apply_tag_metadata_masks(
    stats_by_tag: Dict[str, Dict[str, object]],
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
) -> None:
    for tag, tag_stats in stats_by_tag.items():
        metadata = tag_metadata_by_tag.get(tag)
        if not metadata:
            continue
        normalize_gripper = metadata.get("normalize_gripper")
        if not isinstance(normalize_gripper, bool):
            raise ValueError(
                f"LeRobot tag metadata for tag '{tag}' must define boolean normalize_gripper."
            )
        state_keys = _require_tag_state_keys(tag, metadata)
        for key, mask in (
            (metadata.get("action_key"), metadata.get("action_mask")),
            (_state_stats_key(state_keys), metadata.get("state_mask")),
        ):
            if not isinstance(key, str) or not key:
                continue
            if key not in tag_stats or not isinstance(tag_stats[key], dict):
                continue
            if isinstance(mask, (list, tuple, np.ndarray)):
                stat_dim = _stat_dim(tag_stats[key])
                if stat_dim is None:
                    log.warning("Unable to infer stat dimension for tag %s key %s; skipping mask.", tag, key)
                    continue
                if len(mask) != stat_dim:
                    log.warning(
                        "Mask length %d does not match stat dim %d for tag %s key %s; skipping explicit mask.",
                        len(mask),
                        stat_dim,
                        tag,
                        key,
                    )
                    continue
                tag_stats[key]["mask"] = [bool(v) for v in mask]
                continue
            if mask is not None:
                log.warning("Mask for tag %s key %s is not a list; ignoring explicit mask.", tag, key)
            default_mask = _default_feature_mask(tag_stats[key], normalize_gripper=normalize_gripper)
            if default_mask is None:
                log.warning("Unable to infer stat dimension for tag %s key %s; skipping default mask.", tag, key)
                continue
            tag_stats[key]["mask"] = default_mask


def _collect_tagged_stats(
    robot_mixture: List[KwargsMixture],
    *,
    root_base: Optional[str],
    tag_metadata_by_tag: Dict[str, Dict[str, object]],
) -> tuple[Dict[str, Dict[str, object]], Dict[str, str], str]:
    lerobot_tag_metadata_by_tag = _normalize_registered_lerobot_tag_metadata(tag_metadata_by_tag)
    repo_to_tag: Dict[str, str] = {}
    tag_to_repos: Dict[str, List[str]] = {}
    for mix in robot_mixture:
        raw_tag = str(mix.name or "default")
        if not any(ds_args.dataset_name.startswith("lerobot:") for ds_args in mix.datasets):
            continue
        if not is_lerobot_tag(raw_tag):
            raise ValueError(
                f"Mixture tag '{raw_tag}' contains LeRobot datasets and must use the "
                f"'{LEROBOT_TAG_PREFIX}<tag_name>' naming convention."
            )
        tag = strip_lerobot_tag_prefix(raw_tag)
        if tag not in lerobot_tag_metadata_by_tag:
            raise ValueError(f"Missing required LeRobot tag metadata for tag '{raw_tag}'.")
        for ds_args in mix.datasets:
            name = ds_args.dataset_name
            if not name.startswith("lerobot:"):
                continue
            parsed = _parse_repo_spec(name[len("lerobot:"):])
            repo_id = parsed.repo_id
            if repo_id in repo_to_tag and repo_to_tag[repo_id] != tag:
                raise ValueError(
                    f"Repo {repo_id} appears under multiple tags: "
                    f"{repo_to_tag[repo_id]} and {tag}"
                )
            repo_to_tag[repo_id] = tag
            tag_to_repos.setdefault(tag, []).append(repo_id)

    rank = get_global_rank()
    world_size = get_world_size()

    def _load_repo_stats(repo_id: str, repo_root: Optional[str]) -> Dict[str, object]:
        try:
            meta = LeRobotDatasetMetadata(repo_id, root=repo_root)
            feature_names_by_key: Dict[str, List[str]] = {}
            for key, feature_spec in (meta.features or {}).items():
                names = _extract_feature_names(feature_spec)
                if names:
                    feature_names_by_key[str(key)] = names
            return {
                "stats": meta.stats or {},
                "feature_names_by_key": feature_names_by_key,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load LeRobot metadata/stats for repo '{repo_id}' "
                f"at root '{repo_root}'. If local data is missing, ensure the repo "
                f"exists on the Hub and is accessible for download."
            ) from exc

    def _compute_tagged_stats() -> tuple[Dict[str, Dict[str, object]], Dict[str, str], str]:
        stats_cache: Dict[str, Optional[Dict[str, object]]] = {}
        stats_by_tag: Dict[str, Dict[str, object]] = {}

        for tag, repos in tag_to_repos.items():
            metadata = lerobot_tag_metadata_by_tag.get(tag)
            if not metadata:
                raise ValueError(f"Missing required LeRobot tag metadata for tag '{tag}'.")
            action_key = str(metadata.get("action_key") or "")
            state_keys = _require_tag_state_keys(tag, metadata)
            if not action_key:
                raise ValueError(
                    f"Tag '{tag}' must define non-empty action_key metadata."
                )
            allowed_keys = {action_key, *state_keys}
            require_exact_name_match = len(set(repos)) > 1
            merged: Dict[str, object] = {}
            feature_names_by_key: Dict[str, List[str]] = {}
            feature_name_source_repo: Dict[str, str] = {}
            for repo_id in repos:
                if repo_id not in stats_cache:
                    repo_root = _resolve_repo_root(repo_id, root_base)
                    stats_cache[repo_id] = _load_repo_stats(repo_id, repo_root)
                repo_metadata = stats_cache.get(repo_id)
                if not repo_metadata:
                    continue
                repo_stats = repo_metadata.get("stats") or {}
                repo_feature_names = repo_metadata.get("feature_names_by_key") or {}
                for key in allowed_keys:
                    names = repo_feature_names.get(key)
                    canonical_names = (
                        _canonicalize_feature_names(tag, key, names)
                        if names
                        else None
                    )
                    if require_exact_name_match:
                        if not canonical_names:
                            raise ValueError(
                                f"Tag '{tag}' groups multiple repos {sorted(set(repos))}, but repo '{repo_id}' "
                                f"is missing feature names for key '{key}'. Exact name matching is required "
                                "when multiple repos share a tag."
                            )
                        existing_names = feature_names_by_key.get(key)
                        if existing_names is None:
                            feature_names_by_key[key] = list(canonical_names)
                            feature_name_source_repo[key] = repo_id
                        elif list(existing_names) != list(canonical_names):
                            raise ValueError(
                                f"Inconsistent feature names for tag '{tag}' key '{key}' across repos. "
                                f"Expected {existing_names} from repo '{feature_name_source_repo[key]}', "
                                f"got {names} from repo '{repo_id}'."
                            )
                    elif canonical_names and key not in feature_names_by_key:
                        feature_names_by_key[key] = list(canonical_names)
                        feature_name_source_repo[key] = repo_id
                    if key not in repo_stats:
                        if key in state_keys:
                            raise ValueError(
                                f"Repo '{repo_id}' under tag '{tag}' is missing stats for state key '{key}'."
                            )
                        continue
                    accumulated = merged.get(key)
                    if accumulated is not None and not isinstance(accumulated, dict):
                        accumulated = None
                    merged[key] = _merge_feature_stats(accumulated, repo_stats[key])
            for key, names in feature_names_by_key.items():
                if key in merged and isinstance(merged[key], dict):
                    merged[key]["names"] = list(names)
            if merged:
                compact_stats: Dict[str, object] = {}
                if action_key in merged:
                    compact_stats[action_key] = merged[action_key]
                compact_stats[_state_stats_key(state_keys)] = _concatenate_state_stats(
                    tag=tag,
                    merged=merged,
                    state_keys=state_keys,
                )
                stats_by_tag[tag] = _serialize_stats(compact_stats)

        if stats_by_tag:
            default_tag = next(iter(stats_by_tag.keys()))
        else:
            default_tag = next(iter(tag_to_repos.keys()), "default")
        return stats_by_tag, repo_to_tag, default_tag

    if world_size <= 1:
        return _compute_tagged_stats()

    local_result: Optional[tuple[Dict[str, Dict[str, object]], Dict[str, str], str]]
    if rank == 0:
        local_result = _compute_tagged_stats()
    else:
        local_result = None

    gathered_results = gather_object(local_result)
    for result in gathered_results:
        if result is not None:
            return result

    raise RuntimeError("Distributed LeRobot stat collection failed: no rank produced a result.")

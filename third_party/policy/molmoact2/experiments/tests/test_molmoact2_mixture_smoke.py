import pytest

from launch_scripts.data_mixtures import TAG_METADATA_BY_TAG
from launch_scripts.lerobot_utils.train_plan import (
    infer_max_action_dim_from_lerobot_metadata,
    infer_max_action_horizon_from_lerobot_metadata,
)
from launch_scripts.train_lerobot import get_lerobot_training_data_plan


def _rate_sum(mixture):
    return sum(float(item.rate) for item in mixture)


def test_requested_lerobot_mixtures_build_and_are_normalized():
    for name in (
        "pre_post_train",
        "droid",
        "libero",
        "libero_goal",
        "yam",
        "so100_so101",
    ):
        plan = get_lerobot_training_data_plan(name)
        assert plan.combined_mixture
        assert _rate_sum(plan.combined_mixture) == pytest.approx(1.0)


def test_removed_lerobot_mixture_names_are_rejected():
    for name in ("pretrain", "pretrain_v2"):
        with pytest.raises(NotImplementedError):
            get_lerobot_training_data_plan(name)


@pytest.mark.parametrize(
    ("name", "expected_max_action_horizon"),
    [
        ("libero_goal", 10),
        ("droid", 15),
        ("pre_post_train", 30),
    ],
)
def test_max_action_horizon_is_inferred_from_mixture_metadata(name, expected_max_action_horizon):
    plan = get_lerobot_training_data_plan(name)

    assert (
        infer_max_action_horizon_from_lerobot_metadata(
            plan.robot_mixture,
            tag_metadata_by_tag=TAG_METADATA_BY_TAG,
        )
        == expected_max_action_horizon
    )


def test_max_action_dim_is_inferred_from_mixture_metadata():
    plan = get_lerobot_training_data_plan("pre_post_train")

    assert (
        infer_max_action_dim_from_lerobot_metadata(
            plan.robot_mixture,
            tag_metadata_by_tag=TAG_METADATA_BY_TAG,
        )
        == 32
    )

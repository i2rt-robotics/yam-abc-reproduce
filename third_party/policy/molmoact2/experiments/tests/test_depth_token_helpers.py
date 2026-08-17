import numpy as np

from olmo.extra_tokens import (
    build_depth_added_tokens,
    build_discrete_depth_string,
    get_robot_output_trigger_suffix,
    style_uses_action_output,
    style_uses_depth_output,
)


def test_build_depth_added_tokens_includes_output_and_boundaries():
    assert build_depth_added_tokens(3) == [
        "<depth_output>",
        "<depth_start>",
        "<depth_end>",
        "<depth_0>",
        "<depth_1>",
        "<depth_2>",
    ]


def test_build_discrete_depth_string_wraps_depth_bins():
    depth_bins = np.asarray([0, 7, 127], dtype=np.int64)
    assert build_discrete_depth_string(depth_bins, num_depth_tokens=128) == (
        "<depth_start><depth_0><depth_7><depth_127><depth_end>"
    )


def test_robot_depth_style_suffixes_and_flags_match_expected_outputs():
    assert get_robot_output_trigger_suffix("robot_action") == "<action_output>"
    assert get_robot_output_trigger_suffix("robot_depth") == "<depth_output>"
    assert get_robot_output_trigger_suffix("robot_depth_action") == "<depth_output><action_output>"
    assert style_uses_depth_output("robot_depth")
    assert style_uses_depth_output("robot_depth_action")
    assert not style_uses_depth_output("robot_action")
    assert style_uses_action_output("robot_action")
    assert style_uses_action_output("robot_depth_action")
    assert not style_uses_action_output("robot_depth")

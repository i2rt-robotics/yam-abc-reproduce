from types import SimpleNamespace

import pytest

from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy


def test_resolve_style_for_inference_rejects_depth_style_when_disabled():
    handles = SimpleNamespace(
        default_style="robot_depth_action",
        enable_depth_reasoning=False,
        depth_start_token_id=1,
        depth_end_token_id=2,
        depth_token_id_to_bin={3: 0},
    )

    with pytest.raises(ValueError, match="enable_depth_reasoning"):
        MolmoAct2Policy._resolve_style_for_inference(handles)

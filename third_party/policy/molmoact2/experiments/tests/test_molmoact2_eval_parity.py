from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lerobot.policies.molmoact2.modeling_molmoact2 import _disable_inference_token_bias
from lerobot.policies.molmoact2.hf_backend import MolmoAct2HFBackend


def test_hf_policy_preserves_native_observation_image_order():
    first = np.full((2, 2, 3), 1, dtype=np.uint8)
    second = np.full((2, 2, 3), 2, dtype=np.uint8)
    obs = {
        "observation.images.image2": first,
        "observation.images.image": second,
    }

    images = MolmoAct2HFBackend._extract_images(obs)

    assert [int(image[0, 0, 0]) for image in images] == [1, 2]


def test_disable_inference_token_bias_removes_legacy_nonpersistent_bias():
    token_bias = object()
    model = SimpleNamespace(transformer=SimpleNamespace(token_bias=token_bias))

    assert _disable_inference_token_bias(model)
    assert model.transformer.token_bias is None
    assert not _disable_inference_token_bias(model)

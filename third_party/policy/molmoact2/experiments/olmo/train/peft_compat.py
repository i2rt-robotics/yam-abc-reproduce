from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
from packaging import version


@contextmanager
def peft_fsdp2_linear_shape_compat(enabled: bool) -> Iterator[bool]:
    if not enabled:
        yield False
        return

    import peft
    from peft.tuners.lora import layer as lora_layer

    if version.parse(peft.__version__) < version.parse("0.18.0"):
        yield False
        return

    original_get_in_out_features = getattr(lora_layer, "_get_in_out_features", None)
    if original_get_in_out_features is None:
        raise RuntimeError(
            "PEFT compatibility patch requires peft.tuners.lora.layer._get_in_out_features, "
            f"but it is missing in PEFT {peft.__version__}."
        )

    def _compat_get_in_out_features(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear) and hasattr(module, "in_features") and hasattr(module, "out_features"):
            return module.in_features, module.out_features
        return original_get_in_out_features(module)

    lora_layer._get_in_out_features = _compat_get_in_out_features
    try:
        yield True
    finally:
        lora_layer._get_in_out_features = original_get_in_out_features


def validate_post_fsdp2_lora_shapes(module: torch.nn.Module, label: str) -> None:
    from peft.tuners.lora.layer import LoraLayer

    for module_name, submodule in module.named_modules():
        if not isinstance(submodule, LoraLayer):
            continue

        base_layer = submodule.get_base_layer()
        if not isinstance(base_layer, torch.nn.Linear):
            continue

        expected_out_features = getattr(base_layer, "out_features", None)
        if expected_out_features is None:
            continue

        for adapter_name, lora_b in submodule.lora_B.items():
            actual_out_features = getattr(lora_b, "out_features", None)
            if actual_out_features != expected_out_features:
                display_name = module_name or "<root>"
                raise RuntimeError(
                    f"{label} LoRA adapter shape mismatch after post-FSDP2 injection at '{display_name}' "
                    f"for adapter '{adapter_name}': expected lora_B.out_features={expected_out_features}, "
                    f"got {actual_out_features}."
                )

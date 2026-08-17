import dataclasses
from typing import Any, Iterable, Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class OverlengthCheckResult:
    actual_length: int
    max_length: int
    reason: str


class MalformedExampleError(ValueError):
    def __init__(
        self,
        *,
        reason: str,
        details: str,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.reason = reason
        self.details = details
        self.metadata = metadata or {}
        dataset_name = self.metadata.get("dataset_name")
        suffix = "" if dataset_name is None else f" dataset={dataset_name}"
        super().__init__(
            f"Malformed example: reason={self.reason} details={self.details}{suffix}"
        )


class OverlongExampleError(ValueError):
    def __init__(
        self,
        result: OverlengthCheckResult,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.result = result
        self.reason = result.reason
        self.actual_length = result.actual_length
        self.max_length = result.max_length
        self.metadata = metadata or {}
        dataset_name = self.metadata.get("dataset_name")
        suffix = "" if dataset_name is None else f" dataset={dataset_name}"
        super().__init__(
            f"Overlong example: reason={self.reason} "
            f"actual_length={self.actual_length} max_length={self.max_length}{suffix}"
        )


class AllLossTokensTruncatedError(OverlongExampleError):
    def __init__(self, *, actual_length: int, max_length: int, metadata: Optional[dict[str, Any]] = None):
        super().__init__(
            OverlengthCheckResult(
                actual_length=int(actual_length),
                max_length=int(max_length),
                reason="all_loss_truncated",
            ),
            metadata=metadata,
        )


def get_image_truncation_token_ids(tokenizer: Any) -> set[int]:
    ids = set()
    for attr in (
        "image_end_token_id",
        "image_start_token_id",
        "image_col_token_id",
        "image_patch_token_id",
        "image_low_res_token_id",
        "low_res_image_start_token_id",
        "frame_start_token_id",
        "frame_end_token_id",
    ):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            ids.add(int(value))
    return ids


def get_overlength_check(
    example: dict[str, Any],
    max_length: Optional[int],
    image_truncation_token_ids: Iterable[int],
) -> Optional[OverlengthCheckResult]:
    if max_length is None:
        return None

    input_tokens = np.asarray(example["input_tokens"])
    actual_length = int(len(input_tokens))
    max_length = int(max_length)
    if actual_length <= max_length:
        return None

    loss_masks = np.asarray(example.get("loss_masks"))
    truncated_tokens = input_tokens[max_length:]
    image_token_ids = set(int(x) for x in image_truncation_token_ids)

    if image_token_ids and any(int(token_id) in image_token_ids for token_id in truncated_tokens):
        reason = "image_truncation"
    elif loss_masks.size > 0 and np.any(loss_masks != 0) and np.all(loss_masks[:max_length] == 0):
        reason = "all_loss_truncated"
    else:
        reason = "plain_overlong"

    return OverlengthCheckResult(
        actual_length=actual_length,
        max_length=max_length,
        reason=reason,
    )

import logging
from typing import Dict, Any, List, Optional, Tuple, Union

import numpy as np
import torch
from olmo.util import flatten_lists

from olmo import tokenizer
from olmo.preprocessing.preprocessor_utils import TensorSpec, VariablePaddingSpec
from olmo.tokenizer import get_special_token_ids

numpy_to_torch_dtype_dict = {
    np.dtype("bool"): torch.bool,
    np.dtype("uint8"): torch.uint8,
    np.dtype("int8"): torch.int8,
    np.dtype("int16"): torch.int16,
    np.dtype("int32"): torch.int32,
    np.dtype("int64"): torch.int64,
    np.dtype("float16"): torch.float16,
    np.dtype("float32"): torch.float32,
    np.dtype("float64"): torch.float64,
    np.dtype("complex64"): torch.complex64,
    np.dtype("complex128"): torch.complex128,
    np.bool: torch.bool,
    np.uint8: torch.uint8,
    np.int8: torch.int8,
    np.int16: torch.int16,
    np.int32: torch.int32,  
    np.int64: torch.int64,
    np.float16: torch.float16,
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.complex64: torch.complex64,
    np.complex128: torch.complex128,
}


def _collate(tensors, max_shape=None, dtype=None, pad=None, pad_value=-1, allow_truncate=True):
    batch_shape = np.stack([x.shape for x in tensors if x is not None], 0).max(0)
    if pad == "to_max":
        row_shape = np.array(max_shape)
        assert np.all(batch_shape[1:] <= row_shape[1:])
        if not allow_truncate:
            if batch_shape[0] > row_shape[0]:
                raise ValueError()
            assert batch_shape[0] <= row_shape[0]
    elif pad is None:
        row_shape = batch_shape
    else:
        raise NotImplementedError(pad)

    # get the max per dim for all the dims in [1:] in tensor
    tensor = [x for x in tensors if x is not None][0]
    arr = np.full([len(tensors)] + row_shape.tolist(), pad_value,
                  dtype=dtype or tensor.dtype)
    for ix, tensor in enumerate(tensors):
        if tensor is not None:
            t = tensor[:row_shape[0]]
            slices = tuple(slice(None, dim) for dim in t.shape)
            arr[(ix,) + slices] = t
    return torch.from_numpy(arr)


class MMCollator:
    """Converts list of examples from our datasets into a tensor batch"""
    TEXT_KEYS = ["input_tokens", "target_tokens", "loss_masks", "subsegment_ids", "position_ids"]

    def __init__(self, special_tokens,
                 shapes_to_pad_to: Optional[Dict[str, Union[VariablePaddingSpec, TensorSpec]]]=None,
                 include_metadata=True, pad=None, skip_padding=None, cp_enabled=False,
                 packed_action_shape: Optional[Tuple[int, int]] = None):
        """
        :param max_text_len: truncate examples longer than this length
        :param include_metadata: whether to include the metadata in the out batch
        :param pad: how to pad the tensors
        :param max_crops: max number of crops to use if padding to the max sequence length
        """
        if pad:
            assert shapes_to_pad_to is not None
        self.shapes_to_pad_to = shapes_to_pad_to
        self.include_metadata = include_metadata
        self.pad = pad
        self.cp_enabled = cp_enabled
        self._packed_action_shape = packed_action_shape
        self._pad_packed_action_chunks = False
        self._packed_action_chunk_cap: Optional[int] = None
        self._special_tokens = np.array([
            special_tokens[tokenizer.IM_END_TOKEN],
            special_tokens[tokenizer.IM_START_TOKEN],
            special_tokens[tokenizer.IM_COL_TOKEN],
            special_tokens[tokenizer.IMAGE_LOW_RES_TOKEN],
            special_tokens[tokenizer.IMAGE_PATCH_TOKEN],
        ])[None, :]

    def set_packing_config(self, packing_config: Optional[Any]) -> None:
        enabled = bool(getattr(packing_config, "pad_action_chunks", False)) if packing_config is not None else False
        cap = getattr(packing_config, "action_chunk_cap", None) if packing_config is not None else None
        if enabled:
            if cap is None or int(cap) < 1:
                raise ValueError("Packed action chunk padding requires action_chunk_cap >= 1.")
            self._packed_action_chunk_cap = int(cap)
        else:
            self._packed_action_chunk_cap = None
        self._pad_packed_action_chunks = enabled

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert len(batch) > 0, "Given an empty batch"
        keys = batch[0].keys()
        if self.pad is not None:
            max_sequence_len = self.shapes_to_pad_to["tokens"].shape[0]
            # Sanity checks
            for ex in batch:
                if np.any(self._special_tokens == ex["input_tokens"][max_sequence_len:][:, None]):
                    raise ValueError("An image would have gotten truncated!")
                if not self.cp_enabled:
                    ## In CP, as a device might only process image + prompt tokens and no response tokens where the loss 
                    ## would be zero whch is ok.
                    if np.any(ex["loss_masks"] != 0) and np.all(ex["loss_masks"][:max_sequence_len] == 0):
                        metadata = ex.get("metadata") or {}
                        first_loss_indices = np.flatnonzero(ex["loss_masks"] != 0)
                        context = []
                        if metadata.get("repo_id"):
                            context.append(f"repo_id={metadata['repo_id']}")
                        if metadata.get("frame_index") is not None:
                            context.append(f"frame_index={metadata['frame_index']}")
                        if first_loss_indices.size > 0:
                            context.append(f"first_loss_idx={int(first_loss_indices[0])}")
                        detail = f" ({', '.join(context)})" if context else ""
                        raise ValueError(f"All loss tokens truncated!{detail}")
        else:
            max_sequence_len = None

        # Collate text fields
        out = {}
        for key in self.TEXT_KEYS:
            # If one example has subsegment_ids, all examples need it as well
            # Note it is okay if some batches have subsegment_ids and some (for different devices)
            # don't since it only used to modify the attention mask
            if key == "subsegment_ids":
                if any(key in ex for ex in batch):
                    for ex in batch:
                        if "subsegment_ids" not in ex:
                            ex["subsegment_ids"] = np.ones_like(ex["input_tokens"])
                else:
                    continue
            dtype = np.float32 if key == "loss_masks" else np.int64
            out[key] = _collate(
                [ex.get(key) for ex in batch], [max_sequence_len], dtype, pad=self.pad)

        # Collate any other fields
        for key, spec in self.shapes_to_pad_to.items():
            if key == "tokens":
                continue
            tensors = [ex.get(key) for ex in batch]
            if all(x is None for x in tensors):
                if self.pad is not None:
                    # Create an all-padding input, we might need this to make sure each device
                    # in a FSDP setup gets the same inputs
                    out[key] = torch.full(
                        [len(tensors)] + list(spec.shape), -1,
                        dtype=numpy_to_torch_dtype_dict[spec.dtype],
                    )
            else:
                if isinstance(spec, VariablePaddingSpec):
                    pad = None
                else:
                    pad = self.pad
                pad_value = 0 if spec.dtype == np.uint8 else -1
                out[key] = _collate([ex.get(key) for ex in batch], spec.shape,
                                        dtype=spec.dtype, pad=pad, pad_value=pad_value, allow_truncate=False)

        def _collate_action_chunks() -> Optional[Dict[str, torch.Tensor]]:
            has_chunks = any(
                (ex.get("packed_states") is not None or
                 ex.get("packed_actions") is not None)
                for ex in batch
            )
            stabilize_chunks = self._pad_packed_action_chunks
            chunk_cap = self._packed_action_chunk_cap
            if not has_chunks and not stabilize_chunks:
                return None

            def _ensure_count(current: Optional[int], arr: np.ndarray, key: str) -> int:
                if current is None:
                    return arr.shape[0]
                if arr.shape[0] != current:
                    raise ValueError(f"Inconsistent chunk counts for '{key}': expected {current}, got {arr.shape[0]}")
                return current

            action_shape = self._packed_action_shape
            if action_shape is None:
                action_shape = next(
                    (
                        tuple(np.asarray(ex["packed_actions"]).shape[1:])
                        for ex in batch
                        if ex.get("packed_actions") is not None
                    ),
                    None,
                )
            action_pad_shape = None if action_shape is None else action_shape[:-1]
            action_dim_pad_shape = None if action_shape is None else (action_shape[-1],)
            inferred_state_shape = next(
                (
                    tuple(np.asarray(ex["packed_states"]).shape[1:])
                    for ex in batch
                    if ex.get("packed_states") is not None
                ),
                None,
            )

            per_example: List[Dict[str, Any]] = []
            packed_num_chunks: List[int] = []
            overflow_flags: List[bool] = []

            for batch_ix, example in enumerate(batch):
                num_chunks: Optional[int] = None
                state_arr = example.get("packed_states")
                if state_arr is not None:
                    arr = np.asarray(state_arr, dtype=np.float32)
                    num_chunks = _ensure_count(num_chunks, arr, "states")
                action_arr = example.get("packed_actions")
                if action_arr is not None:
                    arr = np.asarray(action_arr, dtype=np.float32)
                    num_chunks = _ensure_count(num_chunks, arr, "actions")
                pad_arr = example.get("packed_action_horizon_is_pad")
                if pad_arr is None:
                    pad_arr = example.get("packed_action_is_pad")
                if pad_arr is not None:
                    arr = np.asarray(pad_arr, dtype=np.bool_)
                    num_chunks = _ensure_count(num_chunks, arr, "action_horizon_is_pad")
                dim_pad_arr = example.get("packed_action_dim_is_pad")
                if dim_pad_arr is not None:
                    arr = np.asarray(dim_pad_arr, dtype=np.bool_)
                    num_chunks = _ensure_count(num_chunks, arr, "action_dim_is_pad")
                example_arr = example.get("packed_example_ids")
                if example_arr is not None:
                    arr = np.asarray(example_arr, dtype=np.int64)
                    num_chunks = _ensure_count(num_chunks, arr, "packed_example_ids")
                elif num_chunks is not None:
                    arr = np.zeros(num_chunks, dtype=np.int64)
                else:
                    arr = None
                packed_num_chunks_arr = example.get("packed_num_chunks")
                if packed_num_chunks_arr is not None:
                    declared = np.asarray(packed_num_chunks_arr, dtype=np.int32).reshape(-1)
                    if declared.size != 1:
                        raise ValueError(f"Expected 'packed_num_chunks' to have shape [1], got {declared.shape}")
                    if num_chunks is None:
                        num_chunks = int(declared[0])
                    elif int(declared[0]) != num_chunks:
                        raise ValueError(
                            f"Inconsistent chunk counts for 'packed_num_chunks': expected {num_chunks}, got {int(declared[0])}"
                        )
                actual_num_chunks = 0 if num_chunks is None else int(num_chunks)
                packed_num_chunks.append(actual_num_chunks)
                overflow_flags.append(bool(stabilize_chunks and actual_num_chunks > int(chunk_cap)))
                per_example.append(
                    {
                        "batch_ix": batch_ix,
                        "num_chunks": actual_num_chunks,
                        "states": state_arr,
                        "actions": action_arr,
                        "action_horizon_is_pad": pad_arr,
                        "action_dim_is_pad": dim_pad_arr,
                        "packed_example_ids": arr,
                    }
                )

            packed_num_chunks_tensor = torch.tensor(packed_num_chunks, dtype=torch.int32)

            def _legacy_chunk_tensors() -> Optional[Dict[str, torch.Tensor]]:
                state_chunks: List[np.ndarray] = []
                action_chunks: List[np.ndarray] = []
                pad_chunks: List[np.ndarray] = []
                dim_pad_chunks: List[np.ndarray] = []
                batch_ids: List[np.ndarray] = []
                example_ids: List[np.ndarray] = []
                valid_chunks: List[np.ndarray] = []

                for item in per_example:
                    num_chunks = int(item["num_chunks"])
                    if num_chunks <= 0:
                        continue
                    state_arr = item["states"]
                    action_arr = item["actions"]
                    pad_arr = item["action_horizon_is_pad"]
                    dim_pad_arr = item["action_dim_is_pad"]
                    example_arr = item["packed_example_ids"]
                    if state_arr is not None:
                        state_chunks.append(np.asarray(state_arr, dtype=np.float32))
                    if action_arr is not None:
                        action_chunks.append(np.asarray(action_arr, dtype=np.float32))
                    if pad_arr is not None:
                        pad_chunks.append(np.asarray(pad_arr, dtype=np.bool_))
                    if dim_pad_arr is not None:
                        dim_pad_chunks.append(np.asarray(dim_pad_arr, dtype=np.bool_))
                    if example_arr is None:
                        example_arr = np.zeros(num_chunks, dtype=np.int64)
                    example_ids.append(np.asarray(example_arr, dtype=np.int64))
                    batch_ids.append(np.full(num_chunks, int(item["batch_ix"]), dtype=np.int64))
                    valid_chunks.append(np.ones(num_chunks, dtype=np.bool_))

                if not batch_ids:
                    return None

                tensors: Dict[str, torch.Tensor] = {}
                if state_chunks:
                    tensors["states"] = torch.from_numpy(np.concatenate(state_chunks, axis=0)).to(torch.float32)
                if action_chunks:
                    tensors["actions"] = torch.from_numpy(np.concatenate(action_chunks, axis=0)).to(torch.float32)
                if pad_chunks:
                    tensors["action_horizon_is_pad"] = torch.from_numpy(np.concatenate(pad_chunks, axis=0)).to(torch.bool)
                if dim_pad_chunks:
                    tensors["action_dim_is_pad"] = torch.from_numpy(np.concatenate(dim_pad_chunks, axis=0)).to(torch.bool)
                tensors["packed_batch_idx"] = torch.from_numpy(np.concatenate(batch_ids, axis=0)).to(torch.long)
                tensors["packed_example_ids"] = torch.from_numpy(np.concatenate(example_ids, axis=0)).to(torch.long)
                tensors["packed_action_chunk_is_valid"] = torch.from_numpy(
                    np.concatenate(valid_chunks, axis=0)
                ).to(torch.bool)
                return tensors

            if stabilize_chunks and any(overflow_flags):
                tensors = _legacy_chunk_tensors()
                if tensors is None:
                    return None
                tensors["packed_num_chunks"] = packed_num_chunks_tensor
                tensors["packed_action_chunk_cap"] = torch.full(
                    (len(batch),), int(chunk_cap), dtype=torch.int32
                )
                tensors["packed_action_chunk_overflow"] = torch.tensor(overflow_flags, dtype=torch.bool)
                return tensors

            if not stabilize_chunks:
                tensors = _legacy_chunk_tensors()
                if tensors is None:
                    return None
                tensors["packed_num_chunks"] = packed_num_chunks_tensor
                return tensors

            if action_shape is None:
                raise ValueError(
                    "Packed action chunk padding requires an action shape. "
                    "Configure the collator with packed_action_shape when using actionless packed batches."
                )
            if action_pad_shape is None or action_dim_pad_shape is None or chunk_cap is None:
                raise ValueError("Packed action chunk padding is missing required action padding metadata.")

            state_rows: List[np.ndarray] = []
            action_rows: List[np.ndarray] = []
            pad_rows: List[np.ndarray] = []
            dim_pad_rows: List[np.ndarray] = []
            batch_idx_rows: List[np.ndarray] = []
            example_id_rows: List[np.ndarray] = []
            valid_rows: List[np.ndarray] = []

            for item in per_example:
                batch_ix = int(item["batch_ix"])
                num_chunks = int(item["num_chunks"])
                state_arr = item["states"]
                action_arr = item["actions"]
                pad_arr = item["action_horizon_is_pad"]
                dim_pad_arr = item["action_dim_is_pad"]
                example_arr = item["packed_example_ids"]

                if action_arr is None:
                    action_arr = np.zeros((0, *action_shape), dtype=np.float32)
                else:
                    action_arr = np.asarray(action_arr, dtype=np.float32)
                if action_arr.shape[0] != num_chunks:
                    raise ValueError(
                        f"Inconsistent chunk counts for 'actions': expected {num_chunks}, got {action_arr.shape[0]}"
                    )

                if state_arr is not None:
                    state_arr = np.asarray(state_arr, dtype=np.float32)
                    if state_arr.shape[0] != num_chunks:
                        raise ValueError(
                            f"Inconsistent chunk counts for 'states': expected {num_chunks}, got {state_arr.shape[0]}"
                        )
                    inferred_state_shape = inferred_state_shape or tuple(state_arr.shape[1:])

                if pad_arr is None:
                    pad_arr = np.zeros((num_chunks, *action_pad_shape), dtype=np.bool_)
                else:
                    pad_arr = np.asarray(pad_arr, dtype=np.bool_)
                if pad_arr.shape[0] != num_chunks:
                    raise ValueError(
                        f"Inconsistent chunk counts for 'action_horizon_is_pad': expected {num_chunks}, got {pad_arr.shape[0]}"
                    )

                if dim_pad_arr is None:
                    dim_pad_arr = np.zeros((num_chunks, *action_dim_pad_shape), dtype=np.bool_)
                else:
                    dim_pad_arr = np.asarray(dim_pad_arr, dtype=np.bool_)
                if dim_pad_arr.shape[0] != num_chunks:
                    raise ValueError(
                        f"Inconsistent chunk counts for 'action_dim_is_pad': expected {num_chunks}, got {dim_pad_arr.shape[0]}"
                    )

                if example_arr is None:
                    example_arr = np.zeros(num_chunks, dtype=np.int64)
                else:
                    example_arr = np.asarray(example_arr, dtype=np.int64)
                if example_arr.shape[0] != num_chunks:
                    raise ValueError(
                        f"Inconsistent chunk counts for 'packed_example_ids': expected {num_chunks}, got {example_arr.shape[0]}"
                    )

                pad_count = int(chunk_cap) - num_chunks
                if pad_count < 0:
                    raise ValueError(f"Chunk cap {int(chunk_cap)} is smaller than actual chunk count {num_chunks}")

                action_rows.append(action_arr)
                pad_rows.append(pad_arr)
                dim_pad_rows.append(dim_pad_arr)
                batch_idx_rows.append(np.full(num_chunks, batch_ix, dtype=np.int64))
                example_id_rows.append(example_arr)
                valid_rows.append(np.ones(num_chunks, dtype=np.bool_))

                if inferred_state_shape is not None:
                    if state_arr is None:
                        state_rows.append(np.zeros((num_chunks, *inferred_state_shape), dtype=np.float32))
                    else:
                        state_rows.append(state_arr)

                if pad_count > 0:
                    action_rows.append(np.zeros((pad_count, *action_shape), dtype=np.float32))
                    pad_rows.append(np.ones((pad_count, *action_pad_shape), dtype=np.bool_))
                    dim_pad_rows.append(np.ones((pad_count, *action_dim_pad_shape), dtype=np.bool_))
                    batch_idx_rows.append(np.full(pad_count, batch_ix, dtype=np.int64))
                    example_id_rows.append(np.full(pad_count, -1, dtype=np.int64))
                    valid_rows.append(np.zeros(pad_count, dtype=np.bool_))
                    if inferred_state_shape is not None:
                        state_rows.append(np.zeros((pad_count, *inferred_state_shape), dtype=np.float32))

            tensors = {
                "actions": torch.from_numpy(np.concatenate(action_rows, axis=0)).to(torch.float32),
                "action_horizon_is_pad": torch.from_numpy(np.concatenate(pad_rows, axis=0)).to(torch.bool),
                "action_dim_is_pad": torch.from_numpy(np.concatenate(dim_pad_rows, axis=0)).to(torch.bool),
                "packed_batch_idx": torch.from_numpy(np.concatenate(batch_idx_rows, axis=0)).to(torch.long),
                "packed_example_ids": torch.from_numpy(np.concatenate(example_id_rows, axis=0)).to(torch.long),
                "packed_action_chunk_is_valid": torch.from_numpy(np.concatenate(valid_rows, axis=0)).to(torch.bool),
                "packed_num_chunks": packed_num_chunks_tensor,
                "packed_action_chunk_cap": torch.full((len(batch),), int(chunk_cap), dtype=torch.int32),
                "packed_action_chunk_overflow": torch.tensor(overflow_flags, dtype=torch.bool),
            }
            if state_rows:
                tensors["states"] = torch.from_numpy(np.concatenate(state_rows, axis=0)).to(torch.float32)
            return tensors

        def _stack_optional_dense(key: str, torch_dtype: torch.dtype, numpy_dtype=None, legacy_key: Optional[str] = None):
            values = []
            for ex in batch:
                value = ex.get(key)
                if value is None and legacy_key is not None:
                    value = ex.get(legacy_key)
                values.append(value)
            present = [v is not None for v in values]
            if not any(present):
                return
            if not all(present):
                raise ValueError(f"Examples in batch are missing key '{key}'")
            arrays = []
            for value in values:
                arr = np.asarray(value)
                if numpy_dtype is not None:
                    arr = arr.astype(numpy_dtype, copy=False)
                arrays.append(arr)
            first_shape = arrays[0].shape
            if any(arr.shape != first_shape for arr in arrays[1:]):
                raise ValueError(f"Inconsistent shapes for '{key}': {[arr.shape for arr in arrays]}")
            stacked = np.stack(arrays)
            tensor = torch.from_numpy(stacked)
            out[key] = tensor.to(torch_dtype)

        def _collate_packed_depth_side_channels() -> Optional[Dict[str, torch.Tensor]]:
            has_packed_depth = any(
                ex.get("packed_depth_updated_mask") is not None
                or ex.get("packed_depth_buffer_codes") is not None
                or ex.get("packed_depth_example_ids") is not None
                for ex in batch
            )
            if not has_packed_depth:
                return None

            def _ensure_count(current: Optional[int], arr: np.ndarray, key: str) -> int:
                if current is None:
                    return int(arr.shape[0])
                if int(arr.shape[0]) != int(current):
                    raise ValueError(
                        f"Inconsistent packed depth counts for '{key}': expected {current}, got {arr.shape[0]}"
                    )
                return int(current)

            mask_width = next(
                (
                    int(np.asarray(ex["packed_depth_updated_mask"]).shape[1])
                    for ex in batch
                    if ex.get("packed_depth_updated_mask") is not None
                ),
                None,
            )
            code_width = next(
                (
                    int(np.asarray(ex["packed_depth_buffer_codes"]).shape[1])
                    for ex in batch
                    if ex.get("packed_depth_buffer_codes") is not None
                ),
                None,
            )

            per_example: List[Dict[str, Any]] = []
            max_rows = 0
            num_rows_list: List[int] = []
            for ex in batch:
                num_rows: Optional[int] = None
                updated_mask = ex.get("packed_depth_updated_mask")
                if updated_mask is not None:
                    arr = np.asarray(updated_mask, dtype=np.bool_)
                    num_rows = _ensure_count(num_rows, arr, "packed_depth_updated_mask")
                    if mask_width is not None and int(arr.shape[1]) != int(mask_width):
                        raise ValueError(
                            f"Inconsistent packed depth mask width: expected {mask_width}, got {arr.shape[1]}"
                        )
                buffer_codes = ex.get("packed_depth_buffer_codes")
                if buffer_codes is not None:
                    arr = np.asarray(buffer_codes, dtype=np.int64)
                    num_rows = _ensure_count(num_rows, arr, "packed_depth_buffer_codes")
                    if code_width is not None and int(arr.shape[1]) != int(code_width):
                        raise ValueError(
                            f"Inconsistent packed depth code width: expected {code_width}, got {arr.shape[1]}"
                        )
                example_ids = ex.get("packed_depth_example_ids")
                if example_ids is not None:
                    arr = np.asarray(example_ids, dtype=np.int64).reshape(-1)
                    num_rows = _ensure_count(num_rows, arr, "packed_depth_example_ids")
                packed_num = ex.get("packed_num_depth_examples")
                if packed_num is not None:
                    declared = np.asarray(packed_num, dtype=np.int32).reshape(-1)
                    if declared.size != 1:
                        raise ValueError(
                            f"Expected 'packed_num_depth_examples' to have shape [1], got {declared.shape}"
                        )
                    if num_rows is None:
                        num_rows = int(declared[0])
                    elif int(declared[0]) != int(num_rows):
                        raise ValueError(
                            f"Inconsistent packed depth counts: expected {num_rows}, got {int(declared[0])}"
                        )
                resolved_rows = 0 if num_rows is None else int(num_rows)
                max_rows = max(max_rows, resolved_rows)
                num_rows_list.append(resolved_rows)
                per_example.append(
                    {
                        "updated_mask": None if updated_mask is None else np.asarray(updated_mask, dtype=np.bool_),
                        "buffer_codes": None if buffer_codes is None else np.asarray(buffer_codes, dtype=np.int64),
                        "example_ids": None if example_ids is None else np.asarray(example_ids, dtype=np.int64).reshape(-1),
                    }
                )

            tensors: Dict[str, torch.Tensor] = {
                "packed_num_depth_examples": torch.tensor(num_rows_list, dtype=torch.int32),
                "packed_depth_row_is_valid": torch.zeros((len(batch), max_rows), dtype=torch.bool),
            }
            if mask_width is not None:
                tensors["packed_depth_updated_mask"] = torch.zeros(
                    (len(batch), max_rows, mask_width),
                    dtype=torch.bool,
                )
            if code_width is not None:
                tensors["packed_depth_buffer_codes"] = torch.full(
                    (len(batch), max_rows, code_width),
                    -1,
                    dtype=torch.long,
                )
            tensors["packed_depth_example_ids"] = torch.full(
                (len(batch), max_rows),
                -1,
                dtype=torch.long,
            )

            for batch_ix, (rows, example_data) in enumerate(zip(num_rows_list, per_example)):
                if rows <= 0:
                    continue
                tensors["packed_depth_row_is_valid"][batch_ix, :rows] = True
                if example_data["updated_mask"] is not None and "packed_depth_updated_mask" in tensors:
                    tensors["packed_depth_updated_mask"][batch_ix, :rows] = torch.from_numpy(
                        example_data["updated_mask"]
                    ).to(torch.bool)
                if example_data["buffer_codes"] is not None and "packed_depth_buffer_codes" in tensors:
                    tensors["packed_depth_buffer_codes"][batch_ix, :rows] = torch.from_numpy(
                        example_data["buffer_codes"]
                    ).to(torch.long)
                if example_data["example_ids"] is not None:
                    tensors["packed_depth_example_ids"][batch_ix, :rows] = torch.from_numpy(
                        example_data["example_ids"]
                    ).to(torch.long)
            return tensors

        chunk_tensors = _collate_action_chunks()
        if chunk_tensors is not None:
            out.update(chunk_tensors)
        else:
            _stack_optional_dense("states", torch.float32, numpy_dtype=np.float32)
            _stack_optional_dense("actions", torch.float32, numpy_dtype=np.float32)
            _stack_optional_dense("action_horizon_is_pad", torch.bool, numpy_dtype=np.bool_, legacy_key="action_is_pad")
            _stack_optional_dense("action_dim_is_pad", torch.bool, numpy_dtype=np.bool_)
        packed_depth_tensors = _collate_packed_depth_side_channels()
        if packed_depth_tensors is not None:
            out.update(packed_depth_tensors)
        else:
            _stack_optional_dense("depth_updated_mask", torch.bool, numpy_dtype=np.bool_)
            _stack_optional_dense("depth_buffer_codes", torch.long, numpy_dtype=np.int64)

        out["input_ids"] = out.pop("input_tokens")
        if "target_tokens" in out:
            out["labels"] = out.pop("target_tokens")

        # Maybe add metdata or worker state
        if "data_worker_state" in batch[0]:
            out["data_worker_state"] = [ex["data_worker_state"] for ex in batch]
        if self.include_metadata:
            out["metadata"] = [ex.get("metadata", {}) for ex in batch]
        return out

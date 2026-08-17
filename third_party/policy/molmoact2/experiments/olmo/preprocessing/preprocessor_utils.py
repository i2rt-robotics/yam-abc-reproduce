import dataclasses
from collections import namedtuple
from typing import Optional, Dict, Tuple, List

import numpy as np
from einops import einops

from olmo.util import get_all_keys


TOKEN_POOLING_KEYS = ["token_pooling", "low_res_token_pooling"]


@dataclasses.dataclass
class TokenizedVisionData:
    """Data for visual input that has been converted into tokens"""

    tokens: np.ndarray
    """Token ids to use in the LLM"""

    images: np.ndarray
    """Images in [n_images, n_patches, patch_size] format"""

    image_masks: Optional[np.ndarray] = None
    """Images in [n_images, n_patches] format"""

    token_pooling: Optional[np.ndarray] = None
    """[n_visual_tokens, pooling_dim] array of of how to pool ViT patches for patch_id tokens"""

    low_res_token_pooling: Optional[np.ndarray] = None
    """[n_visual_tokens, pooling_dim] array of of how to pool ViT patches for low_res tokens"""

    position_ids: Optional[np.ndarray] = None
    """Position id for each token if not sequential"""

    other_data: Optional[Dict[str, np.ndarray]] = None
    """Any extra data that needs to be passed to the model"""

    token_mapping: Optional[np.ndarray] = None
    """
    Maps patches in the originally shared visual inputs into `tokens`
    """


@dataclasses.dataclass
class TensorSpec:
    """The expected shape and dtype of a tensor"""
    shape: Tuple[int]
    dtype: np.dtype

    def __post_init__(self):
        assert all(x >= 0 for x in self.shape if x is not None)

    def extend(self, n, dim=0):
        shape = list(self.shape)
        shape[dim] += n
        return TensorSpec(tuple(shape), self.dtype)

    def __mul__(self, other):
        assert isinstance(other, int)
        return TensorSpec(tuple(s*other for s in self.shape), dtype=self.dtype)

    @classmethod
    def build(cls, data):
        if data is None:
            return None
        return TensorSpec(data.shape, data.dtype)

    @classmethod
    def get_spec(cls, src: TokenizedVisionData) -> Dict[str, 'TensorSpec']:
        spec = {} if src.other_data is None else src.other_data
        for k in ["tokens", "images", "image_masks", "token_pooling", "low_res_token_pooling"]:
            v = getattr(src, k)
            if v is not None:
                spec[k] = TensorSpec.build(getattr(src, k))
        return spec

    @staticmethod
    def max(*other: 'TensorSpec'):
        other = [o for o in other if o is not None]
        if not other:
            return None
        rank = len(other[0].shape)
        dtype = other[0].dtype
        assert all(rank == len(o.shape) for o in other)
        assert all(dtype == o.dtype for o in other)
        return TensorSpec(
            [max([o.shape[i] for o in other]) for i in range(rank)],
            dtype
        )

    def max_dictionaries(*other: Dict[str, 'TensorSpec']):
        return {k: TensorSpec.max(*(o[k] for o in other if k in o)) for k in get_all_keys(other)}


@dataclasses.dataclass
class VariablePaddingSpec:
    """Indicates a key can have a variable shape after collation"""

    shape: Tuple[int]
    """Shape of padding tensor if key is not in batch and padding is required"""

    dtype: np.dtype
    """dtype of padding tensor if the key is not in batch and padding is required"""


def batch_pixels_to_patches(array, patch_size):
    """Reshape images of [n_images, h, w, 3] -> [n_images, n_patches, pixels_per_patch]"""
    if len(array.shape) == 3:
        n_crops, h, w = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size])
        array = np.transpose(array, [0, 1, 3, 2, 4])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size])
        return array
    else:
        n_crops, h, w, c = array.shape
        h_patches = h//patch_size
        w_patches = w//patch_size
        array = np.reshape(array, [n_crops, h_patches, patch_size, w_patches, patch_size, c])
        array = np.transpose(array, [0, 1, 3, 2, 4, 5])
        array = np.reshape(array, [n_crops, h_patches*w_patches, patch_size*patch_size*c])
        return array


def arange_for_pooling(idx_arr, pool_h, pool_w):
    h_pad = pool_h * ((idx_arr.shape[0] + pool_h - 1) // pool_h) - idx_arr.shape[0]
    w_pad = pool_w * ((idx_arr.shape[1] + pool_w - 1) // pool_w) - idx_arr.shape[1]
    idx_arr = np.pad(idx_arr, [[h_pad//2, (h_pad+1)//2], [w_pad//2, (w_pad+1)//2]],
                     mode='constant',constant_values=-1)
    return einops.rearrange(
        idx_arr, "(h dh) (w dw) -> h w (dh dw)", dh=pool_h, dw=pool_w)


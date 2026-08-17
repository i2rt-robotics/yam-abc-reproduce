# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import contextlib
from collections.abc import Generator, Iterable
from typing import (
    Optional,
    Union,
    List,
    Dict,
    Any,
    TypeVar,
    Optional,
    Tuple
)

from dataclasses import dataclass

import torch
from torch import nn
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed._composable.replicate import replicate
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.tensor import DTensor
import torch.distributed as dist

from olmo.config import StrEnum, BaseConfig
from olmo.config import DType, StrEnum
from olmo.exceptions import OLMoConfigurationError
from olmo.torch_util import get_num_nodes

log = logging.getLogger(__name__)

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "amp_bf16": torch.bfloat16,
}

__all__ = ["ParallelDims"]


def get_device_info() -> tuple[str, torch.device]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


def get_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.to_local()
        # An `AsyncCollectiveTensor` might be returned, which means the local tensor is not ready
        # yet (i.e. communication is not finished). In this case we need to call `.wait()`
        # to wait the local tensor to be ready.
        if hasattr(x, "wait"):
            return x.wait()  # type: ignore
        else:
            return x
    else:
        return x


def get_device_mesh_info(device_mesh: DeviceMesh) -> str:
    """
    Get a human-readable string representation of a ``DeviceMesh``.

    :param device_mesh: The device mesh to get info for.
    """
    shape: str
    if device_mesh.mesh_dim_names is not None:
        shape = ", ".join(
            f"{dim_name}={d}" for dim_name, d in zip(device_mesh.mesh_dim_names, device_mesh.shape)
        )
    else:
        shape = ", ".join(f"{d}" for d in device_mesh.shape)
    if device_mesh.ndim == 1:
        shape += ","
    return f"{device_mesh.ndim}D device mesh with shape ({shape})"


device_type, device_module = get_device_info()


def _check_num_replicas(num_replicas: int, dp_world_size: int) -> int:
    if dp_world_size % num_replicas != 0:
        # raise OLMoConfigurationError(
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'num_replicas' ({num_replicas})"
        )
    return num_replicas


def _check_shard_degree(shard_degree: int, dp_world_size: int) -> int:
    if dp_world_size % shard_degree != 0:
        raise OLMoConfigurationError(
            f"data parallel world size ({dp_world_size}) must be "
            f"divisible by 'shard_degree' ({shard_degree})"
        )
    return shard_degree


def _get_model_mesh(device_mesh: DeviceMesh) -> Tuple[DeviceMesh, Tuple[str, ...]]:
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP model sub-mesh without dimension names")

    # Expert parallel dims get flattened into a DP dimension.
    if MeshDimName.dp in dim_names and MeshDimName.ep in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.dp,
            MeshDimName.ep,
            name=MeshDimName.dp_ep,
            dim_names=dim_names,
        )
    elif MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.ep_replicate,
            MeshDimName.ep_shard,
            name=MeshDimName.dp,
            dim_names=dim_names,
        )

    # Context parallel dimension gets flattened into the adjacent DP dimension.
    # NOTE: We do this because for param-synchronization purposes a CP group behaves like an extra
    # DP replica set. CP splits the context across ranks but every CP rank still holds a copy of
    # the model parameters. Gradients need to be reduced across the union of DP ranks and CP ranks.
    if MeshDimName.cp in dim_names:
        last_dp_dim = dim_names[dim_names.index(MeshDimName.cp) - 1]
        assert last_dp_dim.startswith("dp")
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            last_dp_dim,
            MeshDimName.cp,
            name=MeshDimName.dp_cp,
            dim_names=dim_names,
        )

    return device_mesh, dim_names


def get_dp_model_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the right sub-mesh for a data parallel model wrapper like FSDP or DDP from a ``DeviceMesh``
    created by :func:`build_world_mesh()`.

    .. important::
        You should use :func:`get_dp_mesh()` instead for getting the sub-mesh to assign ranks
        to data loading workers. In many cases these two functions will return the same result,
        but there are cases where they could be different.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    device_mesh, dim_names = _get_model_mesh(device_mesh)
    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


class DataParallelType(StrEnum):
    fsdp = "fsdp"
    hsdp = "hsdp"
    ddp = "ddp"


class DPMeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimension names for data parallelism.
    """

    replicate = "dp_replicate"
    """
    The device mesh dimension over which the model is replicated.
    """
    shard = "dp_shard"
    """
    The device mesh dimension over which the model is sharded.
    """


class MeshDimName(StrEnum):
    """
    ``DeviceMesh`` dimensions names for different forms of parallelism.
    This are the dimension names that you will find in the mesh created by :func:`build_world_mesh()`.
    """

    dp = "dp"
    """
    Data parallel (DP).
    """

    dp_replicate = DPMeshDimName.replicate
    """
    The DP dimension over which the model is replicated.
    """

    dp_shard = DPMeshDimName.shard
    """
    The DP dimension over which the model is sharded.
    """

    tp = "tp"
    """
    Tensor parallel (TP).
    """

    cp = "cp"
    """
    Context parallel (CP).
    """

    pp = "pp"
    """
    Pipeline parallel (PP).
    """

    ep = "ep"
    """
    Expert parallel (EP).
    """

    ep_replicate = "ep_replicate"
    ep_shard = "ep_shard"

    dp_ep = "dp_ep"
    dp_cp = "dp_cp"


_WORLD_MESH: Optional[DeviceMesh] = None


def flatten_mesh(device_mesh: DeviceMesh, name: Optional[str] = None) -> DeviceMesh:
    """
    Flatten a multi-dimensional ``DeviceMesh`` into a 1D ``DeviceMesh``.

    :param device_mesh: The multi-dimensional ``DeviceMesh`` to flatten.
    :param name: Optional name for the flattened dimension.

    .. important::
        The ``device_mesh`` is modified in-place.
    """
    return device_mesh._flatten(mesh_dim_name=name)


def _flatten_dims(
    device_mesh: DeviceMesh,
    *dims: str,
    name: Optional[str] = None,
    dim_names: Optional[Tuple[str, ...]] = None,
) -> Tuple[DeviceMesh, Tuple[str, ...]]:
    """
    Flatten *dims* into a single dimension called *name*.

    :param device_mesh: The world-mesh object. Only views of *device_mesh* are actually mutated.
    :param dims: The existing dimension names to merge.
    :param name: New dimension name. If ``None`` we join *dims* with "_".
    :param dim_names: Optional cached list of current dimension names. Supplying this avoids
        relying on ``device_mesh.mesh_dim_names`` (which is stale after a prior
        flatten) and therefore allows chaining multiple flatten operations.

    :returns: The root mesh (now indexable by the new dimension names
        as well as the original names) and the new dimension names.
    """
    if name is None:
        name = "_".join(dims)

    curr_names = list(dim_names or device_mesh.mesh_dim_names or [])
    if not curr_names:
        raise RuntimeError("Could not determine current dimension names for flattening")

    log.info(f"Flattening mesh dimensions {dims} into {name}")

    out_names: list[str] = []
    for n in curr_names:
        if n in dims:
            if name not in out_names:
                out_names.append(name)
        else:
            out_names.append(n)

    flatten_mesh(device_mesh[dims], name)  # in-place flatten on sub-mesh
    new_names = tuple(out_names)

    try:
        # NOTE: device_mesh.mesh_dim_names is not updated based on the flatten operation.
        # We need to check that the root mesh is indexable by the new dimension names.
        _ = device_mesh[new_names]
    except KeyError as exc:
        raise RuntimeError(
            "Flattening failed: root device mesh does not recognize the new "
            f"dimension names {new_names}. Original dims: {dims}."
        ) from exc

    return device_mesh, new_names


def get_world_mesh() -> Optional[DeviceMesh]:
    """
    Get the global world mesh built with :meth:`build_world_mesh()`.
    """
    global _WORLD_MESH
    return _WORLD_MESH


def get_world_mesh() -> Optional[DeviceMesh]:
    """
    Get the global world mesh built with :meth:`build_world_mesh()`.
    """
    global _WORLD_MESH
    return _WORLD_MESH


def get_cp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the context parallel sub-mesh associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    if device_mesh.mesh_dim_names is None:
        raise RuntimeError("could not determine context parallel sub-mesh without dimension names")

    if MeshDimName.cp in device_mesh.mesh_dim_names:
        return device_mesh[MeshDimName.cp]
    else:
        raise RuntimeError(
            f"could not determine context parallel sub-mesh from mesh with dimensions {device_mesh.mesh_dim_names}"
        )


def get_dp_mesh(device_mesh: DeviceMesh) -> DeviceMesh:
    """
    Get the data parallel sub-mesh associated from a ``DeviceMesh`` created by :func:`build_world_mesh()`.

    .. important::
        This is the mesh that should be used to assign ranks to data loading workers,
        however you should use :func:`get_dp_model_mesh()` to get the mesh for DDP/FSDP.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    if (dim_names := device_mesh.mesh_dim_names) is None:
        raise RuntimeError("could not determine DP sub-mesh without dimension names")

    # Expert parallel dims get flattened into DP dimension since ranks within each EP group
    # should receive different data instances.
    if MeshDimName.dp in dim_names and MeshDimName.ep in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.dp,
            MeshDimName.ep,
            name=MeshDimName.dp_ep,
            dim_names=dim_names,
        )
    elif MeshDimName.ep_replicate in dim_names and MeshDimName.ep_shard in dim_names:
        device_mesh, dim_names = _flatten_dims(
            device_mesh,
            MeshDimName.ep_replicate,
            MeshDimName.ep_shard,
            name=MeshDimName.dp,
            dim_names=dim_names,
        )

    # Flattened context parallel dimensions should not be in this mesh since ranks within the
    # same CP group should receive the same data instances.
    if MeshDimName.dp_cp in dim_names:
        raise RuntimeError("'get_dp_mesh' should be called on the original world mesh")

    dp_dim_names = tuple(name for name in dim_names if name.startswith("dp"))
    return device_mesh[dp_dim_names]


def get_default_device() -> torch.device:
    """
    Get the default device.
    """
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def is_distributed() -> bool:
    """
    Check if in a distributed context.
    """
    return dist.is_available() and dist.is_initialized()


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    """
    Get the world size of the default distributed process group.

    .. warning::
        This will always return 1 if a distributed group has not been initialized.
    """
    if is_distributed():
        return dist.get_world_size(group)
    else:
        return 1


def get_dp_process_group(device_mesh: DeviceMesh):
    """
    Get the data parallel process group associated with a ``DeviceMesh``
    created from :func:`build_world_mesh()`.

    Like :func:`get_dp_mesh()`, this should be used for data loading, but not necessarily for
    data parallel model wrappers.

    :param device_mesh: The world mesh created by :func:`build_world_mesh()`.
    """
    dp_mesh = get_dp_mesh(device_mesh)
    if len(dp_mesh.shape) > 1:
        return dp_mesh._flatten(mesh_dim_name=MeshDimName.dp).get_group()
    else:
        return dp_mesh.get_group()


def build_world_mesh(
    *,
    dp = None, tp = None, cp = None,
    pp = None, ep = None, device_type: Optional[str] = None,
) -> DeviceMesh:
    """
    Build a :class:`~torch.distributed.device_mesh.DeviceMesh` suitable for the given parallel strategies.

    .. seealso::
        Pass the mesh created by this function to any of the ``get_*_mesh()`` functions in
        this module to get the right sub-mesh for a any given parallel strategy.

        - :func:`get_dp_model_mesh()` gives you the 1 or 2D sub-mesh suitable for data parallel *model*
          wrappers like FSDP(2) or DDP.
        - :func:`get_dp_mesh()` gives you the 1D sub-mesh suitable for configuring *data loaders*.
        - :func:`get_tp_mesh()` gives you the 1D sub-mesh for tensor parallelism.
        - :func:`get_cp_mesh()` gives you the 1D sub-mesh for context parallelism.
        - :func:`get_pp_mesh()` gives you the 1D sub-mesh for pipeline parallelism.
        - :func:`get_ep_mesh()` gives you the 1D sub-mesh for expert parallelism.

    .. important::
        A data parallel config is required if any other parallel config is set.

    .. important::
        Not all parallel strategies are compatible with each other.

    :param dp: Data parallel config.
    :param tp: Tensor parallel config.
    :param cp: Context parallel config.
    :param pp: Pipeline parallel config.
    :param ep: Expert parallel config.
    :param device_type: The device type.

    :returns: The world mesh with a shape compatible with the given parallel configs.
    """
    global _WORLD_MESH

    if _WORLD_MESH is not None:
        raise RuntimeError("world mesh already exists! You can only call 'build_world_mesh' once!")

    device_type = device_type or get_default_device().type
    dp_world_size = get_world_size()

    if pp is None and tp is None and cp is None and dp is None and ep is None:
        return init_device_mesh(device_type, (dp_world_size,), mesh_dim_names=(MeshDimName.dp,))

    if dp is None:
        raise OLMoConfigurationError(
            "Data parallel config is required in addition to expert/tensor/context/pipeline parallel configs"
        )

    # Validate parallelism degrees while adjust the DP degree.
    if pp is not None:
        if pp.degree < 1 or dp_world_size % pp.degree != 0:
            raise OLMoConfigurationError(
                f"{pp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= pp.degree
    
    if cp is not None:
        if cp.degree < 1 or dp_world_size % cp.degree != 0:
            raise OLMoConfigurationError(
                f"{cp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= cp.degree
    if tp is not None:
        if tp.degree < 1 or dp_world_size % tp.degree != 0:
            raise OLMoConfigurationError(
                f"{tp.__class__.__name__}.degree must be at least 1 and divide into the world size"
            )
        dp_world_size //= tp.degree

    if ep is not None:
        raise NotImplementedError("Expert Parallelism is not supported yet.")

    # Build up mesh dimensions.
    names: List[str] = []
    dims: List[int] = []

    # Then data parallel.
    if dp.name == DataParallelType.hsdp:
        num_replicas, shard_degree = dp.get_replicate_and_shard_degree(dp_world_size)
        names.append(MeshDimName.dp_replicate)
        dims.append(num_replicas)
        names.append(MeshDimName.dp_shard)
        dims.append(shard_degree)

    else:
        names.append(MeshDimName.dp)
        dims.append(dp_world_size)

    # Context parallel.
    if cp is not None:
        names.append(MeshDimName.cp)
        dims.append(cp.degree)

    # And lastly tensor parallel.
    if tp is not None:
        names.append(MeshDimName.tp)
        dims.append(tp.degree)

    mesh = init_device_mesh(device_type, tuple(dims), mesh_dim_names=tuple(names))
    log.info(f"Built {get_device_mesh_info(mesh)}")

    # Ensure data parallel process group is created here.
    get_dp_process_group(mesh)

    _WORLD_MESH = mesh

    return mesh


def parallelize_model(
    model,
    *,
    world_mesh: Optional[DeviceMesh],
    compile_model: bool = False,
    float8_config = None,
    dp_config = None, tp_config = None, cp_config = None,
    ep_config = None, pp_enabled = False,
):
    model_parts = [model]

    assert not pp_enabled, "Pipeline Parallelism is not supported yet."
    assert ep_config is None, "Expert Parallelism is not supported yet."

    if float8_config is not None and float8_config.enabled:
        raise NotImplementedError("Float8 is not supported yet.")

    # Maybe apply context parallelism.
    if cp_config is not None:
        assert world_mesh is not None
        cp_mesh = get_cp_mesh(world_mesh)
        for m in model_parts:
            m.apply_cp(
                cp_mesh, 
                load_balancer=cp_config.load_balancer, 
                head_stride=cp_config.head_stride,
                attention_type=cp_config.attention_type,
            )
        log.info(f"Applied context parallelism to the model with {get_device_mesh_info(cp_mesh)}")

    # Maybe shard/replicate according to data parallel config.
    if dp_config is not None:
        assert world_mesh is not None
        dp_mesh = get_dp_model_mesh(world_mesh)
        param_dtype = dp_config.param_dtype.as_pt() if dp_config.param_dtype is not None else None
        if dp_config.name in (DataParallelType.fsdp, DataParallelType.hsdp):
            for m in model_parts:
                m.apply_fsdp2_v2(
                    dp_mesh=dp_mesh,
                    param_dtype=param_dtype,
                    reduce_dtype=dp_config.reduce_dtype.as_pt(),
                    wrapping_strategy=dp_config.wrapping_strategy,
                    pp_enabled=pp_enabled,
                    prefetch_factor=dp_config.prefetch_factor,
                )

            log.info(f"Applied FSDP to the model with {get_device_mesh_info(dp_mesh)}")
        elif dp_config.name == DataParallelType.ddp:
            for m in model_parts:
                m.apply_ddp(dp_mesh=dp_mesh, compile_enabled=compile_model, param_dtype=param_dtype)
            log.info(f"Applied DDP to the model with {get_device_mesh_info(dp_mesh)}")
        else:
            raise NotImplementedError(dp_config.name)

    return model
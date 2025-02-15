# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3

# pyre-strict

import abc
from enum import Enum, unique
from typing import Any, List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner

try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass

try:
    from hammer.ops.triton.utils import triton_autotune
    from hammer.utils import is_dev_mode, set_dev_mode, set_verbose_level
except ImportError:
    # pyre-ignore
    def triton_autotune(
        configs: List[triton.Config],
        key: List[str],
        # pyre-ignore
        prune_configs_by=None,
        # pyre-ignore
        reset_to_zero=None,
        # pyre-ignore
        restore_value=None,
        warmup: int = 25,
        rep: int = 100,
    ):
        # pyre-ignore
        def decorator(fn):
            return Autotuner(
                fn,
                fn.arg_names,
                configs,
                key,
                reset_to_zero,
                restore_value,
                pre_hook=None,
                post_hook=None,
                prune_configs_by=prune_configs_by,
                warmup=warmup,
                rep=rep,
            )

        return decorator

    DEV_MODE: bool = False
    VERBOSE_LEVEL: int = 0

    def set_dev_mode(val: bool) -> None:
        global DEV_MODE
        DEV_MODE = val

    def is_dev_mode() -> bool:
        global DEV_MODE
        return DEV_MODE

    def set_verbose_level(level: int) -> None:
        global VERBOSE_LEVEL
        VERBOSE_LEVEL = level

    def get_verbose_level() -> int:
        global VERBOSE_LEVEL
        return VERBOSE_LEVEL


@unique
class HammerKernel(Enum):
    TRITON = "TRITON"
    PYTORCH = "PYTORCH"
    CUDA = "CUDA"
    TRITON_CC = "TRITON_CC"


class HammerModule(torch.nn.Module, abc.ABC):
    _is_inference: bool
    _use_triton_cc: bool
    _hammer_kernel: Optional[HammerKernel] = None

    def __init__(
        self,
        is_inference: bool,
        use_triton_cc: bool = False,
        hammer_kernel: Optional[HammerKernel] = None,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._hammer_kernel = hammer_kernel
        self._use_triton_cc = use_triton_cc

    def hammer_kernel(self) -> HammerKernel:
        kernel = self._hammer_kernel
        if kernel is not None:
            return kernel
        if self._is_inference and self._use_triton_cc:
            return HammerKernel.TRITON_CC
        else:
            return HammerKernel.TRITON

    # pyre-ignore[2]
    def recursive_setattr(self, name: str, value: Any) -> None:
        for _, module in self.named_modules():
            if hasattr(module, name):
                setattr(module, name, value)

    @property
    def is_inference(self) -> bool:
        return self._is_inference

    @property
    def is_eval(self) -> bool:
        return (not self._is_inference) and (not self.training)

    @property
    def is_train(self) -> bool:
        return (not self._is_inference) and self.training


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )


def apply_sampling(
    lengths: torch.Tensor,
    alpha: float,
    max_seq_len: int,
) -> torch.Tensor:
    threshold = int(max_seq_len ** (alpha / 2))
    no_sample_prob = (max_seq_len**alpha) / torch.pow(lengths, 2)
    users_to_sample = torch.logical_and(
        lengths > threshold,
        torch.rand_like(no_sample_prob) < 1 - no_sample_prob,
    )
    lengths = torch.where(users_to_sample, threshold, lengths)
    return lengths


nv_gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
nv_gpu_available: bool = not nv_gpu_unavailable[0]


amd_gpu_unavailable: Tuple[bool, str] = (
    not torch.version.hip,
    "AMD HIP not available or no GPUs detected",
)
amd_gpu_available: bool = not amd_gpu_unavailable[0]

gpu_unavailable: Tuple[bool, str] = (
    not nv_gpu_available and not amd_gpu_available,
    "CUDA/HIP is not available or no GPUs detected",
)

gpu_available: bool = not gpu_unavailable[0]


def switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range (0, 10**9)
        torch._check(x.size(0) > 0)
        torch._check(x.size(0) < 10**9)
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@torch.fx.wrap
def prev_power_of_2(x: int) -> int:
    if torch.compiler.is_compiling():
        # Re-write to make Dynamo happy
        x_tensor = torch.scalar_tensor(x, dtype=torch.int64)  # type: ignore[arg-type]
        x_tensor_orig = x_tensor.clone()
        out = triton.next_power_of_2(x_tensor)  # type: ignore[arg-type]
        return int(torch.where(torch.lt(x_tensor_orig, out), out // 2, out).item())  # type: ignore[return-value]
    else:
        out = triton.next_power_of_2(x)
        return out // 2 if out > x else out


STATIC_MAX_SEQ_LEN = -1
L2_STATIC_MAX_SEQ_LEN = -1
USE_RUNTIME_MAX_SEQ_LEN = True


def set_static_max_seq_lens(max_seq_len: int, l2_max_seq_len: int) -> None:
    global STATIC_MAX_SEQ_LEN
    global L2_STATIC_MAX_SEQ_LEN
    STATIC_MAX_SEQ_LEN = max_seq_len
    L2_STATIC_MAX_SEQ_LEN = l2_max_seq_len


def get_static_max_seq_lens() -> Tuple[int, int]:
    return STATIC_MAX_SEQ_LEN, L2_STATIC_MAX_SEQ_LEN


def set_use_runtime_max_seq_len(use_runtime_max_seq_len: bool) -> None:
    global USE_RUNTIME_MAX_SEQ_LEN
    USE_RUNTIME_MAX_SEQ_LEN = use_runtime_max_seq_len


def use_runtime_max_seq_len() -> bool:
    return USE_RUNTIME_MAX_SEQ_LEN


def autotune_max_seq_len(runtime_max_seq_len: int) -> int:
    if use_runtime_max_seq_len():
        return prev_power_of_2(runtime_max_seq_len)
    else:
        max_seq_len, l2_max_seq_len = get_static_max_seq_lens()
        assert (
            max_seq_len > 0 and l2_max_seq_len > 0
        ), "max_seq_len and l2_max_seq_len must be greater than 0"
        return l2_max_seq_len if runtime_max_seq_len <= l2_max_seq_len else max_seq_len


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(len, device=device)


@torch.fx.wrap
def fx_infer_max_len(
    lengths: torch.Tensor,
) -> int:
    max_len = int(lengths.max().item())
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        # Tell Dynamo this data-dependent value is in the range [0, 10**9)
        torch._check_is_size(max_len)
        torch._check(max_len < 10**9)
        torch._check(max_len > 0)
    return max_len


@torch.fx.wrap
def fx_mark_length_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.fx.wrap
def fx_torch_ones(
    shape: List[int],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ones(shape, device=device, dtype=dtype)


@torch.fx.wrap
def fx_torch_zeros(shape: List[int], device: torch.device) -> torch.Tensor:
    return torch.zeros(shape, device=device)


@torch.fx.wrap
def jagged_to_padded_dense(
    values: torch.Tensor,
    offsets: List[torch.Tensor],
    max_lengths: List[int],
    padding_value: float,
) -> torch.Tensor:
    return torch.ops.fbgemm.jagged_to_padded_dense(
        values=values,
        offsets=offsets,
        max_lengths=max_lengths,
        padding_value=padding_value,
    )


@torch.fx.wrap
def dense_to_jagged(
    dense: torch.Tensor,
    x_offsets: List[torch.Tensor],
) -> torch.Tensor:
    return torch.ops.fbgemm.dense_to_jagged(
        dense=dense,
        x_offsets=x_offsets,
    )[0]

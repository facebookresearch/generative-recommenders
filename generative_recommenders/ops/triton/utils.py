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
import dataclasses
import logging

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
from triton.runtime.autotuner import Autotuner

logger: logging.Logger = logging.getLogger(__name__)

SpecType = Union[Tuple[str, int], Tuple[str, int, bool], int, str]
NamedSpecType = Dict[str, SpecType]


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
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


@dataclass
class VersionedSpec:
    """
    spec: a dict that maps each argument name to a spec
    version: the version of the spec
    """

    spec: NamedSpecType = dataclasses.field(default_factory=dict)
    version: str = ""
    default_values: Dict[str, Any] = dataclasses.field(default_factory=dict)


# pyre-ignore[2,3]
def register_tritoncc_specs(func, versioned_specs):
    return func


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
            prune_configs_by,
            warmup,
            rep,
        )

    return decorator


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

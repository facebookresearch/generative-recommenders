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


from typing import List, Optional, Tuple

import torch
from generative_recommenders.ops.pytorch.pt_layer_norm import (
    pytorch_layer_norm,
    pytorch_swish_layer_norm,
)
from generative_recommenders.ops.triton.triton_layer_norm import triton_rms_norm

try:
    from hammer.ops.triton.cc.swish_layer_norm.triton_cc_swish_layer_norm import (
        triton_cc_swish_layer_norm,
    )
except ImportError:
    pass
from generative_recommenders.common import HammerKernel, HammerModule
from generative_recommenders.ops.triton.triton_layer_norm import (
    triton_layer_norm,
    triton_swish_layer_norm,
)
from torch.fx._symbolic_trace import is_fx_tracing


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    if kernel == HammerKernel.TRITON:
        if not is_fx_tracing():
            torch._assert(x.is_cuda, "x must be CUDA tensor")
            if weight is not None:
                torch._assert(weight.is_cuda, "weight must be CUDA tensor")
            if bias is not None:
                torch._assert(bias.is_cuda, "bias must be CUDA tensor")
        return triton_layer_norm(x, weight, bias, eps)
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_swish_layer_norm(
            x,
            weight,
            bias,
            eps,
            is_swish=False,
        )
    else:
        return torch.nn.functional.layer_norm(x, (x.shape[1],), weight, bias, eps)


class LayerNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self.normalized_shape: List[int] = [dim]
        self.eps = eps
        self.weight = torch.nn.Parameter(
            torch.ones(self.normalized_shape, dtype=dtype),
        )
        self.bias = torch.nn.Parameter(
            torch.zeros(self.normalized_shape, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hammer_kernel() == HammerKernel.TRITON:
            return triton_layer_norm(x, self.weight, self.bias, self.eps)
        elif self.hammer_kernel() == HammerKernel.TRITON_CC:
            return triton_cc_swish_layer_norm(
                x,
                self.weight,
                self.bias,
                self.eps,
                is_swish=False,
            )
        else:
            return pytorch_layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )


class RMSNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self.normalized_shape: Tuple[int, ...] = (dim,)
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=dtype))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.hammer_kernel() == HammerKernel.TRITON:
            return triton_rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight


class SwishLayerNorm(HammerModule):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        dtype: torch.dtype = torch.float32,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self.weight = torch.nn.Parameter(
            torch.ones(self._normalized_shape, dtype=dtype)
        )
        self.bias = torch.nn.Parameter(torch.zeros(self._normalized_shape, dtype=dtype))
        self._eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.hammer_kernel() == HammerKernel.TRITON:
            return triton_swish_layer_norm(
                x, self._normalized_shape, self.weight, self.bias, self._eps
            )
        elif self.hammer_kernel() == HammerKernel.TRITON_CC:
            return triton_cc_swish_layer_norm(
                x,
                self.weight,
                self.bias,
                self._eps,
                is_swish=True,
            )
        else:
            return pytorch_swish_layer_norm(
                x, self._normalized_shape, self.weight, self.bias, self._eps
            )

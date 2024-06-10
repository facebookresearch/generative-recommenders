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

# pyre-unsafe

from typing import Optional

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl


def _switch_to_contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.stride(-1) == 1:
        return x
    return x.contiguous()


@triton.jit
def _group_norm_fwd(
    X,
    Y,
    W,
    B,
    Mean,
    Rstd,
    D,
    C_per_group,
    Groups,
    eps,
    stride_xb,
    stride_xg,
    stride_yb,
    stride_yg,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_G: tl.constexpr,
    AFFINE: tl.constexpr,
    TRAINING: tl.constexpr,
):
    row = tl.program_id(0)
    X += row.to(tl.int64) * stride_xb
    Y += row.to(tl.int64) * stride_yb
    offs_d = tl.arange(0, BLOCK_D)
    offs_c = tl.arange(0, BLOCK_C)
    offs_g = tl.arange(0, BLOCK_G)
    offsets = (
        offs_g[:, None, None] * stride_xg
        + offs_c[None, :, None] * D
        + offs_d[None, None, :]
    )
    mask_g = offs_g < Groups
    mask_c = offs_c < C_per_group
    mask_d = offs_d < D
    mask = mask_d[None, None, :] & mask_c[None, :, None] & mask_g[:, None, None]

    # Compute mean
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=2)
    mean = tl.sum(mean, axis=1) / (D * C_per_group)
    mean = tl.ravel(mean)

    # Compute variance
    _var = tl.zeros([BLOCK_G, BLOCK_C, BLOCK_D], dtype=tl.float32)
    x_mean = tl.where(mask, x - mean[:, None, None], 0.0)
    _var += x_mean * x_mean
    var = tl.sum(_var, axis=2)
    var = tl.sum(var, axis=1) / (D * C_per_group)
    var = tl.ravel(var)
    rstd = 1 / tl.sqrt(var + eps)
    if TRAINING:
        tl.store(Mean + row * Groups + offs_g, mean, mask=mask_g)
        tl.store(Rstd + row * Groups + offs_g, rstd, mask=mask_g)

    y = x_mean * rstd[:, None, None]  # pyre-ignore [16]
    # Normalize and apply linear transformation
    if AFFINE:
        offs_channel = offs_g[:, None] * C_per_group + offs_c[None, :]
        mask_channel = mask_g[:, None] & mask_c[None, :]
        w = tl.load(W + offs_channel, mask=mask_channel).to(tl.float32)
        b = tl.load(B + offs_channel, mask=mask_channel).to(tl.float32)
        y = y * w[:, :, None] + b[:, :, None]

    # Write output
    offsets_y = (
        offs_g[:, None, None] * stride_yg
        + offs_c[None, :, None] * D
        + offs_d[None, None, :]
    )
    tl.store(Y + offsets_y, y.to(Y.dtype.element_ty), mask=mask)


@torch.fx.wrap
def triton_group_norm(
    x: torch.Tensor,
    num_groups: int,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    assert (
        x.dim() >= 2
    ), f"Expected at least 2 dimensions for input tensor but received {x.dim()}"
    N = x.shape[0]
    input_shape = x.shape
    x = x.reshape(N, num_groups, -1)
    x = _switch_to_contiguous_if_needed(x)
    num_channels = num_groups
    affine: bool = weight is not None
    if affine:
        assert bias is not None and weight is not None
        assert weight.dim() == 1
        assert bias.dim() == 1
        num_channels = bias.numel()
        assert (
            bias.numel() == num_channels
        ), "bias and weight should have the same size"
    channel_per_group = num_channels // num_groups
    linear_dim = x.shape[-1] // channel_per_group

    y = torch.empty_like(x)
    mean = torch.empty((N * num_groups,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((N * num_groups,), dtype=torch.float32, device=x.device)
    if N == 0:
        return y

    # Less than 64KB per feature: enqueue fused kernel
    MAX_FUSED_SIZE = 65536 // x.element_size()
    BLOCK_D = triton.next_power_of_2(linear_dim)
    BLOCK_C = triton.next_power_of_2(channel_per_group)
    BLOCK_G = triton.next_power_of_2(num_groups)
    if BLOCK_D * BLOCK_C * BLOCK_G > MAX_FUSED_SIZE:
        raise RuntimeError(
            "This group norm doesn't support num_channels * linear_dim >= 64KB."
        )

    num_warps = min(max(BLOCK_D * BLOCK_C * BLOCK_G // 256, 1), 8)

    _group_norm_fwd[(N,)](  # pyre-ignore [28]
        x,
        y,
        weight,
        bias,
        mean,
        rstd,
        linear_dim,
        channel_per_group,
        num_groups,
        eps,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        AFFINE=affine,
        TRAINING=True,
        BLOCK_D=BLOCK_D,
        BLOCK_C=BLOCK_C,
        BLOCK_G=BLOCK_G,
        num_warps=num_warps,
    )
    return y.reshape(input_shape)


class GroupNorm(torch.nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError("num_channels must be divisible by num_groups")
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight: torch.nn.Parameter = torch.nn.Parameter(
                torch.ones(
                    num_channels,
                )
            )
            self.bias: torch.nn.Parameter = torch.nn.Parameter(
                torch.zeros(
                    num_channels,
                )
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return triton_group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps
        )

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

from typing import List

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

def get_mm_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        block_mn_range = [32, 64, 128, 256]
        block_k_range = [32]
        num_warps_range = [2, 4]
        group_m_range = [8]
        matrix_instr_nonkdim_range = [16, 32]

        for block_m in block_mn_range:
            for block_n in block_mn_range:
                for block_k in block_k_range:
                    for num_warps in num_warps_range:
                        for group_m in group_m_range:
                            for matrix_instr_nonkdim in matrix_instr_nonkdim_range:
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": block_m,
                                            "BLOCK_N": block_n,
                                            "BLOCK_K": block_k,
                                            "GROUP_M": group_m,
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        },
                                        num_warps=num_warps,
                                    )
                                )
        return configs
    else:
        return [
            triton.Config(
                {
                    "BLOCK_M": 32,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 256,
                    "BLOCK_K": 64,
                    "GROUP_M": 8,
                },
                num_stages=3,
                num_warps=8,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 256,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 64,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 128,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 128,
                    "BLOCK_N": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_M": 64,
                    "BLOCK_N": 32,
                    "BLOCK_K": 32,
                    "GROUP_M": 8,
                },
                num_stages=5,
                num_warps=2,
            ),
        ]
        
@triton.autotune(
    configs=get_mm_configs(),
    key=["N", "K"],
)
@triton.jit
def _addmm_fwd(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wk,
    stride_wn,
    stride_ym,
    stride_yn,
    stride_zm,
    stride_zn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (pid_m * BLOCK_M + offs_m)[:, None] < M
    mask_n = (pid_n * BLOCK_N + offs_n)[None, :] < N
    x_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_xm
    x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_wn
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn)
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask_k = offs_k[None, :] < K - k * BLOCK_K
        x = tl.load(x_ptrs, mask=mask_k & mask_m, other=0.0)
        mask_k = offs_k[:, None] < K - k * BLOCK_K
        w = tl.load(w_ptrs, mask=mask_k & mask_n, other=0.0)
        accumulator += tl.dot(x, w, allow_tf32=ALLOW_TF32)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    z = accumulator.to(z_ptr.dtype.element_ty)
    y_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_ym
    y_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_yn
    y_ptrs = y_ptr + stride_ym * offs_m[:, None] + stride_yn * offs_n[None, :]
    z_mask = mask_m & mask_n
    y = tl.load(y_ptrs, mask=z_mask)
    z = z + y
    z_ptr += pid_m.to(tl.int64) * BLOCK_M * stride_zm
    z_ptr += pid_n.to(tl.int64) * BLOCK_N * stride_zn
    z_ptrs = z_ptr + stride_zm * offs_m[:, None] + stride_zn * offs_n[None, :]
    tl.store(z_ptrs, z, mask=z_mask)

class _AddMmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        M, K = x.shape
        KB, N = w.shape
        assert K == KB, f"incompatible dimensions {K}, {KB}"

        z = torch.empty((M, N), device=x.device, dtype=x.dtype)
        if M == 0 or N == 0:
            return z

        def grid(META):
            return (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

        _addmm_fwd[grid](
            x,
            w,
            y,
            z,
            M,
            N,
            K,
            x.stride(0),
            x.stride(1),
            w.stride(0),
            w.stride(1),
            y.stride(0),
            y.stride(1),
            z.stride(0),
            z.stride(1),
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )
        return z

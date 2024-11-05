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

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

try:
    from hammer.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        autotune_max_seq_len,
        triton_autotune,
        VersionedSpec,
        register_tritoncc_specs,
    )
except ImportError:
    from hammer.oss.generative_recommenders.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        autotune_max_seq_len,
        triton_autotune,
        VersionedSpec,
        register_tritoncc_specs,
    )


def _get_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [32, 64, 128]:
        for BLOCK_N in [32, 64, 128]:
            for num_stages in [2]:
                for num_warps in [2, 4]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": BLOCK_M, "BLOCK_N": BLOCK_N},
                            num_stages=num_stages,
                            num_warps=num_warps,
                        )
                    )
    return configs


@triton_autotune(
    configs=_get_configs(),
    key=["Z", "H", "N"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["N"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _jagged_bias_to_dense(
    Z,
    H,
    N,
    jg_offsets_ptr,
    jg2_offsets_ptr,
    jagged_ptr,
    dense_bias_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    seq_start = tl.load(jg_offsets_ptr + off_z)
    seq_end = tl.load(jg_offsets_ptr + off_z + 1)
    seq_len = seq_end - seq_start
    if start_m >= seq_len:
        return

    bias_start = tl.load(jg2_offsets_ptr + off_z) * H + off_h * seq_len * seq_len
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (offs_m < seq_len)[:, None]
    off_jg_bias = bias_start + offs_m[:, None] * seq_len + offs_n[None, :]
    jg_bias_ptrs = jagged_ptr + off_jg_bias
    off_d_bias = off_hz * N * N + offs_m[:, None] * N + offs_n[None, :]
    d_bias_ptrs = dense_bias_ptr + off_d_bias

    for start_n in range(0, seq_len, BLOCK_N):
        maxk_n = (offs_n < seq_len - start_n)[None, :]
        jg_bias = tl.load(
            jg_bias_ptrs + start_n,
            mask=mask_m & maxk_n,
            other=0.0,
        )
        tl.store(
            d_bias_ptrs + start_n,
            jg_bias,
            mask=mask_m & maxk_n,
        )


@triton_autotune(
    configs=_get_configs(),
    key=["Z", "H", "N"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["N"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
    }
)
@triton.jit
def _dense_bias_to_jagged(
    Z,
    H,
    N,
    jg_offsets_ptr,
    jg2_offsets_ptr,
    dense_bias_ptr,
    jagged_ptr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
):
    start_m = tl.program_id(0) * BLOCK_M
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    seq_start = tl.load(jg_offsets_ptr + off_z)
    seq_end = tl.load(jg_offsets_ptr + off_z + 1)
    seq_len = seq_end - seq_start
    if start_m >= seq_len:
        return

    bias_start = tl.load(jg2_offsets_ptr + off_z) * H + off_h * seq_len * seq_len
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mask_m = (offs_m < seq_len)[:, None]
    off_jg_bias = bias_start + offs_m[:, None] * seq_len + offs_n[None, :]
    jg_bias_ptrs = jagged_ptr + off_jg_bias
    off_d_bias = off_hz * N * N + offs_m[:, None] * N + offs_n[None, :]
    d_bias_ptrs = dense_bias_ptr + off_d_bias

    for start_n in range(0, seq_len, BLOCK_N):
        maxk_n = (offs_n < seq_len - start_n)[None, :]
        d_bias = tl.load(
            d_bias_ptrs + start_n,
            mask=mask_m & maxk_n,
            other=0.0,
        )
        tl.store(
            jg_bias_ptrs + start_n,
            d_bias,
            mask=mask_m & maxk_n,
        )


class _JaggedBiasToDenseFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        Z: int,
        H: int,
        N: int,
        jg_offsets: torch.Tensor,
        jg2_offsets: torch.Tensor,
        jagged: torch.Tensor,
    ):
        jg_offsets = jg_offsets.contiguous()
        jg2_offsets = jg2_offsets.contiguous()
        jagged = jagged.contiguous()
        dense_bias = torch.zeros((Z, H, N, N), device=jagged.device, dtype=jagged.dtype)
        grid = lambda meta: (  # noqa E731
            triton.cdiv(N, meta["BLOCK_M"]),
            Z * H,
        )
        _jagged_bias_to_dense[grid](
            Z,
            H,
            N,
            jg_offsets,
            jg2_offsets,
            jagged,
            dense_bias,
        )
        ctx.Z = Z
        ctx.H = H
        ctx.N = N
        ctx.save_for_backward(jg_offsets, jg2_offsets, jagged)

        return dense_bias

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_dense_bias: torch.Tensor) -> Tuple[
        None,
        None,
        None,
        None,
        None,
        torch.Tensor,
    ]:
        jg_offsets, jg2_offsets, jagged = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.N, meta["BLOCK_M"]),
            ctx.Z * ctx.H,
        )
        _dense_bias_to_jagged[grid](
            ctx.Z,
            ctx.H,
            ctx.N,
            jg_offsets,
            jg2_offsets,
            d_dense_bias,
            d_jagged,
        )

        return None, None, None, None, None, d_jagged


class _DenseBiasToJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        jg_offsets: torch.Tensor,
        jg2_offsets: torch.Tensor,
        dense_bias: torch.Tensor,
    ):
        Z, H, N, _ = dense_bias.shape
        jg_offsets = jg_offsets.contiguous()
        jg2_offsets = jg2_offsets.contiguous()
        dense_bias = dense_bias.contiguous()
        jagged_size = int(jg2_offsets[-1].item()) * H
        jagged = torch.empty(
            (jagged_size,), dtype=dense_bias.dtype, device=dense_bias.device
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(N, meta["BLOCK_M"]),
            Z * H,
        )
        _dense_bias_to_jagged[grid](
            Z,
            H,
            N,
            jg_offsets,
            jg2_offsets,
            dense_bias,
            jagged,
        )

        ctx.Z = Z
        ctx.H = H
        ctx.N = N
        ctx.save_for_backward(jg_offsets, jg2_offsets, dense_bias)

        return jagged

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, d_jagged: torch.Tensor) -> Tuple[
        None,
        None,
        torch.Tensor,
    ]:
        jg_offsets, jg2_offsets, dense_bias = ctx.saved_tensors
        d_dense_bias = torch.empty_like(dense_bias)
        d_dense_bias = torch.zeros(
            (ctx.Z, ctx.H, ctx.N, ctx.N),
            device=d_jagged.device,
            dtype=d_dense_bias.dtype,
        )
        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.N, meta["BLOCK_M"]),
            ctx.Z * ctx.H,
        )
        _jagged_bias_to_dense[grid](
            ctx.Z,
            ctx.H,
            ctx.N,
            jg_offsets,
            jg2_offsets,
            d_jagged,
            d_dense_bias,
        )

        return None, None, d_dense_bias


def _get_bmm_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for BLOCK_K in [32, 64]:
                for num_stages in [2, 3]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


def _get_bmm_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    dtype: str = "*bf16"
    ALLOW_TF32: bool = True
    return (
        [
            VersionedSpec(
                spec={
                    "seq_offsets": ("*i64", s),
                    "Jagged": (dtype, s),
                    "Dense": (dtype, s),
                    "Bias": (dtype, s),
                    "Out": (dtype, s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "N": i,
                    "K": ik,
                    "stride_jm": ik,
                    "stride_db": i,
                    "stride_dk": i,
                    "stride_dn": stride_dn,
                    "stride_bias_b": i,
                    "stride_om": i,
                    "ALLOW_TF32": ALLOW_TF32,
                    "HAS_BIAS": has_bias,
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                },
            )
            for stride_dn in [("i32", 1), ("i32", s)]
            for ik in [("i32", s), "i32"]
            for i in [("i32", s), "i32"]
            for has_bias in [True, False]
            # The spec `("i32", s)` improve vectorization and makes the kernel faster.
            # The second spec does not have such constraints but works on general sizes.
        ]
        + [
            VersionedSpec(
                spec={
                    "seq_offsets": ("*i64", s),
                    "Jagged": (dtype, s),
                    "Dense": (dtype, s),
                    "Bias": (dtype, s),
                    "Out": (dtype, s),
                    "AUTOTUNE_MAX_SEQ_LEN": i,
                    "N": i,
                    "K": i,
                    "stride_jm": i,
                    "stride_db": i,
                    "stride_dk": i,
                    "stride_dn": ("i32", 1),
                    "stride_bias_b": i,
                    "stride_om": i,
                    "ALLOW_TF32": ALLOW_TF32,
                    "HAS_BIAS": has_bias,
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                },
                version="standalone_cint_v1",
            )
            for i in [("i32", s), "i32"]
            for has_bias in [True, False]
            # The spec `("i32", s)` improve vectorization and makes the kernel faster.
            # The second spec does not have such constraints but works on general sizes.
        ]
        + [
            VersionedSpec(
                spec={
                    "seq_offsets": ("*i64", s),
                    "Jagged": (dtype, s),
                    "Dense": (dtype, s),
                    "Bias": (dtype, s),
                    "Out": (dtype, s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "N": i,
                    "K": ik,
                    "stride_jm": ik,
                    "stride_db": i,
                    "stride_dk": i,
                    "stride_dn": stride_dn,
                    "stride_bias_b": i,
                    "stride_om": i,
                    "ALLOW_TF32": ALLOW_TF32,
                    "HAS_BIAS": has_bias,
                    "BLOCK_M": -1,  # autotuned
                    "BLOCK_N": -1,  # autotuned
                    "BLOCK_K": -1,  # autotuned
                },
                version="standalone_cint_v2",
            )
            for stride_dn in [("i32", 1), ("i32", s)]
            for ik in [("i32", s), "i32"]
            for i in [("i32", s), "i32"]
            for has_bias in [True, False]
        ]
    )

@triton_autotune(
    configs=_get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Dense + Bias
    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N), and Out has shape (sum_B(M_i), N)
    """

    off_n = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start * stride_jm
    Dense += off_b * stride_db
    Out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))

jagged_dense_bmm_broadcast_add_kernel = register_tritoncc_specs(
    func=jagged_dense_bmm_broadcast_add_kernel, versioned_specs=_get_bmm_tritoncc_named_specs()
)
jagged_dense_bmm_broadcast_add_kernel = triton_autotune(
    configs=_get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)(jagged_dense_bmm_broadcast_add_kernel.fn)


@triton_autotune(
    configs=_get_bmm_configs(),
    key=["M", "N", "AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _jagged_jagged_bmm_reduce_sum(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    AUTOTUNE_MAX_SEQ_LEN,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computing bmm Out = Jagged x Jagged
    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N), and Out has shape (B, M, N)
    """

    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(
                    out_reduce_ptrs,  # pyre-ignore [61]
                    acc_reduce.to(ReduceOut.dtype.element_ty),
                    mask=(offs_n < N),
                )
        return

    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + offs_m[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b, axis=0)

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )


class _JaggedDenseBmmFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
    ):
        jagged = _switch_to_contiguous_if_needed(jagged)
        L, D = jagged.shape
        B, _, K = dense.shape
        bmm_out = torch.empty((L, K), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(K, meta["BLOCK_N"]),
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            B,
        )

        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=jagged,
            Dense=dense,
            Bias=None,
            Out=bmm_out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            N=K,
            K=D,
            stride_jm=jagged.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(1),
            stride_dn=dense.stride(2),
            stride_bias_b=0,
            stride_om=bmm_out.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.D = D
        return bmm_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_bmm_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        d_dense = torch.empty_like(dense)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.D, meta["BLOCK_N"]),
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            ctx.B,
        )
        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=d_bmm_out,
            Dense=dense,
            Bias=None,
            Out=d_jagged,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            N=ctx.D,
            K=ctx.K,
            stride_jm=d_bmm_out.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(2),
            stride_dn=dense.stride(1),
            stride_bias_b=0,
            stride_om=d_jagged.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        grid = lambda meta: (  # noqa E731
            ctx.B,
            triton.cdiv(ctx.D, meta["BLOCK_M"]),
            triton.cdiv(ctx.K, meta["BLOCK_N"]),
        )
        _jagged_jagged_bmm_reduce_sum[grid](
            seq_offsets=seq_offsets,
            JaggedA=jagged,
            JaggedB=d_bmm_out,
            Out=d_dense,
            ReduceOut=None,
            M=ctx.D,
            N=ctx.K,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            stride_ak=jagged.stride(0),
            stride_bk=d_bmm_out.stride(0),
            stride_ob=d_dense.stride(0),
            stride_om=d_dense.stride(1),
            stride_on=d_dense.stride(2),
            stride_orb=0,
            stride_orn=0,
            REDUCE_JAGGEDB=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return None, None, d_jagged, d_dense


def _get_jagged_dense_broadcast_add_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_N in [16, 32, 64]:
        for num_stages in [1, 2]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_N": BLOCK_N,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                    )
                )
    return configs


def _get_broadcast_add_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    return [
        VersionedSpec(
            spec={
                "seq_offsets": ("*i64", s),
                "Jagged": (dtype, s),
                "Dense": (dtype, s),
                "Out": (dtype, s),
                "AUTOTUNE_MAX_SEQ_LEN": "i32",
                "D": "i32",
                "stride_jn": "i32",
                "stride_db": "i32",
                "stride_on": "i32",
                "BLOCK_N": -1,  # autotuned
                "BLOCK_D": BLOCK_D,
            },
        )
        for dtype in ["*bf16"]
        for BLOCK_D in [32, 64]
    ] + [
        VersionedSpec(
            spec={
                "seq_offsets": ("*i64", s),
                "Jagged": (dtype, s),
                "Dense": (dtype, s),
                "Out": (dtype, s),
                "AUTOTUNE_MAX_SEQ_LEN": "i32",
                "D": "i32",
                "stride_jn": "i32",
                "stride_db": "i32",
                "stride_on": "i32",
                "BLOCK_N": -1,  # autotuned
                "BLOCK_D": BLOCK_D,
            }
        )
        for dtype in ["*fp32"]
        for BLOCK_D in [32, 64]
    ]

@triton_autotune(
    configs=_get_jagged_dense_broadcast_add_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def jagged_dense_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    stride_jn,
    stride_db,
    stride_on,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    Jagged += seq_start * stride_jn
    Dense += off_b * stride_db
    Out += seq_start * stride_on
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_n[:, None] * stride_jn + offs_d[None, :]
    dense_ptrs = Dense + offs_d
    out_ptrs = Out + offs_n[:, None] * stride_jn + offs_d[None, :]
    for d in range(0, D, BLOCK_D):
        jg = tl.load(
            jagged_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_n[:, None] < seq_len) and (d + offs_d)[None, :] < D,
        )
        dn = tl.load(dense_ptrs, mask=d + offs_d < D)
        out = jg + dn[None, :]
        tl.store(
            out_ptrs,
            out,
            mask=(offs_n[:, None] < seq_len) and (d + offs_d)[None, :] < D,
        )
        dense_ptrs += BLOCK_D
        jagged_ptrs += BLOCK_D
        out_ptrs += BLOCK_D

jagged_dense_broadcast_add_kernel = register_tritoncc_specs(
    func=jagged_dense_broadcast_add_kernel, versioned_specs=_get_broadcast_add_tritoncc_named_specs()
)
jagged_dense_broadcast_add_kernel = triton_autotune(
    configs=_get_jagged_dense_broadcast_add_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)(jagged_dense_broadcast_add_kernel.fn)

@triton.jit
def jagged_reduce_sum(
    seq_offsets,
    Jagged,
    Out,
    D,
    stride_jn,
    stride_ob,
    BLOCK_D: tl.constexpr,
):
    """
    Computing Out = Jagged + Dense
    JaggedA has shape (sum_B(N_i), D), Dense has shape (B, D), and Out has shape (sum_B(N_i), D)
    """
    off_b = tl.program_id(0)
    off_d = tl.program_id(1) * BLOCK_D
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    Jagged += seq_start * stride_jn
    Out += off_b * stride_ob
    offs_d = off_d + tl.arange(0, BLOCK_D)
    jagged_ptrs = Jagged + offs_d
    out_ptrs = Out + offs_d
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for _ in range(0, seq_len):
        jg = tl.load(
            jagged_ptrs,
            mask=offs_d < D,
        )
        accumulator += jg
        jagged_ptrs += stride_jn
    out = accumulator.to(Out.dtype.element_ty)
    tl.store(
        out_ptrs,
        out,
        mask=offs_d < D,
    )


class _JaggedDenseBroadcastAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
    ):
        jagged = _switch_to_contiguous_if_needed(jagged)
        dense = _switch_to_contiguous_if_needed(dense)
        L, D = jagged.shape
        B, _ = dense.shape
        out = torch.empty_like(jagged)

        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        jagged_dense_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=jagged,
            Dense=dense,
            Out=out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            stride_jn=jagged.stride(0),
            stride_db=dense.stride(0),
            stride_on=out.stride(0),
            BLOCK_D=BLOCK_D,
        )

        ctx.save_for_backward(seq_offsets)
        ctx.max_seq_len = max_seq_len
        ctx.B = B
        ctx.D = D
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor]:
        seq_offsets = ctx.saved_tensors[0]
        d_dense = torch.empty((ctx.B, ctx.D), device=d_out.device, dtype=d_out.dtype)
        BLOCK_D = triton.next_power_of_2(ctx.D) if ctx.D < 64 else 64
        jagged_reduce_sum[(ctx.B, triton.cdiv(ctx.D, BLOCK_D))](
            seq_offsets=seq_offsets,
            Jagged=d_out,
            Out=d_dense,
            D=ctx.D,
            stride_jn=d_out.stride(0),
            stride_ob=d_dense.stride(0),
            BLOCK_D=BLOCK_D,
        )
        return None, None, d_out, d_dense


class _JaggedDenseBmmBroadcastAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
        bias: torch.Tensor,
    ):
        jagged = _switch_to_contiguous_if_needed(jagged)
        bias = _switch_to_contiguous_if_needed(bias)
        L, K = jagged.shape
        B, _, N = dense.shape
        out = torch.empty((L, N), dtype=jagged.dtype, device=jagged.device)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(max_seq_len, meta["BLOCK_M"]),
            B,
        )

        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=jagged,
            Dense=dense,
            Bias=bias,
            Out=out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            N=N,
            K=K,
            stride_jm=jagged.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(1),
            stride_dn=dense.stride(2),
            stride_bias_b=bias.stride(0),
            stride_om=out.stride(0),
            HAS_BIAS=True,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        d_dense = torch.empty_like(dense)
        d_bias = torch.empty((ctx.B, ctx.N), device=d_out.device, dtype=d_out.dtype)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.K, meta["BLOCK_N"]),
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            ctx.B,
        )
        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=d_out,
            Dense=dense,
            Bias=None,
            Out=d_jagged,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            N=ctx.K,
            K=ctx.N,
            stride_jm=d_out.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(2),
            stride_dn=dense.stride(1),
            stride_bias_b=0,
            stride_om=d_jagged.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        grid = lambda meta: (  # noqa E731
            ctx.B,
            triton.cdiv(ctx.K, meta["BLOCK_M"]),
            triton.cdiv(ctx.N, meta["BLOCK_N"]),
        )
        _jagged_jagged_bmm_reduce_sum[grid](
            seq_offsets=seq_offsets,
            JaggedA=jagged,
            JaggedB=d_out,
            Out=d_dense,
            ReduceOut=d_bias,
            M=ctx.K,
            N=ctx.N,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            stride_ak=jagged.stride(0),
            stride_bk=d_out.stride(0),
            stride_ob=d_dense.stride(0),
            stride_om=d_dense.stride(1),
            stride_on=d_dense.stride(2),
            stride_orb=d_bias.stride(0),
            stride_orn=d_bias.stride(1),
            REDUCE_JAGGEDB=True,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return None, None, d_jagged, d_dense, d_bias


def _get_concat_2D_jagged_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    default_values = {
        "IS_REPLACE": 0,
    }
    return (
        [
            VersionedSpec(
                spec={
                    "OffsetsA": "*i64",
                    "ValuesA": (dtype, s),
                    "OffsetsB": offsets_b_type,
                    "ValuesB": (dtype, s),
                    "DenseSize": "i32",
                    "Out": (dtype, s),
                    "D": "i32",
                    "stride_ad": "i32",
                    "stride_bd": "i32",
                    "stride_dense_batch": "i32",
                    "stride_od": "i32",
                    "IS_DENSE_A": IS_DENSE_A,
                    "IS_DENSE_B": IS_DENSE_B,
                    "BLOCK_D": BLOCK_D,
                    "IS_REPLACE": IS_REPLACE,
                },
                default_values=default_values,
            )
            for BLOCK_D, dtype in [
                (64, "*bf16"),
                (64, "*fp32"),
                (128, "*fp32"),
                (128, "*bf16"),
                (256, "*bf16"),
                (256, "*fp32"),
                (512, "*bf16"),
                (512, "*fp32"),
            ]
            for offsets_b_type in ["*i64", "*i32"]
            for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
            for IS_REPLACE in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "OffsetsA": ("*i64", s),
                    "ValuesA": (dtype, s),
                    "OffsetsB": ("*i64", s),
                    "ValuesB": (dtype, s),
                    "DenseSize": "i32",
                    "Out": (dtype, s),
                    "D": "i32",
                    "stride_ad": "i32",
                    "stride_bd": "i32",
                    "stride_dense_batch": "i32",
                    "stride_od": "i32",
                    "IS_DENSE_A": IS_DENSE_A,
                    "IS_DENSE_B": IS_DENSE_B,
                    "BLOCK_D": BLOCK_D,
                    "IS_REPLACE": False,
                },
                default_values=default_values,
                version="standalone_cint_v1",
            )
            for BLOCK_D, dtype in [
                (64, "*fp32"),
                (128, "*fp32"),
                (128, "*bf16"),
                (256, "*bf16"),
                (256, "*fp32"),
                (512, "*bf16"),
                (512, "*fp32"),
            ]
            for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
        ]
        + [
            VersionedSpec(
                spec={
                    "OffsetsA": "*i64",
                    "ValuesA": (dtype, s),
                    "OffsetsB": offsets_b_type,
                    "ValuesB": (dtype, s),
                    "DenseSize": "i32",
                    "Out": (dtype, s),
                    "D": "i32",
                    "stride_ad": "i32",
                    "stride_bd": "i32",
                    "stride_dense_batch": "i32",
                    "stride_od": "i32",
                    "IS_DENSE_A": IS_DENSE_A,
                    "IS_DENSE_B": IS_DENSE_B,
                    "BLOCK_D": BLOCK_D,
                    "IS_REPLACE": IS_REPLACE,
                },
                default_values=default_values,
                version="standalone_cint_v2",
            )
            for BLOCK_D, dtype in [
                (64, "*fp32"),
                (128, "*fp32"),
                (128, "*bf16"),
                (256, "*bf16"),
                (256, "*fp32"),
                (512, "*bf16"),
                (512, "*fp32"),
            ]
            for offsets_b_type in ["*i64", "*i32"]
            for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
            for IS_REPLACE in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "OffsetsA": "*i64",
                    "ValuesA": (dtype, s),
                    "OffsetsB": offsets_b_type,
                    "ValuesB": (dtype, s),
                    "DenseSize": "i32",
                    "Out": (dtype, s),
                    "D": "i32",
                    "stride_ad": "i32",
                    "stride_bd": "i32",
                    "stride_dense_batch": "i32",
                    "stride_od": "i32",
                    "IS_DENSE_A": IS_DENSE_A,
                    "IS_DENSE_B": IS_DENSE_B,
                    "BLOCK_D": BLOCK_D,
                    "IS_REPLACE": IS_REPLACE,
                },
                default_values=default_values,
                version="standalone_cint_v3_concat_2d",
            )
            for BLOCK_D, dtype in [
                (64, "*bf16"),
                (64, "*fp32"),
                (128, "*fp32"),
                (128, "*bf16"),
                (256, "*bf16"),
                (256, "*fp32"),
                (512, "*bf16"),
                (512, "*fp32"),
            ]
            for offsets_b_type in ["*i64", "*i32"]
            for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
            for IS_REPLACE in [True, False]
        ]
    )


@triton.jit
def concat_2D_jagged_w_prefix(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    n_prefix_from_B,  # nonzero is not supported when IS_REPLACE=True
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_n = tl.program_id(1)
    if IS_DENSE_A:
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b

    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return

    offs_d = tl.arange(0, BLOCK_D)
    if IS_REPLACE:
        out_seq_start = seq_start_a + off_n
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        out_seq_start = seq_start_a + seq_start_b + off_n
        out_seq_b_start = seq_len_a + n_prefix_from_B

    out_ptrs = Out + out_seq_start * stride_od + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_from_B:
        off_a = off_n - n_prefix_from_B
        if IS_DENSE_A:
            in_ptrs = ValuesA + off_a * stride_ad + off_z * stride_dense_batch + offs_d
        else:
            in_ptrs = ValuesA + (off_a + seq_start_a) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_from_B
        if off_n < n_prefix_from_B:
            off_b += out_seq_b_start - n_prefix_from_B
        if IS_DENSE_B:
            in_ptrs = ValuesB + off_b * stride_bd + off_z * stride_dense_batch + offs_d
        else:
            in_ptrs = ValuesB + (off_b + seq_start_b) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def concat_2D_jagged(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    DenseSize,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_dense_batch,
    stride_od,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    concat_2D_jagged_w_prefix(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        DenseSize,
        Out,
        D,
        stride_ad,
        stride_bd,
        stride_dense_batch,
        stride_od,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        IS_REPLACE,
    )

concat_2D_jagged = register_tritoncc_specs(
    func=concat_2D_jagged, versioned_specs=_get_concat_2D_jagged_tritoncc_named_specs()
)

@triton.jit
def concat_2D_jagged_jagged_w_prefix(
    OffsetsA,
    ValuesA,
    OffsetsB,
    ValuesB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
    BLOCK_D: tl.constexpr,
):
    concat_2D_jagged_w_prefix(
        OffsetsA,
        ValuesA,
        OffsetsB,
        ValuesB,
        0,
        Out,
        D,
        stride_ad,
        stride_bd,
        0,
        stride_od,
        n_prefix_from_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        IS_REPLACE=False,
    )


def _get_split_2D_jagged_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    default_values = {
        "IS_REPLACE": 0,
    }
    return [
        VersionedSpec(
            spec={
                "JaggedIn": (dtype, s),
                "DenseSize": "i32",
                "OffsetsA": offsets_a_type,
                "OffsetsB": offsets_b_type,
                "OutA": (dtype, s),
                "OutB": (dtype, s),
                "D": ("i32", s),
                "stride_id": ("i32", s),
                "stride_ad": ("i32", s),
                "stride_bd": ("i32", s),
                "IS_DENSE_A": IS_DENSE_A,
                "IS_DENSE_B": IS_DENSE_B,
                "BLOCK_D": BLOCK_D,
                "IS_REPLACE": False,
            },
            default_values=default_values,
        )
        for BLOCK_D in [64, 128, 256, 512]
        for dtype in ["*bf16", "*fp32"]
        for offsets_a_type in ["*i64", "*i32"]
        for offsets_b_type in ["*i64", "*i32"]
        for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
    ] + [
        VersionedSpec(
            spec={
                "JaggedIn": (dtype, s),
                "DenseSize": "i32",
                "OffsetsA": offsets_a_type,
                "OffsetsB": offsets_b_type,
                "OutA": (dtype, s),
                "OutB": (dtype, s),
                "D": ("i32", s),
                "stride_id": ("i32", s),
                "stride_ad": ("i32", s),
                "stride_bd": ("i32", s),
                "IS_DENSE_A": IS_DENSE_A,
                "IS_DENSE_B": IS_DENSE_B,
                "BLOCK_D": BLOCK_D,
                "IS_REPLACE": False,
            },
            default_values=default_values,
            version="standalone_cint_v2_split2d",
        )
        for BLOCK_D in [64, 128, 256, 512]
        for dtype in ["*bf16", "*fp32"]
        for offsets_a_type in ["*i64", "*i32"]
        for offsets_b_type in ["*i64", "*i32"]
        for IS_DENSE_A, IS_DENSE_B in [(False, False), (True, False), (False, True)]
    ]


@triton.jit
def split_2D_jagged_w_prefix(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_n = tl.program_id(1)
    if IS_DENSE_A:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_start_a = off_z * DenseSize
        seq_len_a = DenseSize
        seq_len_b = seq_end_b - seq_start_b
    elif IS_DENSE_B:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = off_z * DenseSize
        seq_len_b = DenseSize
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    if IS_REPLACE:
        seq_len = seq_len_a
    else:
        seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return

    if IS_REPLACE:
        seq_start = seq_start_a
        out_seq_b_start = seq_len_a - seq_len_b
    else:
        seq_start = seq_start_a + seq_start_b
        out_seq_b_start = seq_len_a + n_prefix_to_B

    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n) * stride_id + offs_d
    if off_n < out_seq_b_start and off_n >= n_prefix_to_B:
        off_a = off_n - n_prefix_to_B
        out_ptrs = OutA + (off_a + seq_start_a) * stride_ad + offs_d
    else:
        off_b = off_n - out_seq_b_start + n_prefix_to_B
        if off_n < n_prefix_to_B:
            off_b += out_seq_b_start - n_prefix_to_B
        out_ptrs = OutB + (off_b + seq_start_b) * stride_bd + offs_d
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def split_2D_jagged(
    JaggedIn,
    DenseSize,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_REPLACE: tl.constexpr,
):
    split_2D_jagged_w_prefix(
        JaggedIn,
        DenseSize,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        0,
        IS_DENSE_A,
        IS_DENSE_B,
        BLOCK_D,
        IS_REPLACE,
    )

split_2D_jagged = register_tritoncc_specs(
    func=split_2D_jagged, versioned_specs=_get_split_2D_jagged_tritoncc_named_specs()
)

@triton.jit
def split_2D_jagged_jagged_w_prefix(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    BLOCK_D: tl.constexpr,
):
    split_2D_jagged_w_prefix(
        JaggedIn,
        0,
        OffsetsA,
        OffsetsB,
        OutA,
        OutB,
        D,
        stride_id,
        stride_ad,
        stride_bd,
        n_prefix_to_B,
        IS_DENSE_A=False,
        IS_DENSE_B=False,
        BLOCK_D=BLOCK_D,
        IS_REPLACE=False,
    )


class _Concat2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        values_a: torch.Tensor,
        values_b: torch.Tensor,
        offsets_a: Optional[torch.Tensor] = None,
        offsets_b: Optional[torch.Tensor] = None,
        is_replace: bool = False,
        n_prefix_from_right: int = 0,
    ):
        values_a = _switch_to_contiguous_if_needed(values_a)
        values_b = _switch_to_contiguous_if_needed(values_b)
        is_dense_a = offsets_a is None
        is_dense_b = offsets_b is None
        dense_size: int = 0
        if is_dense_a:
            assert offsets_b is not None
            B, dense_size, D = values_a.shape
            seq_len_a = dense_size * B
            seq_len_b, _ = values_b.shape
            device = values_b.device
            dtype = values_b.dtype
            stride_dense_batch = values_a.stride(0)
        elif is_dense_b:
            assert offsets_a is not None
            B, dense_size, D = values_b.shape
            seq_len_a, _ = values_a.shape
            seq_len_b = dense_size * B
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = values_b.stride(0)
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1
            seq_len_a, D = values_a.shape
            seq_len_b, _ = values_b.shape
            device = values_a.device
            dtype = values_a.dtype
            stride_dense_batch = 0

        BLOCK_D = triton.next_power_of_2(D)
        if is_replace:
            values_out = torch.empty_like(values_a)
        else:
            values_out = torch.empty(
                (seq_len_a + seq_len_b, D), device=device, dtype=dtype
            )
        if n_prefix_from_right == 0:
            concat_2D_jagged[(B, max_seq_len)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                DenseSize=dense_size,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_dense_batch=stride_dense_batch,
                stride_od=values_out.stride(0),
                IS_DENSE_A=is_dense_a,  # pyre-ignore[6]
                IS_DENSE_B=is_dense_b,  # pyre-ignore[6]
                BLOCK_D=BLOCK_D,
                IS_REPLACE=is_replace,  # pyre-ignore[6]
            )
        else:
            concat_2D_jagged_jagged_w_prefix[(B, max_seq_len)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                Out=values_out,
                D=D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_od=values_out.stride(0),
                n_prefix_from_B=n_prefix_from_right,
                BLOCK_D=BLOCK_D,
            )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.seq_len_a = seq_len_a
        ctx.seq_len_b = seq_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.dense_size = dense_size
        ctx.is_replace = is_replace
        ctx.n_prefix_from_right = n_prefix_from_right
        return values_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, torch.Tensor, torch.Tensor, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        is_dense_a, is_dense_b, is_replace = (
            ctx.is_dense_a,
            ctx.is_dense_b,
            ctx.is_replace,
        )
        dense_size = ctx.dense_size
        if is_dense_a:
            B = offsets_b.shape[0] - 1
        else:
            B = offsets_a.shape[0] - 1
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.zeros(
            (ctx.seq_len_a, D), device=d_out.device, dtype=d_out.dtype
        )
        values_b = torch.empty(
            (ctx.seq_len_b, D), device=d_out.device, dtype=d_out.dtype
        )
        if ctx.n_prefix_from_right == 0:
            split_2D_jagged[(B, ctx.max_seq_len)](
                JaggedIn=d_out,
                DenseSize=dense_size,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=values_a,
                OutB=values_b,
                D=D,
                stride_id=d_out.stride(0),
                stride_ad=values_a.stride(0),
                stride_bd=values_b.stride(0),
                BLOCK_D=BLOCK_D,
                IS_DENSE_A=is_dense_a,
                IS_DENSE_B=is_dense_b,
                IS_REPLACE=is_replace,
            )
        else:
            split_2D_jagged_jagged_w_prefix[(B, ctx.max_seq_len)](
                JaggedIn=d_out,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=values_a,
                OutB=values_b,
                D=D,
                stride_id=d_out.stride(0),
                stride_ad=values_a.stride(0),
                stride_bd=values_b.stride(0),
                n_prefix_to_B=ctx.n_prefix_from_right,
                BLOCK_D=BLOCK_D,
            )

        if is_dense_a:
            values_a = values_a.reshape((B, dense_size, D))
        elif is_dense_b:
            values_b = values_b.reshape((B, dense_size, D))
        return None, values_a, values_b, None, None, None, None


class _Split2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values: torch.Tensor,
        max_seq_len: int,
        offsets_a: Optional[torch.Tensor] = None,
        offsets_b: Optional[torch.Tensor] = None,
        dense_size: int = 0,
        n_prefix_to_right: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = _switch_to_contiguous_if_needed(values)
        is_dense_a: bool = offsets_a is None
        is_dense_b: bool = offsets_b is None
        if is_dense_a:
            L, _ = values.shape
            assert offsets_b is not None
            B = offsets_b.shape[0] - 1
            seq_len_a = dense_size * B
            seq_len_b = L - seq_len_a
            offsets_a = offsets_b.new_empty(0)
        elif is_dense_b:
            L, _ = values.shape
            assert offsets_a is not None
            B = offsets_a.shape[0] - 1
            seq_len_b = dense_size * B
            seq_len_a = L - seq_len_b
            offsets_b = offsets_a.new_empty(0)
        else:
            assert offsets_a is not None and offsets_b is not None
            B = offsets_a.shape[0] - 1
            seq_len_a = int(offsets_a[-1].item())
            seq_len_b = int(offsets_b[-1].item())
        _, D = values.shape
        BLOCK_D = triton.next_power_of_2(D)
        values_a = torch.empty((seq_len_a, D), device=values.device, dtype=values.dtype)
        values_b = torch.empty((seq_len_b, D), device=values.device, dtype=values.dtype)
        if n_prefix_to_right == 0:
            split_2D_jagged[(B, max_seq_len)](
                JaggedIn=values,
                DenseSize=dense_size,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=values_a,
                OutB=values_b,
                D=D,
                stride_id=values.stride(0),
                stride_ad=values_a.stride(0),
                stride_bd=values_b.stride(0),
                IS_DENSE_A=is_dense_a, # pyre-ignore[6]
                IS_DENSE_B=is_dense_b, # pyre-ignore[6]
                BLOCK_D=BLOCK_D,
                IS_REPLACE=False, # pyre-ignore[6]
            )
        else:
            split_2D_jagged_jagged_w_prefix[(B, max_seq_len)](
                JaggedIn=values,
                OffsetsA=offsets_a,
                OffsetsB=offsets_b,
                OutA=values_a,
                OutB=values_b,
                D=D,
                stride_id=values.stride(0),
                stride_ad=values_a.stride(0),
                stride_bd=values_b.stride(0),
                n_prefix_to_B=n_prefix_to_right,
                BLOCK_D=BLOCK_D,
            )
        if is_dense_a:
            values_a = values_a.reshape(B, dense_size, D)
        if is_dense_b:
            values_b = values_b.reshape(B, dense_size, D)
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.seq_len_a = seq_len_a
        ctx.seq_len_b = seq_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.dense_size = dense_size
        ctx.B = B
        ctx.D = D
        ctx.n_prefix_to_right = n_prefix_to_right
        return values_a, values_b

    @staticmethod
    def backward(ctx, *d_values) -> Tuple[torch.Tensor, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        is_dense_a, is_dense_b = ctx.is_dense_a, ctx.is_dense_b
        values_a, values_b = d_values
        if is_dense_a:
            stride_dense_batch = values_a.stride(0)
        elif is_dense_b:
            stride_dense_batch = values_b.stride(0)
        else:
            stride_dense_batch = 0

        BLOCK_D = triton.next_power_of_2(ctx.D)
        dvalues = torch.empty(
            (ctx.seq_len_a + ctx.seq_len_b, ctx.D),
            device=values_a.device,
            dtype=values_b.dtype,
        )
        if ctx.n_prefix_to_right == 0:
            concat_2D_jagged[(ctx.B, ctx.max_seq_len)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                DenseSize=ctx.dense_size,
                Out=dvalues,
                D=ctx.D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_dense_batch=stride_dense_batch,
                stride_od=dvalues.stride(0),
                IS_DENSE_A=is_dense_a,
                IS_DENSE_B=is_dense_b,
                BLOCK_D=BLOCK_D,
                IS_REPLACE=False, # pyre-ignore[6]
            )
        else:
            concat_2D_jagged_jagged_w_prefix[(ctx.B, ctx.max_seq_len)](
                OffsetsA=offsets_a,
                ValuesA=values_a,
                OffsetsB=offsets_b,
                ValuesB=values_b,
                Out=dvalues,
                D=ctx.D,
                stride_ad=values_a.stride(-2),
                stride_bd=values_b.stride(-2),
                stride_od=dvalues.stride(0),
                n_prefix_from_B=ctx.n_prefix_to_right,
                BLOCK_D=BLOCK_D,
            )

        return dvalues, None, None, None, None, None


def _get_copy_2D_jagged_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    # TODO: currently the types are placeholders, need to be changed later
    return [
        VersionedSpec(
            spec={
                "JaggedIn": ("*fp32", s),
                "Offsets": ("*i64", s),
                "JaggedOut": ("*fp32", s),
                "OffsetsOut": ("*i64", s),
                "D": "i32",
                "stride_id": "i32",
                "stride_od": "i32",
                "len_in": "i32",
                "len_out": "i32",
                "BLOCK_D": BLOCK_D,
            }
        )
        for BLOCK_D in [4, 256, 512]
    ]


@triton.jit
def copy_2D_jagged(
    JaggedIn,
    Offsets,
    JaggedOut,
    OffsetsOut,
    D,
    stride_id,
    stride_od,
    len_in,
    len_out,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_z)
    seq_end = tl.load(Offsets + off_z + 1)
    seq_len = seq_end - seq_start
    seq_start_out = tl.load(OffsetsOut + off_z)
    if off_n >= seq_len:
        return

    in_block_start = seq_start + off_n
    out_block_start = seq_start_out + off_n
    x = tl.load(
        tl.make_block_ptr(
            base=JaggedIn,
            shape=(len_in, D),
            strides=(stride_id, 1),
            offsets=(in_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        boundary_check=(1, 0),
    )
    tl.store(
        tl.make_block_ptr(
            base=JaggedOut,
            shape=(len_out, D),
            strides=(stride_od, 1),
            offsets=(out_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        x,
        boundary_check=(1, 0),
    )

copy_2D_jagged = register_tritoncc_specs(
    func=copy_2D_jagged, versioned_specs=_get_copy_2D_jagged_tritoncc_named_specs()
)

@triton.jit
def shrink_2D_jagged_from_tail(
    JaggedIn,
    Offsets,
    JaggedOut,
    OffsetsOut,
    D,
    stride_id,
    stride_od,
    len_in,
    len_out,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_z)
    seq_end = tl.load(Offsets + off_z + 1)
    seq_len = seq_end - seq_start
    seq_start_out = tl.load(OffsetsOut + off_z)
    seq_end_out = tl.load(OffsetsOut + off_z + 1)
    seq_len_out = seq_end_out - seq_start_out
    if off_n >= seq_len_out:
        return

    to_skip = seq_len - seq_len_out
    in_block_start = seq_start + to_skip + off_n
    out_block_start = seq_start_out + off_n
    x = tl.load(
        tl.make_block_ptr(
            base=JaggedIn,
            shape=(len_in, D),
            strides=(stride_id, 1),
            offsets=(in_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        boundary_check=(1, 0),
    )
    tl.store(
        tl.make_block_ptr(
            base=JaggedOut,
            shape=(len_out, D),
            strides=(stride_od, 1),
            offsets=(out_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        x,
        boundary_check=(1, 0),
    )

shrink_2D_jagged_from_tail = register_tritoncc_specs(
    func=shrink_2D_jagged_from_tail, versioned_specs=_get_copy_2D_jagged_tritoncc_named_specs()
)

@triton.jit
def unshrink_2D_jagged_from_tail(
    JaggedIn,
    Offsets,
    JaggedOut,
    OffsetsOut,
    D,
    stride_id,
    stride_od,
    len_in,
    len_out,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_z)
    seq_end = tl.load(Offsets + off_z + 1)
    seq_len = seq_end - seq_start
    seq_start_out = tl.load(OffsetsOut + off_z)
    seq_end_out = tl.load(OffsetsOut + off_z + 1)
    seq_len_out = seq_end_out - seq_start_out
    if off_n >= seq_len:
        return

    to_skip = seq_len_out - seq_len
    in_block_start = seq_start + off_n
    out_block_start = seq_start_out + to_skip + off_n
    x = tl.load(
        tl.make_block_ptr(
            base=JaggedIn,
            shape=(len_in, D),
            strides=(stride_id, 1),
            offsets=(in_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        boundary_check=(1, 0),
    )
    tl.store(
        tl.make_block_ptr(
            base=JaggedOut,
            shape=(len_out, D),
            strides=(stride_od, 1),
            offsets=(out_block_start.to(tl.int32), 0),
            block_shape=(1, BLOCK_D),
            order=(1, 0),
        ),
        x,
        boundary_check=(1, 0),
    )

unshrink_2D_jagged_from_tail = register_tritoncc_specs(
    func=unshrink_2D_jagged_from_tail, versioned_specs=_get_copy_2D_jagged_tritoncc_named_specs()
)

class _Shrink2DJaggedFromHeadFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
        shrunk_max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = _switch_to_contiguous_if_needed(values)
        B = offsets.size(0) - 1
        _, D = values.shape
        BLOCK_D = triton.next_power_of_2(D)
        lengths = offsets[1:] - offsets[:-1]
        seq_len = int(lengths.sum().item())
        shrunk_lengths = torch.clamp(lengths, max=shrunk_max_seq_len)
        shrunk_seq_len = int(shrunk_lengths.sum().item())
        values_o = torch.empty(
            (shrunk_seq_len, D), device=values.device, dtype=values.dtype
        )
        offsets_o = torch.ops.fbgemm.asynchronous_complete_cumsum(shrunk_lengths)
        copy_2D_jagged[(B, min(shrunk_max_seq_len, max_seq_len))](
            JaggedIn=values,
            Offsets=offsets,
            JaggedOut=values_o,
            OffsetsOut=offsets_o,
            D=D,
            stride_id=values.stride(0),
            stride_od=values_o.stride(0),
            len_in=seq_len,
            len_out=shrunk_seq_len,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(offsets, offsets_o)
        ctx.shrunk_max_seq_len = shrunk_max_seq_len
        ctx.seq_len = seq_len
        return values_o, offsets_o

    @staticmethod
    def backward(ctx, *d_outputs) -> Tuple[torch.Tensor, None, None, None, None]:
        d_shrunk_values = d_outputs[0]
        offsets, shrunk_offsets = ctx.saved_tensors
        B = offsets.size(0) - 1
        shrunk_seq_len, D = d_shrunk_values.shape
        BLOCK_D = triton.next_power_of_2(D)
        d_values = torch.zeros(
            (ctx.seq_len, D), device=d_shrunk_values.device, dtype=d_shrunk_values.dtype
        )
        copy_2D_jagged[(B, ctx.shrunk_max_seq_len)](
            JaggedIn=d_shrunk_values,
            Offsets=shrunk_offsets,
            JaggedOut=d_values,
            OffsetsOut=offsets,
            D=D,
            stride_id=d_shrunk_values.stride(0),
            stride_od=d_values.stride(0),
            len_in=shrunk_seq_len,
            len_out=ctx.seq_len,
            BLOCK_D=BLOCK_D,
        )
        return d_values, None, None, None, None


class _Shrink2DJaggedFromTailFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
        shrunk_max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = _switch_to_contiguous_if_needed(values)
        B = offsets.size(0) - 1
        _, D = values.shape
        BLOCK_D = triton.next_power_of_2(D)
        lengths = offsets[1:] - offsets[:-1]
        seq_len = int(lengths.sum().item())
        shrunk_lengths = torch.clamp(lengths, max=shrunk_max_seq_len)
        shrunk_seq_len = int(shrunk_lengths.sum().item())
        values_o = torch.empty(
            (shrunk_seq_len, D), device=values.device, dtype=values.dtype
        )
        offsets_o = torch.ops.fbgemm.asynchronous_complete_cumsum(shrunk_lengths)
        shrink_2D_jagged_from_tail[(B, min(shrunk_max_seq_len, max_seq_len))](
            JaggedIn=values,
            Offsets=offsets,
            JaggedOut=values_o,
            OffsetsOut=offsets_o,
            D=D,
            stride_id=values.stride(0),
            stride_od=values_o.stride(0),
            len_in=seq_len,
            len_out=shrunk_seq_len,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(offsets, offsets_o)
        ctx.shrunk_max_seq_len = shrunk_max_seq_len
        ctx.seq_len = seq_len
        return values_o, offsets_o

    @staticmethod
    def backward(ctx, *d_outputs) -> Tuple[torch.Tensor, None, None, None, None]:
        d_shrunk_values = d_outputs[0]
        offsets, shrunk_offsets = ctx.saved_tensors
        B = offsets.size(0) - 1
        shrunk_seq_len, D = d_shrunk_values.shape
        BLOCK_D = triton.next_power_of_2(D)
        d_values = torch.zeros(
            (ctx.seq_len, D), device=d_shrunk_values.device, dtype=d_shrunk_values.dtype
        )
        unshrink_2D_jagged_from_tail[(B, ctx.shrunk_max_seq_len)](
            JaggedIn=d_shrunk_values,
            Offsets=shrunk_offsets,
            JaggedOut=d_values,
            OffsetsOut=offsets,
            D=D,
            stride_id=d_shrunk_values.stride(0),
            stride_od=d_values.stride(0),
            len_in=shrunk_seq_len,
            len_out=ctx.seq_len,
            BLOCK_D=BLOCK_D,
        )
        return d_values, None, None, None, None


def _get_remove_first_or_last_1D_named_specs() -> List[VersionedSpec]:
    s: int = 16

    return [
        VersionedSpec(
            spec={
                "Offsets": ("*i64", s),
                "JaggedIn": (dtype, s),
                "JaggedOut_no_first": (dtype, s),
                "JaggedOut_no_last": (dtype, s),
                "BLOCK_N": BLOCK_N,
            }
        )
        for dtype in ["*fp32", "*fp16", "*bf16", "*i32"]
        for BLOCK_N in [128, 256]
    ]


@triton.jit
def jagged_remove_first_or_last_1D_kernel(
    JaggedIn,
    Offsets,
    JaggedOut_no_first,
    JaggedOut_no_last,
    BLOCK_N: tl.constexpr,
):
    off_z = tl.program_id(0)  # offset along the batch dimension
    group_n = tl.program_id(1)

    in_seq_start = tl.load(Offsets + off_z)
    in_seq_end = tl.load(Offsets + off_z + 1)
    in_seq_len = in_seq_end - in_seq_start

    out_seq_start = tl.load(Offsets + off_z) - off_z  # ith offset_out = offset_in - i

    off_n = group_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x = tl.load(JaggedIn + in_seq_start + off_n, mask=off_n < in_seq_len)
    tl.store(
        JaggedOut_no_first + out_seq_start + off_n - 1,
        x,
        mask=(off_n > 0) & (off_n < in_seq_len),
    )
    tl.store(
        JaggedOut_no_last + out_seq_start + off_n,
        x,
        mask=off_n < in_seq_len - 1,
    )

jagged_remove_first_or_last_1D_kernel = register_tritoncc_specs(
    func=jagged_remove_first_or_last_1D_kernel, versioned_specs=_get_remove_first_or_last_1D_named_specs()
)

@triton.jit
def jagged_recover_first_or_last_1D_kernel(
    JaggedIn_no_first,
    JaggedIn_no_last,
    JaggedOut,
    OffsetsOut,
    BLOCK_N: tl.constexpr,
):
    off_z = tl.program_id(0)  # offset along the batch dimension
    group_n = tl.program_id(1)

    in_no_last_seq_start = tl.load(OffsetsOut + off_z) - off_z
    in_no_last_seq_end = tl.load(OffsetsOut + off_z + 1) - off_z - 1
    in_seq_len = in_no_last_seq_end - in_no_last_seq_start

    out_seq_start = tl.load(OffsetsOut + off_z)

    off_n = group_n * BLOCK_N + tl.arange(0, BLOCK_N)

    x_first = tl.load(JaggedIn_no_last + in_no_last_seq_start + off_n, mask=off_n == 0)
    tl.store(JaggedOut + out_seq_start + off_n, x_first, mask=off_n == 0)

    x = tl.load(
        JaggedIn_no_last + in_no_last_seq_start + off_n,
        mask=(off_n < in_seq_len) & (off_n > 0),
    ) + tl.load(
        JaggedIn_no_first + in_no_last_seq_start + off_n - 1,
        mask=(off_n < in_seq_len) & (off_n > 0),
    )
    tl.store(
        JaggedOut + out_seq_start + off_n, x, mask=(off_n < in_seq_len) & (off_n > 0)
    )

    x_last = tl.load(
        JaggedIn_no_first + in_no_last_seq_start + off_n, mask=off_n == in_seq_len - 1
    )
    tl.store(
        JaggedOut + out_seq_start + off_n + 1, x_last, mask=off_n == in_seq_len - 1
    )


class _JaggedRemoveFirstOrLast1D(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values: torch.Tensor,
        lengths: torch.Tensor,
        offsets: torch.Tensor,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        B = lengths.size(0)
        N = values.size(0)

        shrunk_seq_len = N - B
        values_no_first = torch.empty(
            (shrunk_seq_len,), device=values.device, dtype=values.dtype
        )
        values_no_last = torch.empty(
            (shrunk_seq_len,), device=values.device, dtype=values.dtype
        )

        BLOCK_N = 128

        grid = (B, triton.cdiv(max_seq_len, BLOCK_N))  # noqa E731

        jagged_remove_first_or_last_1D_kernel[grid](
            JaggedIn=values,
            Offsets=offsets,
            JaggedOut_no_first=values_no_first,
            JaggedOut_no_last=values_no_last,
            BLOCK_N=BLOCK_N,  # pyre-ignore[6]
        )

        ctx.save_for_backward(offsets, lengths)
        ctx.N = N
        ctx.max_seq_len = max_seq_len

        return values_no_first, values_no_last

    @staticmethod
    def backward(ctx, *d_values_in) -> Tuple[torch.Tensor, None, None, None]:

        d_values_no_first, d_values_no_last = d_values_in
        offsets, lengths = ctx.saved_tensors
        max_seq_len = ctx.max_seq_len
        B = offsets.size(0) - 1
        d_values = torch.empty(
            (ctx.N,),
            device=d_values_no_first.device,
            dtype=d_values_no_first.dtype,
        )

        BLOCK_N = 128
        grid = (B, triton.cdiv(max_seq_len, BLOCK_N))  # noqa E731

        jagged_recover_first_or_last_1D_kernel[grid](
            JaggedIn_no_first=d_values_no_first,
            JaggedIn_no_last=d_values_no_last,
            JaggedOut=d_values,
            OffsetsOut=offsets,
            BLOCK_N=BLOCK_N,  # pyre-ignore
        )

        return d_values, None, None, None

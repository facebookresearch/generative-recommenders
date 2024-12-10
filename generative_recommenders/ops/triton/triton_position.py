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
from hammer.ops.triton.utils import prev_power_of_2

try:
    from hammer.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        autotune_max_seq_len,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )

    torch.ops.load_library("//hammer/ops/cuda:cuda_ops")
except ImportError:
    from generative_recommenders.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        autotune_max_seq_len,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )


def _add_position_embeddings_configs() -> List[triton.Config]:
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


def _add_position_embeddings_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    return [
        VersionedSpec(
            spec={
                "Jagged": (dtype, s),
                "seq_offsets": ("*i64", s),
                "high_inds": ("*i64", s),
                "Dense": (dtype, s),
                "Out": (dtype, s),
                "AUTOTUNE_MAX_SEQ_LEN": "i32",
                "D": "i32",
                "scale": "f32",
                "stride_jn": "i32",
                "stride_dk": "i32",
                "stride_on": "i32",
                "SCALE_JAGGED": True,
                "BLOCK_N": -1,  # autotuned
                "BLOCK_D": BLOCK_D,
            },
        )
        for dtype in ["*bf16", "*fp32"]
        for BLOCK_D in [32, 64]
    ]


@triton_autotune(
    configs=_add_position_embeddings_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _add_position_embeddings_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    Dense,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    scale,
    stride_jn,
    stride_dk,
    stride_on,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Jagged has shape (sum_B(N_i), D),
    Dense has shape (K, D),
    Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(seq_offsets + off_b)
    seq_end = tl.load(seq_offsets + off_b + 1)
    max_ind = tl.load(high_inds + off_b)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    clamped_offs_n = tl.where(offs_n >= max_ind, max_ind, offs_n)
    offs_d = tl.arange(0, BLOCK_D)
    Jagged += seq_start.to(tl.int64) * stride_jn
    jagged_ptr_offsets = offs_n[:, None] * stride_jn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_ptrs = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    dense_ptrs = Dense + clamped_offs_n[:, None] * stride_dk + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) and offs_d[None, :] < D
        jg = tl.load(Jagged + jagged_ptr_offsets, mask=mask)
        if SCALE_JAGGED:
            jg = jg * scale
        dn = tl.load(dense_ptrs, mask=mask)
        jg += dn
        tl.store(out_ptrs, jg, mask=mask)
        dense_ptrs += BLOCK_D
        out_ptrs += BLOCK_D
        offs_d += BLOCK_D
        jagged_ptr_offsets += BLOCK_D


_add_position_embeddings_kernel = register_tritoncc_specs(
    func=_add_position_embeddings_kernel,
    versioned_specs=_add_position_embeddings_tritoncc_named_specs(),
)
_add_position_embeddings_kernel = triton_autotune(
    configs=_add_position_embeddings_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)(_add_position_embeddings_kernel.fn)


@triton.jit
def _add_position_embeddings_bwd_kernel(
    Jagged,
    seq_offsets,
    high_inds,
    DenseOut,
    JaggedOut,
    B,
    D,
    scale,
    stride_jn,
    stride_jon,
    stride_don,
    SCALE_JAGGED: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_k = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off_b in range(0, B):
        max_ind = tl.load(high_inds + off_b)
        if off_k < max_ind:
            seq_start = tl.load(seq_offsets + off_b)
            jagged_ptr = (
                Jagged
                + seq_start.to(tl.int64) * stride_jn
                + off_k.to(tl.int64) * stride_jn
            )
            jagged_ptrs = jagged_ptr + offs_d
            jg = tl.load(
                jagged_ptrs,
                mask=offs_d < D,
            )
            accumulator += jg
            if SCALE_JAGGED:
                out_jagged_ptr = (
                    JaggedOut
                    + seq_start.to(tl.int64) * stride_jon
                    + off_k.to(tl.int64) * stride_jon
                )
                out_jagged_ptrs = out_jagged_ptr + offs_d
                tl.store(
                    out_jagged_ptrs,
                    jg * scale,
                    mask=offs_d < D,
                )
        elif off_k == max_ind:
            seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
            seq_end = tl.load(seq_offsets + off_b + 1)
            for k in range(seq_start + max_ind, seq_end):
                jagged_ptr = Jagged + k * stride_jn
                jagged_ptrs = jagged_ptr + offs_d
                jg = tl.load(
                    jagged_ptrs,
                    mask=offs_d < D,
                )
                accumulator += jg
                if SCALE_JAGGED:
                    out_jagged_ptr = JaggedOut + k * stride_jon
                    out_jagged_ptrs = out_jagged_ptr + offs_d
                    tl.store(
                        out_jagged_ptrs,
                        jg * scale,
                        mask=offs_d < D,
                    )
    out = accumulator.to(DenseOut.dtype.element_ty)
    out_ptrs = DenseOut + off_k * stride_don + offs_d
    tl.store(
        out_ptrs,
        out,
        mask=offs_d < D,
    )


class _AddPositionEmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        jagged: torch.Tensor,
        jagged_offsets: torch.Tensor,
        high_inds: torch.Tensor,
        max_seq_len: int,
        dense: torch.Tensor,
        scale: float = 1.0,
    ):
        jagged = _switch_to_contiguous_if_needed(jagged)
        dense = _switch_to_contiguous_if_needed(dense)
        L, D = jagged.shape
        assert len(dense.shape) == 2
        out = torch.empty_like(jagged)
        B = high_inds.size(0)
        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        _add_position_embeddings_kernel[grid](
            Jagged=jagged,
            seq_offsets=jagged_offsets,
            high_inds=high_inds,
            Dense=dense,
            Out=out,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            scale=scale,
            stride_jn=jagged.stride(0),
            stride_dk=dense.stride(0),
            stride_on=out.stride(0),
            SCALE_JAGGED=scale != 1.0,
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(jagged_offsets, high_inds)
        ctx.B = B
        ctx.D = D
        ctx.scale = scale
        ctx.K = dense.size(0)
        ctx.BLOCK_D = BLOCK_D
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None, None, torch.Tensor, None]:
        jagged_offsets, high_inds = ctx.saved_tensors
        d_dense = torch.empty((ctx.K, ctx.D), device=d_out.device, dtype=d_out.dtype)
        scale_jagged = ctx.scale != 1.0
        if scale_jagged:
            d_jagged = torch.empty_like(d_out)
        BLOCK_D = triton.next_power_of_2(ctx.D)
        _add_position_embeddings_bwd_kernel[(ctx.K,)](
            Jagged=d_out,
            seq_offsets=jagged_offsets,
            high_inds=high_inds,
            DenseOut=d_dense,
            JaggedOut=d_jagged if scale_jagged else None,  # pyre-ignore[61]
            B=ctx.B,
            D=ctx.D,
            scale=ctx.scale,
            stride_jn=d_out.stride(0),
            stride_jon=d_jagged.stride(0) if scale_jagged else 0,
            stride_don=d_dense.stride(0),
            SCALE_JAGGED=scale_jagged,
            BLOCK_D=BLOCK_D,
        )
        # pyre-ignore[61]
        return d_jagged if scale_jagged else d_out, None, None, None, d_dense, None


def _add_timestamp_position_embeddings_tritoncc_named_specs() -> List[VersionedSpec]:
    s: int = 16
    return (
        [
            VersionedSpec(
                spec={
                    "SeqEmb": (dtype, s),
                    "Offsets": ("*i64", s),
                    "Lengths": ("*i64", s),
                    "PosEmb": (dtype, s),
                    "TsEmb": (dtype, s),
                    "Out": (dtype, s),
                    "TS": ("*i64", s),
                    "PosInds": ("*i32", s),
                    "TsInds": ("*i32", s),
                    "NumTargets": ("*i64", s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "D": "i32",
                    "num_time_buckets": "i32",
                    "time_bucket_increments": "fp32",
                    "time_bucket_scale": "fp32",
                    "time_delta": "i32",
                    "max_contextual_seq_len": "i32",
                    "max_pos_ind": "i32",
                    "stride_sn": ("i32", s),
                    "stride_pn": ("i32", s),
                    "stride_tn": ("i32", s),
                    "stride_ts": "i32",
                    "stride_on": ("i32", s),
                    "TRAINING": False,
                    "HAS_MULTIPLE_TARGETS": has_multiple_targets,
                    "INTERLEAVE_TARGETS": interleave_targets,
                    "TIME_BUCKET_FN": time_bucket_fn,
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_N": -1,  # autotuned
                },
            )
            for dtype in ["*bf16", "*fp32", "*fp16"]
            for has_multiple_targets in [True, False]
            for interleave_targets in [True, False]
            for time_bucket_fn in ["log", "sqrt"]
            for BLOCK_D in [32, 64]
        ]
        + [
            VersionedSpec(
                spec={
                    "SeqEmb": (dtype, s),
                    "Offsets": ("*i64", s),
                    "Lengths": ("*i64", s),
                    "PosEmb": (dtype, s),
                    "TsEmb": (dtype, s),
                    "Out": (dtype, s),
                    "TS": ("*i64", s),
                    "PosInds": ("*i32", s),
                    "TsInds": ("*i32", s),
                    "NumTargets": ("*i64", s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "D": "i32",
                    "num_time_buckets": "i32",
                    "time_bucket_increments": "fp32",
                    "time_bucket_scale": "fp32",
                    "time_delta": "i32",
                    "max_contextual_seq_len": "i32",
                    "max_pos_ind": "i32",
                    "stride_sn": ("i32", s),
                    "stride_pn": ("i32", s),
                    "stride_tn": ("i32", s),
                    "stride_ts": "i32",
                    "stride_on": ("i32", s),
                    "TRAINING": False,
                    "HAS_MULTIPLE_TARGETS": has_multiple_targets,
                    "INTERLEAVE_TARGETS": interleave_targets,
                    "TIME_BUCKET_FN": time_bucket_fn,
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_N": -1,  # autotuned
                },
                version="standalone_cint_v5",
            )
            for dtype in ["*bf16", "*fp32", "*fp16"]
            for has_multiple_targets in [True, False]
            for interleave_targets in [True, False]
            for time_bucket_fn in ["log", "sqrt"]
            for BLOCK_D in [32, 64]
        ]
        + [
            VersionedSpec(
                spec={
                    "SeqEmb": (dtype, s),
                    "Offsets": ("*i64", s),
                    "Lengths": ("*i64", s),
                    "PosEmb": (dtype, s),
                    "TsEmb": (dtype, s),
                    "Out": (dtype, s),
                    "TS": ("*i64", s),
                    "PosInds": ("*i32", s),
                    "TsInds": ("*i32", s),
                    "NumTargets": ("*i64", s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "D": "i32",
                    "num_time_buckets": "i32",
                    "time_bucket_increments": "fp32",
                    "time_bucket_scale": "fp32",
                    "time_delta": "i32",
                    "max_contextual_seq_len": "i32",
                    "max_pos_ind": "i32",
                    "stride_sn": ("i32", s),
                    "stride_pn": ("i32", s),
                    "stride_tn": ("i32", s),
                    "stride_ts": "i32",
                    "stride_on": ("i32", s),
                    "TRAINING": False,
                    "HAS_MULTIPLE_TARGETS": has_multiple_targets,
                    "INTERLEAVE_TARGETS": interleave_targets,
                    "TIME_BUCKET_FN": time_bucket_fn,
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_N": -1,  # autotuned
                },
                version="standalone_cint_v1_time_position_emb",
            )
            for dtype in ["*bf16", "*fp32"]
            for has_multiple_targets in [True, False]
            for interleave_targets in [True, False]
            for time_bucket_fn in ["log", "sqrt"]
            for BLOCK_D in [32, 64]
        ]
        + [
            VersionedSpec(
                spec={
                    "SeqEmb": (dtype, s),
                    "Offsets": ("*i64", s),
                    "Lengths": ("*i64", s),
                    "PosEmb": (dtype, s),
                    "TsEmb": (dtype, s),
                    "Out": (dtype, s),
                    "TS": ("*i64", s),
                    "PosInds": ("*i32", s),
                    "TsInds": ("*i32", s),
                    "NumTargets": ("*i64", s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "D": "i32",
                    "num_time_buckets": "i32",
                    "time_bucket_increments": "fp32",
                    "time_bucket_scale": "fp32",
                    "time_delta": "i32",
                    "max_contextual_seq_len": "i32",
                    "max_pos_ind": "i32",
                    "stride_sn": ("i32", s),
                    "stride_pn": ("i32", s),
                    "stride_tn": ("i32", s),
                    "stride_ts": "i32",
                    "stride_on": ("i32", s),
                    "TRAINING": False,
                    "HAS_MULTIPLE_TARGETS": has_multiple_targets,
                    "INTERLEAVE_TARGETS": interleave_targets,
                    "TIME_BUCKET_FN": time_bucket_fn,
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_N": -1,  # autotuned
                },
                version="standalone_cint_v4",
            )
            for dtype in ["*bf16", "*fp32"]
            for has_multiple_targets in [True, False]
            for interleave_targets in [True, False]
            for time_bucket_fn in ["log", "sqrt"]
            for BLOCK_D in [32, 64]
        ]
        + [
            VersionedSpec(
                spec={
                    "SeqEmb": (dtype, s),
                    "Offsets": ("*i64", s),
                    "Lengths": ("*i64", s),
                    "PosEmb": (dtype, s),
                    "TsEmb": (dtype, s),
                    "Out": (dtype, s),
                    "TS": ("*i64", s),
                    "PosInds": ("*i32", s),
                    "TsInds": ("*i32", s),
                    "NumTargets": ("*i64", s),
                    "AUTOTUNE_MAX_SEQ_LEN": "i32",
                    "D": "i32",
                    "num_time_buckets": "i32",
                    "time_bucket_increments": "fp32",
                    "time_bucket_scale": "fp32",
                    "time_delta": "i32",
                    "max_contextual_seq_len": "i32",
                    "max_pos_ind": "i32",
                    "stride_sn": ("i32", s),
                    "stride_pn": ("i32", s),
                    "stride_tn": ("i32", s),
                    "stride_ts": "i32",
                    "stride_on": ("i32", s),
                    "TRAINING": False,
                    "HAS_MULTIPLE_TARGETS": has_multiple_targets,
                    "INTERLEAVE_TARGETS": interleave_targets,
                    "TIME_BUCKET_FN": time_bucket_fn,
                    "BLOCK_D": BLOCK_D,
                    "BLOCK_N": -1,  # autotuned
                },
                version="amd_standalone_cint_v2",
            )
            for dtype in ["*bf16", "*fp32"]
            for has_multiple_targets in [True, False]
            for interleave_targets in [True, False]
            for time_bucket_fn in ["log", "sqrt"]
            for BLOCK_D in [32, 64]
        ]
    )


@triton_autotune(
    configs=_add_position_embeddings_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _add_timestamp_position_embeddings_kernel(
    SeqEmb,
    Offsets,
    Lengths,
    PosEmb,
    TsEmb,
    Out,
    TS,
    PosInds,
    TsInds,
    NumTargets,
    AUTOTUNE_MAX_SEQ_LEN,
    D,
    num_time_buckets,
    time_bucket_increments,
    time_bucket_scale,
    time_delta,
    max_contextual_seq_len,
    max_pos_ind,
    stride_sn,
    stride_pn,
    stride_tn,
    stride_ts,
    stride_on,
    TRAINING: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    INTERLEAVE_TARGETS: tl.constexpr,
    TIME_BUCKET_FN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    SeqEmb has shape (sum_B(N_i), D),
    PosEmb has shape (N_p, D),
    TsEmb has shape (N_t, D),
    Out has shape (sum_B(N_i), D)
    """

    off_b = tl.program_id(0)
    off_n = tl.program_id(1)
    seq_start = tl.load(Offsets + off_b)
    seq_end = tl.load(Offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_n = off_n * BLOCK_N
    if start_n >= seq_len:
        return
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    seq_emb_offsets = offs_n[:, None] * stride_sn + offs_d[None, :]
    SeqEmb += seq_start.to(tl.int64) * stride_sn
    mask_n = offs_n < seq_len
    # position encoding
    seq_len = tl.load(Lengths + off_b)
    if HAS_MULTIPLE_TARGETS:
        num_targets = tl.load(NumTargets + off_b)
        if INTERLEAVE_TARGETS:
            high_ind = seq_len - num_targets * 2
        else:
            high_ind = seq_len - num_targets
    else:
        high_ind = seq_len
    pos_inds = tl.where(offs_n < high_ind, offs_n, high_ind)
    pos_inds = high_ind - pos_inds + max_contextual_seq_len
    pos_inds = tl.where(pos_inds < max_pos_ind - 1, pos_inds, max_pos_ind - 1)
    pos_inds = tl.where(offs_n < max_contextual_seq_len, offs_n, pos_inds)
    if TRAINING:
        tl.store(PosInds + seq_start + offs_n, pos_inds, mask=mask_n)
    pos_emb_offsets = pos_inds[:, None] * stride_pn + offs_d[None, :]
    # timestamp encoding
    ts = tl.load(TS + off_b * stride_ts + offs_n, mask=mask_n)
    query_time = tl.load(TS + off_b * stride_ts + seq_len - 1)
    ts = query_time - ts + time_delta
    ts = tl.where(ts > 1e-6, ts, 1e-6) / time_bucket_increments
    if TIME_BUCKET_FN == "log":
        ts = tl.log(ts)
    else:
        ts = tl.sqrt(ts)
    ts = ts * time_bucket_scale
    ts = ts.to(tl.int32)
    ts = tl.where(ts > 0, ts, 0)
    ts = tl.where(ts < num_time_buckets, ts, num_time_buckets)
    if TRAINING:
        tl.store(TsInds + seq_start + offs_n, ts, mask=mask_n)
    ts_emb_offsets = ts[:, None] * stride_tn + offs_d[None, :]
    Out += seq_start.to(tl.int64) * stride_on
    out_offsets = Out + offs_n[:, None] * stride_on + offs_d[None, :]
    for _d in range(0, D, BLOCK_D):
        mask = (offs_n[:, None] < seq_len) and offs_d[None, :] < D
        seq_emb = tl.load(SeqEmb + seq_emb_offsets, mask=mask)
        pos_emb = tl.load(PosEmb + pos_emb_offsets, mask=mask)
        ts_emb = tl.load(TsEmb + ts_emb_offsets, mask=mask)
        tl.store(out_offsets, seq_emb + (pos_emb + ts_emb).to(seq_emb.dtype), mask=mask)
        seq_emb_offsets += BLOCK_D
        pos_emb_offsets += BLOCK_D
        ts_emb_offsets += BLOCK_D
        out_offsets += BLOCK_D
        offs_d += BLOCK_D


_add_timestamp_position_embeddings_kernel = register_tritoncc_specs(
    func=_add_timestamp_position_embeddings_kernel,
    versioned_specs=_add_timestamp_position_embeddings_tritoncc_named_specs(),
)
_add_timestamp_position_embeddings_kernel = triton_autotune(
    configs=_add_position_embeddings_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN"],
)(_add_timestamp_position_embeddings_kernel.fn)


def bwd_pre_hook(nargs):
    nargs["Out"].zero_()


def _add_embeddings_bwd_configs() -> List[triton.Config]:
    configs = []
    for BLOCK in [32, 64, 128]:
        for num_stages in [2, 3, 4]:
            for num_warps in [2, 4, 8]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK": BLOCK,
                        },
                        num_stages=num_stages,
                        num_warps=num_warps,
                        pre_hook=bwd_pre_hook,
                    )
                )
    return configs


@triton_autotune(
    configs=_add_embeddings_bwd_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "AUTOTUNE_B", "D"],
)
@triton.jit
def _add_embeddings_bwd_kernel(
    In,
    KeyInds,
    ValueInds,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    B,
    AUTOTUNE_B,
    D,
    jagged_size,
    stride_in,
    stride_on,
    BLOCK_D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    off_block = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < D
    key_ind = -1
    key_ind = key_ind.to(KeyInds.dtype.element_ty)  # pyre-ignore[16]
    accumulator = tl.zeros((BLOCK_D,), dtype=In.dtype.element_ty)
    for off_i in range(0, BLOCK):
        off = off_block * BLOCK + off_i
        if off < jagged_size:
            value_ind = tl.load(ValueInds + off)
            in_offset = In + value_ind.to(tl.int64) * stride_in
            jagged_in = tl.load(in_offset + offs_d, mask=mask_d)
            key_ind_new = tl.load(KeyInds + off)
            if key_ind == key_ind_new:
                accumulator += jagged_in
            else:
                if key_ind >= 0:
                    out_offset = Out + key_ind.to(tl.int64) * stride_on
                    tl.atomic_add(
                        out_offset + offs_d,
                        accumulator.to(Out.dtype.element_ty),
                        mask=mask_d,
                        sem="relaxed",
                    )
                key_ind = key_ind_new
                accumulator = jagged_in
    if key_ind >= 0:
        out_offset = Out + key_ind.to(tl.int64) * stride_on
        tl.atomic_add(
            out_offset + offs_d,
            accumulator.to(Out.dtype.element_ty),
            mask=mask_d,
            sem="relaxed",
        )


class _AddTimestampPositionEmbeddingsFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        pos_embeddings: torch.Tensor,
        ts_embeddings: torch.Tensor,
        timestamps: torch.Tensor,
        max_seq_len: int,
        max_contextual_seq_len: int,
        seq_lengths: torch.Tensor,
        num_targets: Optional[torch.Tensor],
        interleave_targets: bool,
        time_bucket_fn: str,
    ):
        seq_embeddings = _switch_to_contiguous_if_needed(seq_embeddings)
        pos_embeddings = _switch_to_contiguous_if_needed(pos_embeddings)
        ts_embeddings = _switch_to_contiguous_if_needed(ts_embeddings)

        max_pos_ind = pos_embeddings.shape[0]
        B = seq_lengths.shape[0]
        assert timestamps.shape[0] == B, "shape[0] of timestamps much match batch size"
        assert (
            timestamps.shape[1] >= max_seq_len
        ), "shape[1] of timestamps much >= max_seq_len"

        N, D = seq_embeddings.shape
        assert len(pos_embeddings.shape) == 2
        assert len(ts_embeddings.shape) == 2
        assert (
            pos_embeddings.shape[1] == D
        ), "shape[1] of pos_embeddings much match seq_embeddings"
        assert (
            ts_embeddings.shape[1] == D
        ), "shape[1] of ts_embeddings much match seq_embeddings"
        out = torch.empty_like(seq_embeddings)

        timestamps = _switch_to_contiguous_if_needed(timestamps)
        ts_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)
        pos_inds = torch.empty_like(seq_embeddings[:, 0], dtype=torch.int32)

        grid = lambda meta: (  # noqa E731
            B,
            triton.cdiv(max_seq_len, meta["BLOCK_N"]),
        )
        BLOCK_D = triton.next_power_of_2(D) if D < 64 else 64
        _add_timestamp_position_embeddings_kernel[grid](
            SeqEmb=seq_embeddings,
            Offsets=seq_offsets,
            Lengths=seq_lengths,
            PosEmb=pos_embeddings,
            TsEmb=ts_embeddings,
            Out=out,
            TS=timestamps,
            PosInds=pos_inds,
            TsInds=ts_inds,
            NumTargets=num_targets,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
            D=D,
            num_time_buckets=2048,
            time_bucket_increments=60.0,
            time_bucket_scale=1.0,
            time_delta=0,
            max_contextual_seq_len=max_contextual_seq_len,
            max_pos_ind=max_pos_ind,
            stride_sn=seq_embeddings.stride(0),
            stride_pn=pos_embeddings.stride(0),
            stride_tn=ts_embeddings.stride(0),
            stride_ts=timestamps.stride(0),
            stride_on=out.stride(0),
            TRAINING=True,
            HAS_MULTIPLE_TARGETS=num_targets is not None,
            INTERLEAVE_TARGETS=interleave_targets,
            TIME_BUCKET_FN=time_bucket_fn,
            BLOCK_D=BLOCK_D,
        )
        values = torch.arange(0, N, dtype=torch.int32, device=timestamps.device)
        sorted_ts_key_inds, sorted_ts_value_inds = torch.ops.hammer.sort_kv_pairs(
            ts_inds, values
        )
        sorted_pos_key_inds, sorted_pos_value_inds = torch.ops.hammer.sort_kv_pairs(
            pos_inds, values
        )
        ctx.save_for_backward(
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
        )
        ctx.B = B
        ctx.D = D
        ctx.max_seq_len = max_seq_len
        ctx.pos_emb_size = pos_embeddings.shape[0]
        ctx.ts_emb_size = ts_embeddings.shape[0]
        ctx.pos_dtype = pos_embeddings.dtype
        ctx.ts_dtype = ts_embeddings.dtype
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        None,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        (
            sorted_pos_key_inds,
            sorted_pos_value_inds,
            sorted_ts_key_inds,
            sorted_ts_value_inds,
        ) = ctx.saved_tensors
        d_pos_embeddings = torch.empty(
            (ctx.pos_emb_size, ctx.D), device=d_out.device, dtype=torch.float32
        )
        d_ts_embeddings = torch.empty(
            (ctx.ts_emb_size, ctx.D), device=d_out.device, dtype=torch.float32
        )
        grid = lambda meta: (triton.cdiv(d_out.shape[0], meta["BLOCK"]),)  # noqa E731
        B = ctx.B
        AUTOTUNE_B = prev_power_of_2(B)
        _add_embeddings_bwd_kernel[grid](
            In=d_out,
            KeyInds=sorted_pos_key_inds,
            ValueInds=sorted_pos_value_inds,
            Out=d_pos_embeddings,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            B=B,
            AUTOTUNE_B=AUTOTUNE_B,
            D=ctx.D,
            jagged_size=d_out.shape[0],
            stride_in=d_out.stride(0),
            stride_on=d_pos_embeddings.stride(0),
            BLOCK_D=triton.next_power_of_2(ctx.D),
        )
        _add_embeddings_bwd_kernel[grid](
            In=d_out,
            KeyInds=sorted_ts_key_inds,
            ValueInds=sorted_ts_value_inds,
            Out=d_ts_embeddings,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            B=B,
            AUTOTUNE_B=AUTOTUNE_B,
            D=ctx.D,
            jagged_size=d_out.shape[0],
            stride_in=d_out.stride(0),
            stride_on=d_ts_embeddings.stride(0),
            BLOCK_D=triton.next_power_of_2(ctx.D),
        )
        return (
            d_out,
            None,
            d_pos_embeddings.to(ctx.pos_dtype),
            d_ts_embeddings.to(ctx.ts_dtype),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

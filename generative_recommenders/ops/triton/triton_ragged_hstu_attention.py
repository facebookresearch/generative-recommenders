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
        NamedSpecType,
        prev_power_of_2,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )
except ImportError:
    from hammer.oss.generative_recommenders.ops.triton.utils import (
        _switch_to_contiguous_if_needed,
        autotune_max_seq_len,
        NamedSpecType,
        prev_power_of_2,
        register_tritoncc_specs,
        triton_autotune,
        VersionedSpec,
    )

try:
    # @manual=//triton:triton
    from triton.language.extra.libdevice import fast_dividef
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        # pyre-ignore: Undefined import [21]
        # @manual=//triton:triton
        from triton.language.math import fast_dividef


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [0, 1]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2]:
                                configs.append(
                                    triton.Config(
                                        {
                                            "BLOCK_M": BLOCK_M,
                                            "BLOCK_N": BLOCK_N,
                                            "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                            "waves_per_eu": waves_per_eu,
                                        },
                                        num_stages=num_stages,
                                        num_warps=num_warps,
                                    )
                                )
    else:
        configs = [
            triton.Config(
                {"BLOCK_M": 16, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 32, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=2,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 32},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=2,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64},
                num_stages=4,
                num_warps=8,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128},
                num_stages=2,
                num_warps=8,
            ),
        ]
    return configs


@triton.jit
def _ragged_hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    mask_m,
    mask_n,
    q,
    K_block_ptr,
    V_block_ptr,
    n_targets,
    ts_1_ptrs,
    ts_0,
    TW,
    PW,
    alpha,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    bias_ptrs,
    MAX_ATTN_LEN: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == "lower_triangular":
            offs_m = tl.where(
                offs_m < seq_len - n_targets,
                offs_m,
                seq_len - n_targets,
            )
            offs_n = tl.where(
                offs_n < seq_len - n_targets,
                offs_n,
                seq_len - n_targets,
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            offs_m = tl.where(offs_m > n_targets - 1, offs_m, n_targets - 1)
            offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
    offs_n_minus_m = offs_n[None, :] - offs_m[:, None]
    if MAX_ATTN_LEN > 0:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask = invalid_mask or (
                offs_n_minus_m < 0 and offs_n_minus_m >= -MAX_ATTN_LEN
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask = invalid_mask or (
                offs_n_minus_m > 0 and offs_n_minus_m <= MAX_ATTN_LEN
            )
    else:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask = invalid_mask or offs_n_minus_m < 0
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask = invalid_mask or offs_n_minus_m > 0
    if CONTEXTUAL_SEQ_LEN > 0:
        if INVALID_MASK_TYPE == "lower_triangular":
            # offs_m[:, None]: [BLOCK_M, BLOCK_N] global row indices shortcut at seq_len - n_targets
            # offs_n[None, :]: [BLOCK_M, BLOCK_N] global col indices shortcut at seq_len - n_targets
            row_filter = offs_m < CONTEXTUAL_SEQ_LEN
            if HAS_MULTIPLE_TARGETS:
                col_filter = offs_n < seq_len - n_targets
            else:
                col_filter = offs_n < seq_len
            invalid_mask = invalid_mask or (row_filter[:, None] and col_filter[None, :])
    if ATTN_BIAS_TYPE == "fused":
        attn_bias = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_1 = tl.load(ts_1_ptrs + start_n, mask=mask_n)
            else:
                ts_1 = tl.load(ts_1_ptrs + start_n + 1, mask=mask_n)
            ts = ts_0[:, None] - ts_1[None, :]
            ts = ts + time_delta
            ts = tl.where(ts > 1e-6, ts, 1e-6)
            ts = ts * (1.0 / time_bucket_incr)
            if BUCKET_FN == "log":
                ts = tl.log(ts)
            elif BUCKET_FN == "sqrt":
                ts = tl.sqrt(ts)
            ts = ts * (1.0 / time_bucket_div)
            ts = ts.to(tl.int32)
            ts = tl.where(ts > 0, ts, 0)
            ts = tl.where(ts < num_buckets, ts, num_buckets)
            ts_w = tl.load(
                TW + ts,
                mask=mask_m[:, None] and mask_n[None, :],
            )
            attn_bias = attn_bias + ts_w
        if USE_POS_BIAS:
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n_minus_m + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(
                    offs_pos_w < 2 * max_pos_ind - 2,
                    offs_pos_w,
                    2 * max_pos_ind - 2,
                )
            else:
                offs_pos_w = offs_n_minus_m + MAX_SEQ_LEN - 1
            pos_w = tl.load(
                PW + offs_pos_w,
                mask=mask_m[:, None] and mask_n[None, :],
            )
            attn_bias = attn_bias + pos_w
        qk = qk + attn_bias
    elif ATTN_BIAS_TYPE == "separate":
        attn_bias = tl.load(
            bias_ptrs + start_n,
            mask=mask_m[:, None] & mask_n[None, :],
            other=0.0,
        )
        qk = qk + attn_bias
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _ragged_hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    delta_x_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    stride_oh,
    alpha,
    Z,
    H,
    MAX_SEQ_LEN,
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    off_z,
    off_h,
    pid,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        delta_start = tl.load(delta_x_offsets + off_z * DeltaSize)
        start_m = (start_m_delta + delta_start - seq_start).to(tl.int32)
    else:
        start_m_delta = 0
        start_m = pid * BLOCK_M
    if start_m < seq_len:
        if HAS_MULTIPLE_TARGETS:
            n_targets = tl.load(num_targets + off_z).to(tl.int32)
        else:
            n_targets = None

        # initialize offsets
        offs_m = start_m + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        if IS_DELTA_Q:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + off_z * DeltaSize * stride_qm,
                shape=(DeltaSize, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m_delta, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        else:
            Q_block_ptr = tl.make_block_ptr(
                base=Q + off_h * stride_qh + seq_start * stride_qm,
                shape=(seq_len, BLOCK_D_Q),
                strides=(stride_qm, 1),
                offsets=(start_m, 0),
                block_shape=(BLOCK_M, BLOCK_D_Q),
                order=(1, 0),
            )
        K_block_ptr = tl.make_block_ptr(
            base=K + off_h * stride_kh + seq_start * stride_kn,
            shape=(BLOCK_D_Q, seq_len),
            strides=(1, stride_kn),
            offsets=(0, 0),
            block_shape=(BLOCK_D_Q, BLOCK_N),
            order=(0, 1),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + off_h * stride_vh + seq_start * stride_vn,
            shape=(seq_len, BLOCK_D_V),
            strides=(stride_vn, 1),
            offsets=(0, 0),
            block_shape=(BLOCK_N, BLOCK_D_V),
            order=(1, 0),
        )
        mask_m = offs_m < seq_len
        if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS:
            ts_0_ptrs = TS + off_z * stride_ts + offs_m
            ts_1_ptrs = TS + off_z * stride_ts + offs_n
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
        elif ATTN_BIAS_TYPE == "separate":
            seq2_start = tl.load(seq2_offsets + off_z)
            bias_start = seq2_start * H + off_h * seq_len * seq_len
            off_bias = offs_m[:, None] * seq_len + offs_n[None, :]
            bias_ptrs = Bias + bias_start + off_bias

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if INVALID_MASK_TYPE == "lower_triangular":
            if HAS_MULTIPLE_TARGETS:
                if MAX_ATTN_LEN > 0:
                    start_m_index = (
                        seq_len - n_targets
                        if start_m > seq_len - n_targets
                        else start_m
                    )
                    low = start_m_index - MAX_ATTN_LEN
                    low = low if low > 0 else 0
                else:
                    low = 0
                uih_end = (seq_len - n_targets + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                if uih_end < start_m:
                    high = seq_len - n_targets
                else:
                    high = start_m + BLOCK_M
                if CONTEXTUAL_SEQ_LEN > 0:
                    if start_m < CONTEXTUAL_SEQ_LEN:
                        high = seq_len - n_targets
            else:
                if MAX_ATTN_LEN > 0:
                    low = start_m - MAX_ATTN_LEN
                    low = low if low > 0 else 0
                else:
                    low = 0
                high = start_m + BLOCK_M
                if CONTEXTUAL_SEQ_LEN > 0:
                    if start_m < CONTEXTUAL_SEQ_LEN:
                        high = seq_len
        elif INVALID_MASK_TYPE == "upper_triangular":
            low = start_m
            high = seq_len
        # pyre-ignore[61]
        if low > 0:
            # pyre-ignore[61]
            K_block_ptr = tl.advance(K_block_ptr, (0, low))
            # pyre-ignore[61]
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        # pyre-ignore[61]
        for start_n in range(low, high, BLOCK_N):
            cur_offs_n = offs_n + start_n
            mask_n = cur_offs_n < seq_len
            acc += _ragged_hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=cur_offs_n,
                mask_m=mask_m,
                mask_n=mask_n,
                q=q,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                ts_1_ptrs=(
                    # pyre-ignore[61]
                    ts_1_ptrs
                    if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                    else None
                ),
                # pyre-ignore[61]
                ts_0=ts_0 if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS else None,
                TW=TW,
                PW=PW,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                # pyre-ignore[61]
                bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                IS_DELTA_Q=IS_DELTA_Q,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if HAS_MULTIPLE_TARGETS and INVALID_MASK_TYPE == "lower_triangular":
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - uih_end).to(tl.int32)  # pyre-ignore [61]
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    cur_offs_n = offs_n + start_delta
                    mask_n = cur_offs_n < seq_len
                    acc += _ragged_hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=cur_offs_n,
                        mask_m=mask_m,
                        mask_n=mask_n,
                        q=q,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        ts_1_ptrs=(
                            # pyre-ignore[61]
                            ts_1_ptrs
                            if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                            else None
                        ),
                        ts_0=(
                            # pyre-ignore[61]
                            ts_0
                            if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS
                            else None
                        ),
                        TW=TW,
                        PW=PW,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        num_buckets=num_buckets,
                        max_pos_ind=max_pos_ind,
                        MAX_ATTN_LEN=MAX_ATTN_LEN,
                        time_bucket_incr=time_bucket_incr,
                        time_bucket_div=time_bucket_div,
                        time_delta=time_delta,
                        # pyre-ignore[61]
                        bias_ptrs=bias_ptrs if ATTN_BIAS_TYPE == "separate" else None,
                        CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                        CAUSAL=CAUSAL,
                        BUCKET_FN=BUCKET_FN,
                        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                        USE_TIME_BIAS=USE_TIME_BIAS,
                        USE_POS_BIAS=USE_POS_BIAS,
                        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        IS_DELTA_Q=IS_DELTA_Q,
                        ALLOW_TF32=ALLOW_TF32,
                        BLOCK_M=BLOCK_M,
                        BLOCK_N=BLOCK_N,
                    )
                    K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
                    V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

        if IS_DELTA_Q:
            start_m_delta = pid * BLOCK_M
            offs_m_delta = start_m_delta + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + off_z * DeltaSize * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m_delta[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m_delta < DeltaSize)[:, None])
        else:
            # rematerialize offsets to save registers
            start_m = pid * BLOCK_M
            offs_m = start_m + tl.arange(0, BLOCK_M)
            offs_v_d = tl.arange(0, BLOCK_D_V)
            off_o = Out + seq_start * stride_om + off_h * stride_oh
            out_ptrs = off_o + offs_m[:, None] * stride_om + offs_v_d[None, :]
            tl.store(out_ptrs, acc, mask=(offs_m < seq_len)[:, None])


@triton.autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _ragged_hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    delta_x_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _ragged_hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        TS=TS,
        TW=TW,
        PW=PW,
        Bias=Bias,
        seq2_offsets=seq2_offsets,
        delta_x_offsets=delta_x_offsets,
        num_targets=num_targets,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_ts=stride_ts,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        Z=Z,
        H=H,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        INVALID_MASK_TYPE=INVALID_MASK_TYPE,
        CAUSAL=CAUSAL,
        BUCKET_FN=BUCKET_FN,
        ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
        USE_TIME_BIAS=USE_TIME_BIAS,
        USE_POS_BIAS=USE_POS_BIAS,
        HAS_MAX_POS_IND=HAS_MAX_POS_IND,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        MAX_ATTN_LEN=MAX_ATTN_LEN,
        CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )


@triton.autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _ragged_hstu_attn_fwd_persistent(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    delta_x_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_om,
    stride_oh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    DeltaSize,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    n_tile_num = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
    prog_id = tl.program_id(0)
    num_progs = tl.num_programs(0)

    total_tiles = n_tile_num * Z * H

    tiles_per_sm = total_tiles // num_progs
    if prog_id < total_tiles % num_progs:
        tiles_per_sm += 1

    tile_idx = prog_id
    for _ in range(0, tiles_per_sm):
        pid = (total_tiles - tile_idx - 1) // (Z * H)
        off_hz = (total_tiles - tile_idx - 1) % (Z * H)
        off_z = off_hz // H
        off_h = off_hz % H
        _ragged_hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            seq_offsets=seq_offsets,
            TS=TS,
            TW=TW,
            PW=PW,
            Bias=Bias,
            seq2_offsets=seq2_offsets,
            delta_x_offsets=delta_x_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_ts=stride_ts,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            Z=Z,
            H=H,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DimQ=DimQ,
            DimV=DimV,
            DeltaSize=DeltaSize,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
            CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        tile_idx += num_progs


def _get_named_specs() -> List[VersionedSpec]:
    s: int = 16
    INVALID_MASK_TYPE: str = "lower_triangular"
    CAUSAL: bool = True
    USE_TIME_BIAS: bool = True
    USE_POS_BIAS: bool = True

    def _common_specs(dtype: str = "*bf16") -> NamedSpecType:
        return {
            "Q": (dtype, s),
            "K": (dtype, s),
            "V": (dtype, s),
            "seq_offsets": ("*i64", s),
            "TS": ("*i64", s),
            "Bias": (dtype, s, False),
            "seq2_offsets": ("*i64", s, False),
            "Out": (dtype, s),
            "stride_qm": ("i32", s),
            "stride_qh": ("i32", s),
            "stride_kn": ("i32", s),
            "stride_kh": ("i32", s),
            "stride_vn": ("i32", s),
            "stride_vh": ("i32", s),
            "stride_ts": "i32",
            "stride_om": ("i32", s),
            "stride_oh": ("i32", s),
            "alpha": "fp32",
            "Z": "i32",
            "AUTOTUNE_Z": "i32",
            "H": "i32",
            "MAX_SEQ_LEN": "i32",
            "AUTOTUNE_MAX_SEQ_LEN": "i32",
            "DimQ": "i32",
            "DimV": "i32",
            "DeltaSize": "i32",
            "num_buckets": "i32",
            "max_pos_ind": "i32",
            "time_bucket_incr": "fp32",
            "time_bucket_div": "fp32",
            "time_delta": "fp32",
            "sort_by_length_indices": ("*i64", s, False),
            "INVALID_MASK_TYPE": INVALID_MASK_TYPE,
            "CAUSAL": CAUSAL,
            "USE_TIME_BIAS": USE_TIME_BIAS,
            "USE_POS_BIAS": USE_POS_BIAS,
            "BLOCK_M": -1,  # autotuned
            "BLOCK_N": -1,  # autotuned
            "MAX_ATTN_LEN": 0,
            "CONTEXTUAL_SEQ_LEN": 0,
            "HAS_SORT_BY_LENGTH_INDICES": False,
        }

    default_values = {
        "MAX_ATTN_LEN": 0,
        "CONTEXTUAL_SEQ_LEN": 0,
        "HAS_SORT_BY_LENGTH_INDICES": 0,
    }

    return (
        [
            VersionedSpec(
                spec={
                    "TW": ("*bf16", s),
                    "PW": ("*bf16", s),
                    "delta_x_offsets": ("*i64", s, False),
                    "num_targets": ("*i64", s, False),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": False,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": False,
                    "BLOCK_D_Q": 128,
                    "BLOCK_D_V": 128,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                # `(dict, name)` denotes the version with name will be used in production,
                # and the kernel with this spec will be added to predictor.
                version="standalone_magic",
            )
            for has_max_pos_ind in [False, True]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": ("*bf16", s),
                    "PW": ("*bf16", s),
                    "delta_x_offsets": ("*i64", s, False),
                    "num_targets": ("*i64", s, False),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": False,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": False,
                    "BLOCK_D_Q": block_dq,
                    "BLOCK_D_V": block_dv,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
            )
            for block_dq, block_dv in [(128, 128), (32, 64)]
            for has_max_pos_ind in [False, True]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i64", s, True),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
            )
            for has_max_pos_ind in [True, False]
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i64", s, True),
                    "HAS_MAX_POS_IND": False,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
            )
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i64", s, True),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v1",
            )
            for has_max_pos_ind in [True, False]
            for block in [64, 128]
            for is_delta_q in [False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i64", s, True),
                    "HAS_MAX_POS_IND": False,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v1",
            )
            for block in [64, 128]
            for is_delta_q in [False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v2",
            )
            for has_max_pos_ind in [True, False]
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": False,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v2",
            )
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            # standalone magic model
            VersionedSpec(
                spec={
                    "TW": ("*bf16", s),
                    "PW": ("*bf16", s),
                    "delta_x_offsets": ("*i64", s, False),
                    "num_targets": ("*i64", s, False),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": False,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": False,
                    "BLOCK_D_Q": block_dq,
                    "BLOCK_D_V": block_dv,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v2",
            )
            for block_dq, block_dv in [(32, 64)]
            for has_max_pos_ind in [False, True]
        ]
        + [
            # standalone magic model without RAB
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, False),
                    "num_targets": ("*i64", s, False),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": False,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": False,
                    "BLOCK_D_Q": block_dq,
                    "BLOCK_D_V": block_dv,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v2",
            )
            for block_dq, block_dv in [(32, 64)]
            for has_max_pos_ind in [False, True]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": ("*bf16", s),
                    "PW": ("*bf16", s),
                    "delta_x_offsets": ("*i64", s, False),
                    "num_targets": ("*i32", s, False),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": False,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": False,
                    "BLOCK_D_Q": block_dq,
                    "BLOCK_D_V": block_dv,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="amd_standalone_cint_v2",
            )
            for block_dq, block_dv in [(128, 128), (32, 64)]
            for has_max_pos_ind in [False, True]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="amd_standalone_cint_v2",
            )
            for has_max_pos_ind in [True, False]
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": False,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="amd_standalone_cint_v2",
            )
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            # with RAB
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": has_max_pos_ind,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "fused",
                    "BUCKET_FN": "sqrt",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v4",
            )
            for has_max_pos_ind in [True, False]
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
        + [
            # no RAB
            VersionedSpec(
                spec={
                    "TW": "*bf16",
                    "PW": "*bf16",
                    "delta_x_offsets": ("*i64", s, is_delta_q),
                    "num_targets": ("*i32", s, True),
                    "HAS_MAX_POS_IND": False,
                    "HAS_MULTIPLE_TARGETS": True,
                    "ATTN_BIAS_TYPE": "none",
                    "BUCKET_FN": "none",
                    "IS_DELTA_Q": is_delta_q,
                    "BLOCK_D_Q": block,
                    "BLOCK_D_V": block,
                    "ALLOW_TF32": True,
                    **_common_specs(dtype="*bf16"),
                },
                default_values=default_values,
                version="standalone_cint_v4",
            )
            for block in [64, 128]
            for is_delta_q in [True, False]
        ]
    )


_ragged_hstu_attn_fwd = register_tritoncc_specs(
    func=_ragged_hstu_attn_fwd, versioned_specs=_get_named_specs()
)
_ragged_hstu_attn_fwd = triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)(_ragged_hstu_attn_fwd.fn)

_ragged_hstu_attn_fwd_persistent = register_tritoncc_specs(
    func=_ragged_hstu_attn_fwd_persistent, versioned_specs=_get_named_specs()
)
_ragged_hstu_attn_fwd_persistent = triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)(_ragged_hstu_attn_fwd_persistent.fn)


@triton.jit
def _ragged_hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    mask_n,
    ts_0_ptrs,
    ts_1,
    bias_ptrs_trans,
    dbias_ptrs_trans,
    do_ptrs,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    n_targets,
    TW,
    PW,
    DTW,
    DPW,
    LOCK,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    MAX_ATTN_LEN: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == "lower_triangular":
            pos_offs_m = tl.where(
                pos_offs_m < seq_len - n_targets,
                pos_offs_m,
                seq_len - n_targets,
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            pos_offs_m = tl.where(pos_offs_m > n_targets - 1, pos_offs_m, n_targets - 1)
    q_trans = tl.load(
        q_ptrs_trans + start_m * stride_qm,
        mask=mask_m[None, :],
        other=0.0,
    )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    if ATTN_BIAS_TYPE == "fused":
        attn_bias_trans = tl.zeros([BLOCK_N, BLOCK_M], dtype=tl.float32)
        if USE_TIME_BIAS:
            if CAUSAL:
                ts_0 = tl.load(ts_0_ptrs + start_m + 1, mask=mask_m)
            else:
                ts_0 = tl.load(ts_0_ptrs + start_m, mask=mask_m)
            ts_trans = ts_0[None, :] - ts_1[:, None]
            ts_trans = ts_trans + time_delta
            ts_trans = tl.where(ts_trans > 1e-6, ts_trans, 1e-6)
            ts_trans = ts_trans * (1.0 / time_bucket_incr)
            if BUCKET_FN == "log":
                ts_trans = tl.log(ts_trans)
            elif BUCKET_FN == "sqrt":
                ts_trans = tl.sqrt(ts_trans)
            ts_trans = ts_trans * (1.0 / time_bucket_div)
            ts_trans = ts_trans.to(tl.int32)
            ts_trans = tl.where(ts_trans > 0, ts_trans, 0)
            ts_trans = tl.where(ts_trans < num_buckets, ts_trans, num_buckets)
            ts_w_trans = tl.load(
                TW + ts_trans,
                mask=mask_m[None, :] and mask_n[:, None],
            )
            attn_bias_trans = attn_bias_trans + ts_w_trans
        if USE_POS_BIAS:
            offs_pos_w_trans = None
            if HAS_MAX_POS_IND:
                offs_pos_w_trans = (
                    pos_offs_n[:, None] - pos_offs_m[None, :] + max_pos_ind - 1
                )
                offs_pos_w_trans = tl.where(offs_pos_w_trans > 0, offs_pos_w_trans, 0)
                offs_pos_w_trans = tl.where(
                    offs_pos_w_trans < 2 * max_pos_ind - 2,
                    offs_pos_w_trans,
                    2 * max_pos_ind - 2,
                )
            else:
                offs_pos_w_trans = (
                    pos_offs_n[:, None] - pos_offs_m[None, :] + MAX_SEQ_LEN - 1
                )
            pos_w_trans = tl.load(
                PW + offs_pos_w_trans,
                mask=mask_m[None, :] and mask_n[:, None],
            )
            attn_bias_trans = attn_bias_trans + pos_w_trans
        qk_trans = qk_trans + attn_bias_trans
    elif ATTN_BIAS_TYPE == "separate":
        attn_bias_trans = tl.load(
            bias_ptrs_trans + start_m * seq_len,
            mask=mask_m[None, :] & mask_n[:, None],
            other=0.0,
        )
        qk_trans = qk_trans + attn_bias_trans
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    if MAX_ATTN_LEN > 0:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask_trans = invalid_mask_trans or (
                pos_offs_m[None, :] > pos_offs_n[:, None]
                and pos_offs_n[:, None] - pos_offs_m[None, :] >= -MAX_ATTN_LEN
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask_trans = invalid_mask_trans or (
                pos_offs_m[None, :] < pos_offs_n[:, None]
                and pos_offs_n[:, None] - pos_offs_m[None, :] <= MAX_ATTN_LEN
            )
    else:
        if INVALID_MASK_TYPE == "lower_triangular":
            invalid_mask_trans = (
                invalid_mask_trans or pos_offs_m[None, :] > pos_offs_n[:, None]
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            invalid_mask_trans = (
                invalid_mask_trans or pos_offs_m[None, :] < pos_offs_n[:, None]
            )
    if CONTEXTUAL_SEQ_LEN > 0 and INVALID_MASK_TYPE == "lower_triangular":
        row_filter = pos_offs_m < CONTEXTUAL_SEQ_LEN
        if HAS_MULTIPLE_TARGETS:
            col_filter = pos_offs_n < seq_len - n_targets
        else:
            col_filter = pos_offs_n < seq_len
        invalid_mask_trans = invalid_mask_trans or (
            row_filter[None, :] and col_filter[:, None]
        )
    silu_trans = tl.where(invalid_mask_trans, silu_trans, 0)
    silu_trans = silu_trans.to(k.dtype)
    # compute dv
    do = tl.load(
        do_ptrs + start_m * stride_dom,
        mask=mask_m[:, None],
        other=0.0,
    )
    dv += tl.dot(silu_trans, do, allow_tf32=ALLOW_TF32)

    # compute dk and dq
    dqk_trans = tl.dot(v, tl.trans(do), allow_tf32=ALLOW_TF32)
    dqk_trans = (
        dqk_trans * sig_trans * (1 + qk_trans * (1 - sig_trans)) * (1.0 / MAX_SEQ_LEN)
    )
    dqk_trans = tl.where(invalid_mask_trans, dqk_trans, 0)
    dqk_trans = dqk_trans.to(k.dtype)

    if ATTN_BIAS_TYPE == "fused" and FUSED_BIAS_BWD:
        if USE_TIME_BIAS:
            tl.atomic_add(
                # pyre-ignore[61]
                DTW + ts_trans,
                dqk_trans,
                mask=mask_m[None, :] & mask_n[:, None] & invalid_mask_trans,
                sem="relaxed",
            )
        if USE_POS_BIAS:
            tl.atomic_add(
                # pyre-ignore[61]
                DPW + offs_pos_w_trans,
                dqk_trans,
                mask=mask_m[None, :] & mask_n[:, None] & invalid_mask_trans,
                sem="relaxed",
            )
    elif ATTN_BIAS_TYPE == "separate":
        tl.store(
            dbias_ptrs_trans + start_m * seq_len,
            dqk_trans,
            mask=mask_m[None, :] & mask_n[:, None],
        )
    # Note: the factor `alpha` is delayed until the end of the function to reduce the cost
    dk += tl.dot(dqk_trans, tl.trans(q_trans), allow_tf32=ALLOW_TF32)
    if ATOMIC_ADD:
        lock_id = start_m // BLOCK_M
        stride_lock = tl.cdiv(MAX_SEQ_LEN, BLOCK_M)
        lock = LOCK + tl.program_id(0) * stride_lock + lock_id
        tl.debug_barrier()  # add a barrier to force sync
        while tl.atomic_cas(lock, 0, 1) == 1:
            pass
    dq_trans = tl.load(
        dq_ptrs_trans + start_m * stride_dqm,
        mask=mask_m[None, :],
        other=0.0,
        eviction_policy="evict_last",
    )
    dq_trans += tl.dot(tl.trans(k), dqk_trans, allow_tf32=ALLOW_TF32) * alpha
    dq_trans = dq_trans.to(k.dtype)
    tl.store(
        dq_ptrs_trans + start_m * stride_dqm,
        dq_trans,
        mask=mask_m[None, :],
        eviction_policy="evict_last",
    )
    if ATOMIC_ADD:
        tl.atomic_xchg(lock, 0)  # pyre-ignore [61]
    return dk, dv


@triton.jit
def _ragged_hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    Q,
    K,
    V,
    TS,
    TW,
    PW,
    Bias,
    DOut,
    DQ,
    DK,
    DV,
    DBias,
    DTW,
    DPW,
    LOCK,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    alpha,
    MAX_SEQ_LEN,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    MAX_ATTN_LEN: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if INVALID_MASK_TYPE == "lower_triangular":
        if HAS_MULTIPLE_TARGETS:
            low = start_n
            if MAX_ATTN_LEN > 0:
                high = start_n + MAX_ATTN_LEN + BLOCK_N
                high = high if high + n_targets < seq_len else seq_len
            else:
                high = seq_len
        else:
            low = start_n
            if MAX_ATTN_LEN > 0:
                high = start_n + MAX_ATTN_LEN + BLOCK_N
                high = high if high < seq_len else seq_len
            else:
                high = seq_len
        if CONTEXTUAL_SEQ_LEN > 0:
            contextual_block_end = tl.cdiv(CONTEXTUAL_SEQ_LEN, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    elif INVALID_MASK_TYPE == "upper_triangular":
        low = 0
        high = start_n + BLOCK_N

    # initialize row/col offsets
    offs_m = tl.arange(0, BLOCK_M)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    offs_n = start_n + tl.arange(0, BLOCK_N)

    # initialize pointers to value-like data
    q_ptrs_trans = Q + (offs_m[None, :] * stride_qm + offs_qk_d[:, None])
    dq_ptrs_trans = DQ + (offs_m[None, :] * stride_dqm + offs_qk_d[:, None])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_qk_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_v_d[None, :])
    mask_n = offs_n < seq_len

    ts_0_ptrs = None
    ts_1_ptrs = None
    ts_1 = None
    off_bias_trans = None
    bias_ptrs_trans = None
    dbias_ptrs_trans = None
    if ATTN_BIAS_TYPE == "fused" and USE_TIME_BIAS:
        ts_0_ptrs = TS + offs_m
        ts_1_ptrs = TS + offs_n
        if CAUSAL:
            ts_1 = tl.load(ts_1_ptrs, mask=mask_n)
        else:
            ts_1 = tl.load(ts_1_ptrs + 1, mask=mask_n)
    elif ATTN_BIAS_TYPE == "separate":
        off_bias_trans = offs_m[None, :] * seq_len + offs_n[:, None]
        bias_ptrs_trans = Bias + off_bias_trans
        dbias_ptrs_trans = DBias + off_bias_trans
    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    if HAS_MULTIPLE_TARGETS:
        if INVALID_MASK_TYPE == "lower_triangular":
            pos_offs_n = tl.where(
                offs_n < seq_len - n_targets,
                offs_n,
                seq_len - n_targets,
            )
        elif INVALID_MASK_TYPE == "upper_triangular":
            pos_offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
    else:
        pos_offs_n = offs_n
    # loop over rows
    if CONTEXTUAL_SEQ_LEN > 0 and INVALID_MASK_TYPE == "lower_triangular":
        for start_m in range(0, CONTEXTUAL_SEQ_LEN, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _ragged_hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                mask_n=mask_n,
                ts_0_ptrs=ts_0_ptrs,
                ts_1=ts_1,
                bias_ptrs_trans=bias_ptrs_trans,
                dbias_ptrs_trans=dbias_ptrs_trans,
                do_ptrs=do_ptrs,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                TW=TW,
                PW=PW,
                DTW=DTW,
                DPW=DPW,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ATOMIC_ADD=ATOMIC_ADD,
            )
    # pyre-ignore[61]
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _ragged_hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            mask_n=mask_n,
            ts_0_ptrs=ts_0_ptrs,
            ts_1=ts_1,
            bias_ptrs_trans=bias_ptrs_trans,
            dbias_ptrs_trans=dbias_ptrs_trans,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            # pyre-fixme[61]: `pos_offs_n` is undefined, or not always defined.
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            n_targets=n_targets,
            TW=TW,
            PW=PW,
            DTW=DTW,
            DPW=DPW,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
            CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS,
            FUSED_BIAS_BWD=FUSED_BIAS_BWD,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            ATOMIC_ADD=ATOMIC_ADD,
        )
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_v_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_qk_d[None, :])
    dk = dk * alpha
    tl.store(dv_ptrs, dv.to(k.dtype), mask=mask_n[:, None])
    tl.store(dk_ptrs, dk.to(k.dtype), mask=mask_n[:, None])


def _bwd_pre_hook(nargs):
    nargs["DQ"].zero_()
    if nargs["DTW"] is not None:
        nargs["DTW"].zero_()
    if nargs["DPW"] is not None:
        nargs["DPW"].zero_()
    if nargs["SEQUENCE_PARALLEL"] is True:
        nargs["LOCK"].zero_()


def _get_bw_configs() -> List[triton.Config]:
    if torch.version.hip:
        configs = []
        for BLOCK_M in [32, 64]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            for waves_per_eu in [0, 2, 4]:
                                for sp in [True, False]:
                                    configs.append(
                                        triton.Config(
                                            {
                                                "BLOCK_M": BLOCK_M,
                                                "BLOCK_N": BLOCK_N,
                                                "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                                "waves_per_eu": waves_per_eu,
                                                "SEQUENCE_PARALLEL": sp,
                                                "UNROLL": 1,
                                            },
                                            num_stages=num_stages,
                                            num_warps=num_warps,
                                            pre_hook=_bwd_pre_hook,
                                        )
                                    )
        return configs

    configs = [
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 16, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 2},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False, "UNROLL": 4},
            num_stages=2,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 16, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=2,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 32, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=1,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=2,
            num_warps=4,
            pre_hook=_bwd_pre_hook,
        ),
        triton.Config(
            {"BLOCK_M": 32, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True, "UNROLL": 1},
            num_stages=3,
            num_warps=8,
            pre_hook=_bwd_pre_hook,
        ),
    ]
    return configs


@triton_autotune(
    configs=_get_bw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "BUCKET_FN",
        "ATTN_BIAS_TYPE",
    ],
)
@triton.jit
def _ragged_hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    TS,
    TW,
    PW,
    Bias,
    seq2_offsets,
    num_targets,
    DOut,
    DQ,
    DK,
    DV,
    DBias,
    DTW,
    DPW,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    ATTN_BIAS_TYPE: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    FUSED_BIAS_BWD: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(0)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    off_h = off_h.to(tl.int64)
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if HAS_MULTIPLE_TARGETS:
        n_targets = tl.load(num_targets + off_z).to(tl.int32)
    else:
        n_targets = None
    # offset pointers for batch/head
    Q = Q + seq_start * stride_qm + off_h * stride_qh
    K = K + seq_start * stride_kn + off_h * stride_kh
    V = V + seq_start * stride_vn + off_h * stride_vh
    DOut = DOut + seq_start * stride_dom + off_h * stride_doh
    DQ = DQ + seq_start * stride_dqm + off_h * stride_dqh
    DK = DK + seq_start * stride_dkn + off_h * stride_dkh
    DV = DV + seq_start * stride_dvn + off_h * stride_dvh
    if ATTN_BIAS_TYPE == "fused":
        if USE_TIME_BIAS:
            TS = TS + off_z * stride_ts
        if FUSED_BIAS_BWD:
            if USE_TIME_BIAS:
                DTW = DTW + off_hz * (num_buckets + 1)
            if USE_POS_BIAS:
                if HAS_MAX_POS_IND:
                    DPW = DPW + off_hz * (2 * max_pos_ind - 1)
                else:
                    DPW = DPW + off_hz * (2 * MAX_SEQ_LEN - 1)
    elif ATTN_BIAS_TYPE == "separate":
        seq2_start = tl.load(seq2_offsets + off_z)
        bias_start = seq2_start * H + off_h * seq_len * seq_len
        Bias = Bias + bias_start
        DBias = DBias + bias_start

    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _ragged_hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            Q=Q,
            K=K,
            V=V,
            TS=TS,
            TW=TW,
            PW=PW,
            Bias=Bias,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
            DBias=DBias,
            DTW=DTW,
            DPW=DPW,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_kn=stride_kn,
            stride_vn=stride_vn,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            stride_dkn=stride_dkn,
            stride_dvn=stride_dvn,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            INVALID_MASK_TYPE=INVALID_MASK_TYPE,
            CAUSAL=CAUSAL,
            BUCKET_FN=BUCKET_FN,
            ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
            USE_TIME_BIAS=USE_TIME_BIAS,
            USE_POS_BIAS=USE_POS_BIAS,
            FUSED_BIAS_BWD=FUSED_BIAS_BWD,
            HAS_MAX_POS_IND=HAS_MAX_POS_IND,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            UNROLL=UNROLL,
            ATOMIC_ADD=True,
        )
    else:
        for start_n in range(0, seq_len, BLOCK_N):
            _ragged_hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                Q=Q,
                K=K,
                V=V,
                TS=TS,
                TW=TW,
                PW=PW,
                Bias=Bias,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
                DBias=DBias,
                DTW=DTW,
                DPW=DPW,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_kn=stride_kn,
                stride_vn=stride_vn,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                stride_dkn=stride_dkn,
                stride_dvn=stride_dvn,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                num_buckets=num_buckets,
                max_pos_ind=max_pos_ind,
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                time_bucket_incr=time_bucket_incr,
                time_bucket_div=time_bucket_div,
                time_delta=time_delta,
                INVALID_MASK_TYPE=INVALID_MASK_TYPE,
                CAUSAL=CAUSAL,
                BUCKET_FN=BUCKET_FN,
                ATTN_BIAS_TYPE=ATTN_BIAS_TYPE,
                USE_TIME_BIAS=USE_TIME_BIAS,
                USE_POS_BIAS=USE_POS_BIAS,
                FUSED_BIAS_BWD=FUSED_BIAS_BWD,
                HAS_MAX_POS_IND=HAS_MAX_POS_IND,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                CONTEXTUAL_SEQ_LEN=CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
            )


def triton_ragged_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor],
    attn_bias: Optional[torch.Tensor],
    seq2_offsets: Optional[torch.Tensor],
    max_attn_len: Optional[int],
    contextual_seq_len: Optional[int],
    sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    assert invalid_attn_mask_type in [
        "lower_triangular",
        "upper_triangular",
    ]
    if invalid_attn_mask_type != "lower_triangular":
        assert contextual_seq_len is None or contextual_seq_len == 0
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape

    out = torch.empty_like(v)
    max_attn_len = max_attn_len or 0
    contextual_seq_len = contextual_seq_len or 0
    has_multiple_targets = num_targets is not None
    has_attn_bias = attn_bias is not None
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _ragged_hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        TS=None,
        TW=None,
        PW=None,
        Bias=attn_bias,
        seq2_offsets=seq2_offsets,
        delta_x_offsets=None,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=None,
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        num_buckets=None,
        max_pos_ind=None,
        time_bucket_incr=None,
        time_bucket_div=None,
        time_delta=None,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=None,
        BUCKET_FN="none",
        ATTN_BIAS_TYPE="separate" if has_attn_bias else "none",
        USE_TIME_BIAS=False,
        USE_POS_BIAS=False,
        HAS_MAX_POS_IND=False,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        MAX_ATTN_LEN=max_attn_len,
        CONTEXTUAL_SEQ_LEN=contextual_seq_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
    )
    return out


def triton_ragged_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    attn_bias: Optional[torch.Tensor],
    seq2_offsets: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    invalid_attn_mask_type: float,
    contextual_seq_len: Optional[int],
    sort_by_length_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    dout = _switch_to_contiguous_if_needed(dout)
    dq = _switch_to_contiguous_if_needed(dq)
    dk = _switch_to_contiguous_if_needed(dk)
    dv = _switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        if attn_bias is not None:
            dbias = torch.zeros_like(attn_bias)
            assert dbias.is_contiguous()
        else:
            dbias = None
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v), dbias
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
    if attn_bias is not None:
        dbias = torch.zeros_like(attn_bias)
        assert dbias.is_contiguous()
    else:
        dbias = None
    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    AUTOTUNE_Z = prev_power_of_2(Z)
    _ragged_hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        TS=None,
        TW=None,
        PW=None,
        Bias=attn_bias,
        seq2_offsets=seq2_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        DBias=dbias,
        DTW=None,
        DPW=None,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=None,
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        CONTEXTUAL_SEQ_LEN=0 if contextual_seq_len is None else contextual_seq_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        num_buckets=None,
        max_pos_ind=None,
        MAX_ATTN_LEN=max_attn_len,
        time_bucket_incr=None,
        time_bucket_div=None,
        time_delta=None,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=None,
        BUCKET_FN="none",
        ATTN_BIAS_TYPE="separate" if attn_bias is not None else "none",
        USE_TIME_BIAS=False,
        USE_POS_BIAS=False,
        FUSED_BIAS_BWD=None,
        HAS_MAX_POS_IND=False,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
    )

    return dq, dk, dv, dbias


class RaggedAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        invalid_attn_mask_type: str,
        num_targets: Optional[torch.Tensor],
        attn_bias: Optional[torch.Tensor],
        seq2_offsets: Optional[torch.Tensor],
        max_attn_len: Optional[int],
        contextual_seq_len: Optional[int],
        sort_by_length: bool,
    ) -> torch.Tensor:
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors = [q, k, v, seq_offsets]
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if attn_bias is not None:
            assert seq2_offsets is not None
            saved_tensors.extend([attn_bias, seq2_offsets])
        contextual_seq_len = contextual_seq_len or 0
        max_attn_len = max_attn_len or 0
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.has_attn_bias = attn_bias is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_ragged_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            num_targets=num_targets,
            attn_bias=attn_bias,
            seq2_offsets=seq2_offsets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            q, k, v, seq_offsets = ctx.saved_tensors[:4]
            idx = 4
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.has_attn_bias:
                attn_bias, seq2_offsets = ctx.saved_tensors[idx : idx + 2]
                idx += 2
            else:
                attn_bias = None
                seq2_offsets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv, dbias = triton_ragged_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                attn_bias=attn_bias,
                seq2_offsets=seq2_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                invalid_attn_mask_type=ctx.invalid_attn_mask_type,
                contextual_seq_len=ctx.contextual_seq_len,
                sort_by_length_indices=sort_by_length_indices,
            )
            return (
                None,
                None,
                dq,
                dk,
                dv,
                None,
                None,
                None,
                dbias,
                None,
                None,
                None,
                None,
            )


@triton.jit
def _attn_bias_bwd(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    TS,
    TW,
    PW,
    num_targets,
    DOut,
    DTW,
    DPW,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_ts,
    stride_dom,
    stride_doh,
    alpha,
    Z,
    H,
    MAX_SEQ_LEN,
    DimQ,
    DimV,
    num_buckets,
    max_pos_ind,
    time_bucket_incr,
    time_bucket_div,
    time_delta,
    MAX_ATTN_LEN: tl.constexpr,
    INVALID_MASK_TYPE: tl.constexpr,
    CAUSAL: tl.constexpr,
    BUCKET_FN: tl.constexpr,
    USE_TIME_BIAS: tl.constexpr,
    USE_POS_BIAS: tl.constexpr,
    HAS_MAX_POS_IND: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_N_BLOCKS: tl.constexpr,
    NUM_OUT_GROUPS: tl.constexpr,
):
    off_mn = tl.program_id(0)
    off_m = off_mn // NUM_N_BLOCKS
    off_n = off_mn % NUM_N_BLOCKS
    widx = off_m * (off_m + 1) // 2 + off_n
    widx = widx % NUM_OUT_GROUPS
    start_m = off_m * BLOCK_N
    start_n = off_n * BLOCK_N
    offs_m = start_m + tl.arange(0, BLOCK_N)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_qk_d = tl.arange(0, BLOCK_D_Q)
    offs_v_d = tl.arange(0, BLOCK_D_V)
    dbias_pos = None
    offs_pos_w = None
    if USE_POS_BIAS:
        if not HAS_MULTIPLE_TARGETS:
            dbias_pos = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
            if HAS_MAX_POS_IND:
                offs_pos_w = offs_n[None, :] - offs_m[:, None] + max_pos_ind - 1
                offs_pos_w = tl.where(offs_pos_w > 0, offs_pos_w, 0)
                offs_pos_w = tl.where(
                    offs_pos_w < 2 * max_pos_ind - 2, offs_pos_w, 2 * max_pos_ind - 2
                )
            else:
                offs_pos_w = offs_n[None, :] - offs_m[:, None] + MAX_SEQ_LEN - 1
    if HAS_MULTIPLE_TARGETS:
        invalid_mask = offs_m[:, None] == offs_n[None, :]
    else:
        # mt_invalid_mask will be equal to invalid_mask
        if MAX_ATTN_LEN > 0:
            if INVALID_MASK_TYPE == "lower_triangular":
                invalid_mask = (offs_m[:, None] >= offs_n[None, :]) and (
                    offs_m[:, None] - offs_n[None, :] <= MAX_ATTN_LEN
                )
            elif INVALID_MASK_TYPE == "upper_triangular":
                invalid_mask = (offs_m[:, None] <= offs_n[None, :]) and (
                    offs_n[None, :] - offs_m[:, None] <= MAX_ATTN_LEN
                )
        else:
            if INVALID_MASK_TYPE == "lower_triangular":
                invalid_mask = offs_m[:, None] >= offs_n[None, :]
            elif INVALID_MASK_TYPE == "upper_triangular":
                invalid_mask = offs_m[:, None] <= offs_n[None, :]
    for off_z in range(Z):
        seq_start = tl.load(seq_offsets + off_z)
        seq_end = tl.load(seq_offsets + off_z + 1)
        seq_len = (seq_end - seq_start).to(tl.int32)
        if INVALID_MASK_TYPE == "lower_triangular":
            if HAS_MULTIPLE_TARGETS:
                low = start_n
                if MAX_ATTN_LEN > 0:
                    n_targets = tl.load(num_targets + off_z).to(tl.int32)
                    high = start_n + MAX_ATTN_LEN + BLOCK_N
                    high = high if high + n_targets < seq_len else seq_len
                else:
                    high = seq_len
            else:
                low = start_n
                if MAX_ATTN_LEN > 0:
                    high = start_n + MAX_ATTN_LEN + BLOCK_N
                    high = high if high < seq_len else seq_len
                else:
                    high = seq_len
        elif INVALID_MASK_TYPE == "upper_triangular":
            low = 0
            high = start_n + BLOCK_N
        # pyre-ignore[61]
        if start_n < seq_len and (start_m >= low and start_m < high):
            q_ptrs = (
                Q
                + seq_start * stride_qm
                + offs_m[:, None] * stride_qm
                + offs_qk_d[None, :]
            )
            k_ptrs = (
                K
                + seq_start * stride_kn
                + offs_n[:, None] * stride_kn
                + offs_qk_d[None, :]
            )
            v_ptrs = (
                V
                + seq_start * stride_vn
                + offs_n[:, None] * stride_vn
                + offs_v_d[None, :]
            )
            do_ptrs = (
                DOut
                + seq_start * stride_dom
                + offs_m[:, None] * stride_dom
                + offs_v_d[None, :]
            )
            mask_m = offs_m < seq_len
            mask_n = offs_n < seq_len

            if HAS_MULTIPLE_TARGETS:
                if (INVALID_MASK_TYPE != "lower_triangular") or MAX_ATTN_LEN == 0:
                    n_targets = tl.load(num_targets + off_z).to(tl.int32)

                if INVALID_MASK_TYPE == "lower_triangular":
                    pos_offs_m = tl.where(
                        # pyre-ignore[61]
                        offs_m < seq_len - n_targets,
                        offs_m,
                        # pyre-ignore[61]
                        seq_len - n_targets,
                    )
                    pos_offs_n = tl.where(
                        # pyre-ignore[61]
                        offs_n < seq_len - n_targets,
                        offs_n,
                        # pyre-ignore[61]
                        seq_len - n_targets,
                    )
                elif INVALID_MASK_TYPE == "upper_triangular":
                    # pyre-fixme[61]: `n_targets` is undefined, or not always defined.
                    pos_offs_m = tl.where(offs_m > n_targets - 1, offs_m, n_targets - 1)
                    # pyre-fixme[61]: `n_targets` is undefined, or not always defined.
                    pos_offs_n = tl.where(offs_n > n_targets - 1, offs_n, n_targets - 1)
            else:
                pos_offs_n = offs_n
                pos_offs_m = offs_m
            mt_offs_pos_w = None
            if USE_POS_BIAS:
                if HAS_MULTIPLE_TARGETS:
                    if HAS_MAX_POS_IND:
                        mt_offs_pos_w = (
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            pos_offs_n[None, :]
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            - pos_offs_m[:, None]
                            + max_pos_ind
                            - 1
                        )
                        mt_offs_pos_w = tl.where(mt_offs_pos_w > 0, mt_offs_pos_w, 0)
                        mt_offs_pos_w = tl.where(
                            mt_offs_pos_w < 2 * max_pos_ind - 2,
                            mt_offs_pos_w,
                            2 * max_pos_ind - 2,
                        )
                    else:
                        mt_offs_pos_w = (
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            pos_offs_n[None, :]
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            - pos_offs_m[:, None]
                            + MAX_SEQ_LEN
                            - 1
                        )
                else:
                    mt_offs_pos_w = offs_pos_w
            if HAS_MULTIPLE_TARGETS:
                if MAX_ATTN_LEN > 0:
                    if INVALID_MASK_TYPE == "lower_triangular":
                        # pyre-ignore[61]
                        mt_invalid_mask = invalid_mask or (
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            pos_offs_m[:, None] > pos_offs_n[None, :]
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            and pos_offs_n[None, :] - pos_offs_m[:, None]
                            >= -MAX_ATTN_LEN
                        )
                    elif INVALID_MASK_TYPE == "upper_triangular":
                        # pyre-ignore[61]
                        mt_invalid_mask = invalid_mask or (
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            pos_offs_m[:, None] < pos_offs_n[None, :]
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            and pos_offs_n[None, :] - pos_offs_m[:, None]
                            <= MAX_ATTN_LEN
                        )
                else:
                    if INVALID_MASK_TYPE == "lower_triangular":
                        mt_invalid_mask = (
                            # pyre-ignore[61]
                            invalid_mask
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            or pos_offs_m[:, None] > pos_offs_n[None, :]
                        )
                    elif INVALID_MASK_TYPE == "upper_triangular":
                        mt_invalid_mask = (
                            # pyre-ignore[61]
                            invalid_mask
                            # pyre-fixme[61]: `pos_offs_m` is undefined, or not
                            #  always defined.
                            # pyre-fixme[61]: `pos_offs_n` is undefined, or not
                            #  always defined.
                            or pos_offs_m[:, None] < pos_offs_n[None, :]
                        )
            else:
                # pyre-ignore[61]
                mt_invalid_mask = invalid_mask
            if CONTEXTUAL_SEQ_LEN > 0:
                if INVALID_MASK_TYPE == "lower_triangular":
                    row_filter = offs_m < CONTEXTUAL_SEQ_LEN
                    if HAS_MULTIPLE_TARGETS:
                        # pyre-ignore[61]
                        col_filter = offs_n < seq_len - n_targets
                    else:
                        col_filter = offs_n < seq_len
                    invalid_mask = invalid_mask or (
                        row_filter[:, None] and col_filter[None, :]
                    )
            ts = None
            if USE_TIME_BIAS:
                ts_ptrs = TS + off_z * stride_ts
                ts_0_ptrs = ts_ptrs + offs_m
                ts_1_ptrs = ts_ptrs + offs_n
                if CAUSAL:
                    ts_0 = tl.load(ts_0_ptrs + 1, mask=mask_m)
                    ts_1 = tl.load(ts_1_ptrs, mask=mask_n)
                else:
                    ts_0 = tl.load(ts_0_ptrs, mask=mask_m)
                    ts_1 = tl.load(ts_1_ptrs + 1, mask=mask_n)
                ts = ts_0[:, None] - ts_1[None, :]
                ts = ts + time_delta
                ts = tl.where(ts > 1e-6, ts, 1e-6)
                ts = ts * (1.0 / time_bucket_incr)
                if BUCKET_FN == "log":
                    ts = tl.log(ts)
                elif BUCKET_FN == "sqrt":
                    ts = tl.sqrt(ts)
                ts = ts * (1.0 / time_bucket_div)
                ts = ts.to(tl.int32)
                ts = tl.where(ts > 0, ts, 0)
                ts = tl.where(ts < num_buckets, ts, num_buckets)

            attn_bias = tl.zeros([BLOCK_N, BLOCK_N], dtype=tl.float32)
            if USE_TIME_BIAS:
                ts_w = tl.load(
                    TW + ts,
                    mask=mask_m[:, None] & mask_n[None, :] & mt_invalid_mask,  # pyre-ignore[61]
                )
                attn_bias = attn_bias + ts_w
            if USE_POS_BIAS:
                pos_w = tl.load(
                    PW + mt_offs_pos_w,
                    mask=mask_m[:, None] & mask_n[None, :] & mt_invalid_mask,  # pyre-ignore[61]
                )
                attn_bias = attn_bias + pos_w

            dbias = tl.zeros((BLOCK_N, BLOCK_N), dtype=tl.float32)
            for off_h in range(H):
                q = tl.load(q_ptrs + off_h * stride_qh, mask=mask_m[:, None], other=0.0)
                k = tl.load(k_ptrs + off_h * stride_kh, mask=mask_n[:, None], other=0.0)
                qk = tl.dot(q, tl.trans(k), allow_tf32=ALLOW_TF32) * alpha
                qk = qk + attn_bias
                # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
                sig = fast_dividef(1.0, 1.0 + tl.exp(-qk))
                do = tl.load(
                    do_ptrs + off_h * stride_doh,
                    mask=mask_m[:, None],
                    other=0.0,
                )
                v = tl.load(v_ptrs + off_h * stride_vh, mask=mask_n[:, None], other=0.0)
                dqk = tl.dot(do, tl.trans(v), allow_tf32=ALLOW_TF32)
                dqk = dqk * sig * (1 + qk * (1 - sig)) * (1.0 / MAX_SEQ_LEN)
                dbias = dbias + dqk

            if USE_TIME_BIAS:
                dtw_ptrs = DTW + widx * (num_buckets + 1)
                tl.atomic_add(
                    dtw_ptrs + ts,
                    dbias,
                    mask=mask_m[:, None] & mask_n[None, :] & mt_invalid_mask,  # pyre-ignore[61]
                    sem="relaxed",
                )
            if USE_POS_BIAS:
                if HAS_MULTIPLE_TARGETS:
                    if HAS_MAX_POS_IND:
                        dpw_ptrs = DPW + widx * (2 * max_pos_ind - 1)
                    else:
                        dpw_ptrs = DPW + widx * (2 * MAX_SEQ_LEN - 1)
                    tl.atomic_add(
                        dpw_ptrs + mt_offs_pos_w,
                        dbias,
                        mask=mask_m[:, None] & mask_n[None, :] & mt_invalid_mask,  # pyre-ignore[61]
                        sem="relaxed",
                    )
                else:
                    dbias_pos += dbias

    if USE_POS_BIAS and not HAS_MULTIPLE_TARGETS:
        if HAS_MAX_POS_IND:
            dpw_ptrs = DPW + widx * (2 * max_pos_ind - 1)
        else:
            dpw_ptrs = DPW + widx * (2 * MAX_SEQ_LEN - 1)
        tl.atomic_add(
            dpw_ptrs + offs_pos_w,
            dbias_pos,
            # pyre-ignore[61]
            mask=invalid_mask,
            sem="relaxed",
        )


def triton_ragged_attention_relative_bias_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    timestamps: torch.Tensor,
    ts_weights: torch.Tensor,
    pos_weights: torch.Tensor,
    causal: bool,
    num_buckets: int,
    time_bucket_fn: str,
    time_bucket_incr: float,
    time_bucket_div: float,
    time_delta: float,
    max_pos_ind: Optional[int],
    num_targets: Optional[torch.Tensor],
    relative_bias_type: str,
    max_attn_len: Optional[int],
    use_time_bias: bool,
    use_pos_bias: bool,
    contextual_seq_len: Optional[int],
    sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    Z = timestamps.size(0)
    AUTOTUNE_Z = prev_power_of_2(Z)
    N = timestamps.size(1) - 1
    has_multiple_targets = num_targets is not None
    has_max_pos_id = max_pos_ind is not None
    has_sort_by_length_indices = sort_by_length_indices is not None
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    if L == 0:
        return out
    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    contextual_seq_len = 0 if contextual_seq_len is None else contextual_seq_len

    _ragged_hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        TS=timestamps,
        TW=ts_weights,
        PW=pos_weights,
        Bias=None,
        seq2_offsets=None,
        delta_x_offsets=None,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=timestamps.stride(0),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=0,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=causal,
        BUCKET_FN=time_bucket_fn,
        ATTN_BIAS_TYPE="fused",
        USE_TIME_BIAS=use_time_bias,
        USE_POS_BIAS=use_pos_bias,
        HAS_MAX_POS_IND=has_max_pos_id,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        MAX_ATTN_LEN=max_attn_len or 0,
        CONTEXTUAL_SEQ_LEN=contextual_seq_len or 0,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
    )
    return out


def triton_ragged_attention_relative_bias_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    ts_weights: torch.Tensor,
    pos_weights: torch.Tensor,
    timestamps: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    use_time_bias: bool,
    use_pos_bias: bool,
    N: int,
    num_buckets: int,
    max_pos_ind: int,
    max_attn_len: int,
    alpha: float,
    time_bucket_incr: float,
    time_bucket_div: float,
    time_delta: float,
    invalid_attn_mask_type: str,
    causal: bool,
    time_bucket_fn: str,
    has_multiple_targets: bool,
    contextual_seq_len: Optional[int],
    sort_by_length_indices: Optional[torch.Tensor],
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    dout = _switch_to_contiguous_if_needed(dout)
    dq = _switch_to_contiguous_if_needed(dq)
    dk = _switch_to_contiguous_if_needed(dk)
    dv = _switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return (
            dq,
            dk,
            dv,
            torch.zeros_like(ts_weights) if use_time_bias else None,
            torch.zeros_like(pos_weights) if use_pos_bias else None,
        )

    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    _, H, DimQ = q.shape
    _, _, DimV = v.shape

    contextual_seq_len = 0 if contextual_seq_len is None else contextual_seq_len

    if autotune_max_seq_len(N) < 1024:
        fused_bias_bwd = True
        # have to explicitly use fp32 since 'atomic_add does not support bf16'
        d_ts_weights = torch.empty(
            (Z * H, num_buckets + 1), dtype=torch.float32, device=q.device
        )
        assert d_ts_weights.is_contiguous()
        pos_weights_size = 2 * N - 1 if max_pos_ind is None else 2 * max_pos_ind - 1
        d_pos_weights = torch.empty(
            (Z * H, pos_weights_size), dtype=torch.float32, device=q.device
        )
        assert d_pos_weights.is_contiguous()
    else:
        fused_bias_bwd = False
        BLOCK_N = 32
        NUM_N_BLOCKS = triton.cdiv(N, BLOCK_N)
        SEQLEN_THRESHOLD = 2048
        NUM_N_GROUPS = triton.cdiv(min(N, SEQLEN_THRESHOLD), BLOCK_N)
        NUM_OUT_GROUPS = (NUM_N_GROUPS + 1) * NUM_N_GROUPS // 2
        # have to explicitly use fp32 since 'atomic_add does not support bf16'
        d_ts_weights = torch.zeros(
            (NUM_OUT_GROUPS, num_buckets + 1),
            dtype=torch.float32,
            device=q.device,
        )
        assert d_ts_weights.is_contiguous()
        pos_weights_size = 2 * N - 1 if max_pos_ind is None else 2 * max_pos_ind - 1
        d_pos_weights = torch.zeros(
            (NUM_OUT_GROUPS, pos_weights_size),
            dtype=torch.float32,
            device=q.device,
        )
        assert d_pos_weights.is_contiguous()
        # pyre-ignore [28]
        _attn_bias_bwd[(NUM_N_BLOCKS * NUM_N_BLOCKS,)](
            Q=q,
            K=k,
            V=v,
            seq_offsets=seq_offsets,
            TS=timestamps,
            TW=ts_weights,
            PW=pos_weights,
            num_targets=num_targets,
            DOut=dout,
            DTW=d_ts_weights,
            DPW=d_pos_weights,
            stride_qm=q.stride(0),
            stride_qh=q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_ts=timestamps.stride(0),
            stride_dom=dout.stride(0),
            stride_doh=dout.stride(1),
            alpha=alpha,
            Z=Z,
            H=H,
            MAX_SEQ_LEN=N,
            DimQ=DimQ,
            DimV=DimV,
            num_buckets=num_buckets,
            max_pos_ind=max_pos_ind,
            MAX_ATTN_LEN=max_attn_len,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            INVALID_MASK_TYPE=invalid_attn_mask_type,
            CAUSAL=causal,
            BUCKET_FN=time_bucket_fn,
            USE_TIME_BIAS=use_time_bias,
            USE_POS_BIAS=use_pos_bias,
            HAS_MAX_POS_IND=max_pos_ind is not None,
            HAS_MULTIPLE_TARGETS=has_multiple_targets,
            CONTEXTUAL_SEQ_LEN=contextual_seq_len,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            BLOCK_N=BLOCK_N,
            NUM_N_BLOCKS=NUM_N_BLOCKS,
            NUM_OUT_GROUPS=NUM_OUT_GROUPS,
            num_stages=2,
            num_warps=2,
        )
    grid = lambda meta: (  # noqa E731
        Z * H,
        (triton.cdiv(N, meta["BLOCK_N"]) if meta["SEQUENCE_PARALLEL"] else 1),
    )
    # The minimum size of BLOCK_M used in `_get_bw_configs`.
    # TODO (linjianma): avoid hardcoding the value.
    MIN_BLOCK_M = 16
    lock = torch.empty(
        (Z * H, triton.cdiv(N, MIN_BLOCK_M)),
        dtype=torch.int32,
        device=q.device,
    )
    _ragged_hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        TS=timestamps,
        TW=ts_weights,
        PW=pos_weights,
        Bias=None,
        seq2_offsets=None,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        DBias=None,
        DTW=d_ts_weights if fused_bias_bwd else None,
        DPW=d_pos_weights if fused_bias_bwd else None,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_ts=timestamps.stride(0),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        num_buckets=num_buckets,
        max_pos_ind=max_pos_ind,
        MAX_ATTN_LEN=max_attn_len,
        time_bucket_incr=time_bucket_incr,
        time_bucket_div=time_bucket_div,
        time_delta=time_delta,
        INVALID_MASK_TYPE=invalid_attn_mask_type,
        CAUSAL=causal,
        BUCKET_FN=time_bucket_fn,
        ATTN_BIAS_TYPE="fused",
        USE_TIME_BIAS=use_time_bias,
        USE_POS_BIAS=use_pos_bias,
        FUSED_BIAS_BWD=fused_bias_bwd,
        HAS_MAX_POS_IND=max_pos_ind is not None,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        CONTEXTUAL_SEQ_LEN=contextual_seq_len,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
    )
    return (
        dq,
        dk,
        dv,
        d_ts_weights.sum(dim=0) if use_time_bias else None,
        d_pos_weights.sum(dim=0) if use_pos_bias else None,
    )


class RaggedAttentionRelativeBiasFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        N: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: torch.Tensor,
        invalid_attn_mask_type: str,
        timestamps: torch.Tensor,
        ts_weights: torch.Tensor,
        pos_weights: torch.Tensor,
        causal: bool,
        num_buckets: int,
        time_bucket_fn: str,
        time_bucket_incr: float,
        time_bucket_div: float,
        time_delta: float,
        max_pos_ind: Optional[int],
        num_targets: Optional[torch.Tensor],
        relative_bias_type: str,
        max_attn_len: Optional[int],
        contextual_seq_len: Optional[int],
        sort_by_length: bool,
    ) -> torch.Tensor:
        use_time_bias = relative_bias_type == "TIME" or relative_bias_type == "ALL"
        use_pos_bias = relative_bias_type == "POSITION" or relative_bias_type == "ALL"
        sort_by_length_indices = None
        if sort_by_length:
            seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
            _, sort_by_length_indices = torch.sort(
                seq_lengths, descending=True, stable=False
            )
        saved_tensors: List[torch.Tensor] = [
            timestamps,
            ts_weights,
            pos_weights,
            q,
            k,
            v,
            seq_offsets,
        ]
        contextual_seq_len = 0 if contextual_seq_len is None else contextual_seq_len
        max_attn_len = max_attn_len or 0
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.invalid_attn_mask_type = invalid_attn_mask_type
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_pos_ind = max_pos_ind
        ctx.N = N
        ctx.num_buckets = num_buckets
        ctx.time_bucket_fn = time_bucket_fn
        ctx.time_bucket_incr = time_bucket_incr
        ctx.time_bucket_div = time_bucket_div
        ctx.causal = causal
        ctx.time_delta = time_delta
        ctx.use_time_bias = use_time_bias
        ctx.use_pos_bias = use_pos_bias
        ctx.max_attn_len = max_attn_len
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_ragged_attention_relative_bias_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            timestamps=timestamps,
            ts_weights=ts_weights,
            pos_weights=pos_weights,
            causal=causal,
            num_buckets=num_buckets,
            time_bucket_fn=time_bucket_fn,
            time_bucket_incr=time_bucket_incr,
            time_bucket_div=time_bucket_div,
            time_delta=time_delta,
            max_pos_ind=max_pos_ind,
            num_targets=num_targets,
            relative_bias_type=relative_bias_type,
            max_attn_len=max_attn_len,
            use_time_bias=use_time_bias,
            use_pos_bias=use_pos_bias,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(ctx, dout: torch.Tensor) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ]:
        with torch.inference_mode():
            (
                timestamps,
                ts_weights,
                pos_weights,
                q,
                k,
                v,
                seq_offsets,
            ) = ctx.saved_tensors[:7]
            idx = 7
            if ctx.has_multiple_targets:
                num_targets = ctx.saved_tensors[idx]
                idx += 1
            else:
                num_targets = None
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv, d_ts_weights, d_pos_weights = (
                triton_ragged_attention_relative_bias_bwd(
                    dout=dout,
                    q=q,
                    k=k,
                    v=v,
                    dq=dq,
                    dk=dk,
                    dv=dv,
                    ts_weights=ts_weights,
                    pos_weights=pos_weights,
                    timestamps=timestamps,
                    seq_offsets=seq_offsets,
                    num_targets=num_targets,
                    use_time_bias=ctx.use_time_bias,
                    use_pos_bias=ctx.use_pos_bias,
                    N=ctx.N,
                    num_buckets=ctx.num_buckets,
                    max_pos_ind=ctx.max_pos_ind,
                    max_attn_len=ctx.max_attn_len,
                    alpha=ctx.alpha,
                    time_bucket_incr=ctx.time_bucket_incr,
                    time_bucket_div=ctx.time_bucket_div,
                    time_delta=ctx.time_delta,
                    invalid_attn_mask_type=ctx.invalid_attn_mask_type,
                    causal=ctx.causal,
                    time_bucket_fn=ctx.time_bucket_fn,
                    has_multiple_targets=ctx.has_multiple_targets,
                    contextual_seq_len=ctx.contextual_seq_len,
                    sort_by_length_indices=sort_by_length_indices,
                )
            )
        return (
            None,
            None,
            dq,
            dk,
            dv,
            None,
            None,
            None,
            d_ts_weights,
            d_pos_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

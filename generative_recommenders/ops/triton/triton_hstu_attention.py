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

from typing import List, Optional, Tuple

import torch

# @manual=//triton:triton
import triton

# @manual=//triton:triton
import triton.language as tl

from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
    triton_autotune,
)

try:
    from triton.language.extra.libdevice import fast_dividef  # @manual=//triton:triton
except ImportError:
    try:
        # @manual=//triton:triton
        from triton.language.extra.cuda.libdevice import fast_dividef
    except ImportError:
        # pyre-ignore[21]
        from triton.language.math import fast_dividef  # @manual=//triton:triton


def _get_fw_configs() -> List[triton.Config]:  # noqa: C901
    configs = []
    if torch.version.hip:
        for BLOCK_M in [32, 64, 128]:
            for BLOCK_N in [32, 64]:
                for num_stages in [1, 2]:
                    for num_warps in [4, 8]:
                        for matrix_instr_nonkdim in [16, 32]:
                            configs.append(
                                triton.Config(
                                    {
                                        "BLOCK_M": BLOCK_M,
                                        "BLOCK_N": BLOCK_N,
                                        "matrix_instr_nonkdim": matrix_instr_nonkdim,
                                        "waves_per_eu": 0,
                                        "kpack": 2,
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
def _hstu_attn_fwd_one_block(  # noqa: C901
    start_n,
    seq_len,
    offs_m,
    offs_n,
    q,
    K_block_ptr,
    V_block_ptr,
    n_targets,
    alpha,
    MAX_SEQ_LEN,
    contextual_seq_len,
    MAX_ATTN_LEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    start_n = tl.multiple_of(start_n, BLOCK_N)
    # -- compute qk ----
    k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
    qk = tl.dot(q, k, allow_tf32=ALLOW_TF32) * alpha
    invalid_mask = offs_m[:, None] == offs_n[None, :]
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        offs_m = offs_m - contextual_seq_len + 1
        offs_m = tl.where(
            offs_m > 0,
            offs_m,
            0,
        )
        offs_n = offs_n - contextual_seq_len + 1
        offs_n = tl.where(
            offs_n > 0,
            offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        offs_m = tl.where(
            offs_m < max_ids,
            offs_m,
            max_ids,
        )
        offs_n = tl.where(
            offs_n < max_ids,
            offs_n,
            max_ids,
        )
    offs_m_minus_n = offs_m[:, None] - offs_n[None, :]
    if not CAUSAL:
        offs_m_minus_n = tl.where(offs_m_minus_n > 0, offs_m_minus_n, -offs_m_minus_n)
    invalid_mask = invalid_mask or (offs_m_minus_n > 0)
    if MAX_ATTN_LEN > 0:
        invalid_mask = invalid_mask and offs_m_minus_n <= MAX_ATTN_LEN
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask = invalid_mask or (
            offs_m[:, None] == 0 and offs_n[None, :] < max_ids
        )
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    silu = fast_dividef(qk, 1.0 + tl.exp(-qk)) * (1.0 / MAX_SEQ_LEN)
    silu = tl.where(invalid_mask, silu, 0)
    v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
    silu = silu.to(v.dtype)
    return tl.dot(silu, v, allow_tf32=ALLOW_TF32)


@triton.jit
def _hstu_attn_fwd_compute(  # noqa C901
    Q,
    K,
    V,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_om,
    stride_oh,
    alpha,
    MAX_SEQ_LEN,
    DeltaSize,
    contextual_seq_len,
    off_z,
    off_h,
    pid,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
):
    seq_start = tl.load(seq_offsets + off_z).to(tl.int64)
    off_h = off_h.to(tl.int64)
    off_z = off_z.to(tl.int64)
    seq_end = tl.load(seq_offsets + off_z + 1)
    seq_len = (seq_end - seq_start).to(tl.int32)
    if IS_DELTA_Q:
        start_m_delta = pid * BLOCK_M
        start_m = (start_m_delta + seq_len - DeltaSize).to(tl.int32)
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

        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")
        acc = tl.zeros([BLOCK_M, BLOCK_D_V], dtype=tl.float32)
        if CAUSAL:
            if HAS_MULTIPLE_TARGETS:
                uih_end = seq_len - n_targets
            else:
                uih_end = seq_len
            if HAS_CONTEXTUAL_SEQ_LEN is True and start_m < contextual_seq_len:
                # uih_end must be larger than start_m
                low = 0
                high = seq_len
            else:
                low = 0
                high = start_m + BLOCK_M
                if MAX_ATTN_LEN > 0:
                    if start_m > uih_end:
                        low = uih_end - MAX_ATTN_LEN
                    else:
                        low = start_m - MAX_ATTN_LEN
                    if HAS_CONTEXTUAL_SEQ_LEN:
                        low = low if low > contextual_seq_len else 0
                    else:
                        low = low if low > 0 else 0
                if HAS_MULTIPLE_TARGETS:
                    uih_end = (uih_end + BLOCK_N - 1) // BLOCK_N * BLOCK_N
                    if uih_end < start_m:
                        high = seq_len - n_targets
        else:
            low = 0
            high = seq_len

        if low > 0:
            K_block_ptr = tl.advance(K_block_ptr, (0, low))
            V_block_ptr = tl.advance(V_block_ptr, (low, 0))
        end_n = low
        for start_n in range(low, high, BLOCK_N):
            acc += _hstu_attn_fwd_one_block(
                start_n=start_n,
                seq_len=seq_len,
                offs_m=offs_m,
                offs_n=offs_n + start_n,
                q=q,
                K_block_ptr=K_block_ptr,
                V_block_ptr=V_block_ptr,
                n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                contextual_seq_len=contextual_seq_len,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_N=BLOCK_N,
            )
            K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            end_n += BLOCK_N

        if HAS_MULTIPLE_TARGETS and CAUSAL:
            # pyre-ignore[61]
            if uih_end < start_m:
                low_delta = start_m
                high_delta = start_m + BLOCK_M
                offset = (low_delta - end_n).to(tl.int32)
                K_block_ptr = tl.advance(K_block_ptr, (0, offset))
                V_block_ptr = tl.advance(V_block_ptr, (offset, 0))
                for start_delta in tl.range(
                    low_delta, high_delta, BLOCK_N, num_stages=0
                ):
                    acc += _hstu_attn_fwd_one_block(
                        start_n=start_delta,
                        seq_len=seq_len,
                        offs_m=offs_m,
                        offs_n=offs_n + start_delta,
                        q=q,
                        K_block_ptr=K_block_ptr,
                        V_block_ptr=V_block_ptr,
                        n_targets=n_targets if HAS_MULTIPLE_TARGETS else None,
                        alpha=alpha,
                        MAX_SEQ_LEN=MAX_SEQ_LEN,
                        MAX_ATTN_LEN=MAX_ATTN_LEN,
                        contextual_seq_len=contextual_seq_len,
                        CAUSAL=CAUSAL,
                        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                        ALLOW_TF32=ALLOW_TF32,
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
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
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
    contextual_seq_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    HAS_SORT_BY_LENGTH_INDICES: tl.constexpr,
):
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    if HAS_SORT_BY_LENGTH_INDICES:
        off_z = tl.load(sort_by_length_indices + off_z)
    off_h = off_hz % H
    pid = tl.program_id(0)
    _hstu_attn_fwd_compute(
        Q=Q,
        K=K,
        V=V,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=Out,
        stride_qm=stride_qm,
        stride_qh=stride_qh,
        stride_kn=stride_kn,
        stride_kh=stride_kh,
        stride_vn=stride_vn,
        stride_vh=stride_vh,
        stride_om=stride_om,
        stride_oh=stride_oh,
        alpha=alpha,
        MAX_SEQ_LEN=MAX_SEQ_LEN,
        DeltaSize=DeltaSize,
        contextual_seq_len=contextual_seq_len,
        off_z=off_z,
        off_h=off_h,
        pid=pid,
        CAUSAL=CAUSAL,
        HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
        IS_DELTA_Q=IS_DELTA_Q,
        ALLOW_TF32=ALLOW_TF32,
        BLOCK_D_Q=BLOCK_D_Q,
        BLOCK_D_V=BLOCK_D_V,
        MAX_ATTN_LEN=MAX_ATTN_LEN,
        HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
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
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)
@triton.jit
def _hstu_attn_fwd_persistent(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    Out,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
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
    contextual_seq_len,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    IS_DELTA_Q: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MAX_ATTN_LEN: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
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
        _hstu_attn_fwd_compute(
            Q=Q,
            K=K,
            V=V,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            Out=Out,
            stride_qm=stride_qm,
            stride_qh=stride_qh,
            stride_kn=stride_kn,
            stride_kh=stride_kh,
            stride_vn=stride_vn,
            stride_vh=stride_vh,
            stride_om=stride_om,
            stride_oh=stride_oh,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            DeltaSize=DeltaSize,
            contextual_seq_len=contextual_seq_len,
            off_z=off_z,
            off_h=off_h,
            pid=pid,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            IS_DELTA_Q=IS_DELTA_Q,
            ALLOW_TF32=ALLOW_TF32,
            BLOCK_D_Q=BLOCK_D_Q,
            BLOCK_D_V=BLOCK_D_V,
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )
        tile_idx += num_progs


_hstu_attn_fwd = triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)(_hstu_attn_fwd.fn)

_hstu_attn_fwd_persistent = triton_autotune(
    configs=_get_fw_configs(),
    key=[
        "AUTOTUNE_Z",
        "H",
        "AUTOTUNE_MAX_SEQ_LEN",
        "DimQ",
        "DimV",
        "DeltaSize",
        "IS_DELTA_Q",
    ],
)(_hstu_attn_fwd_persistent.fn)


@triton.jit
def _hstu_attn_bwd_one_block(  # noqa C901
    start_m,
    offs_n,
    offs_m,
    q_ptrs_trans,
    dq_ptrs_trans,
    mask_n,
    do_ptrs,
    dk,
    dv,
    k,
    v,
    pos_offs_n,
    seq_len,
    n_targets,
    max_ids,
    contextual_seq_len,
    LOCK,
    stride_qm,
    stride_dom,
    stride_dqm,
    alpha,
    MAX_SEQ_LEN,
    MAX_ATTN_LEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    pos_offs_m = offs_m + start_m
    mask_m = pos_offs_m < seq_len
    invalid_mask_trans = pos_offs_m[None, :] == offs_n[:, None]
    # recompute qk and silu
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_m = pos_offs_m - contextual_seq_len + 1
        pos_offs_m = tl.where(
            pos_offs_m > 0,
            pos_offs_m,
            0,
        )
    if HAS_MULTIPLE_TARGETS:
        pos_offs_m = tl.where(
            pos_offs_m < max_ids,
            pos_offs_m,
            max_ids,
        )
    q_trans = tl.load(
        q_ptrs_trans + start_m * stride_qm,
        mask=mask_m[None, :],
        other=0.0,
    )
    qk_trans = tl.dot(k, q_trans, allow_tf32=ALLOW_TF32) * alpha
    # pyre-fixme[16]: Module `math` has no attribute `fast_dividef`.
    sig_trans = fast_dividef(1.0, 1.0 + tl.exp(-qk_trans))
    silu_trans = qk_trans * sig_trans * (1.0 / MAX_SEQ_LEN)
    pos_offs_m_minus_n = pos_offs_m[None, :] - pos_offs_n[:, None]
    if not CAUSAL:
        pos_offs_m_minus_n = tl.where(
            pos_offs_m_minus_n > 0, pos_offs_m_minus_n, -pos_offs_m_minus_n
        )
    invalid_mask_trans = invalid_mask_trans or (pos_offs_m_minus_n > 0)
    if MAX_ATTN_LEN > 0:
        invalid_mask_trans = invalid_mask_trans and pos_offs_m_minus_n <= MAX_ATTN_LEN
    if HAS_CONTEXTUAL_SEQ_LEN:
        invalid_mask_trans = invalid_mask_trans or (
            pos_offs_m[None, :] == 0 and pos_offs_n[:, None] < max_ids
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
def _hstu_attn_bwd_one_col_block(  # noqa C901
    start_n,
    seq_len,
    n_targets,
    contextual_seq_len,
    Q,
    K,
    V,
    DOut,
    DQ,
    DK,
    DV,
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
    MAX_ATTN_LEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_D_Q: tl.constexpr,
    BLOCK_D_V: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    UNROLL: tl.constexpr,
    ATOMIC_ADD: tl.constexpr,
):
    # Work on the subsequence dv[start_n, start_n + BLOCK_N, :]
    if CAUSAL:
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
        if HAS_CONTEXTUAL_SEQ_LEN:
            contextual_block_end = tl.cdiv(contextual_seq_len, BLOCK_M) * BLOCK_M
            if low < contextual_block_end:
                low = contextual_block_end
    else:
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

    do_ptrs = DOut + (offs_m[:, None] * stride_dom + offs_v_d[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_D_V], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_D_Q], dtype=tl.float32)
    # k and v stay in SRAM throughout
    k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)
    v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
    max_ids = seq_len
    if HAS_CONTEXTUAL_SEQ_LEN:
        pos_offs_n = offs_n - contextual_seq_len + 1
        pos_offs_n = tl.where(
            pos_offs_n > 0,
            pos_offs_n,
            0,
        )
        max_ids = max_ids - contextual_seq_len + 1
    else:
        pos_offs_n = offs_n
    if HAS_MULTIPLE_TARGETS:
        max_ids = max_ids - n_targets
        pos_offs_n = tl.where(
            pos_offs_n < max_ids,
            pos_offs_n,
            max_ids,
        )
    # loop over rows
    if HAS_CONTEXTUAL_SEQ_LEN and CAUSAL:
        for start_m in range(0, contextual_seq_len, BLOCK_M):
            start_m = tl.multiple_of(start_m, BLOCK_M)
            dk, dv = _hstu_attn_bwd_one_block(
                start_m=start_m,
                offs_n=offs_n,
                offs_m=offs_m,
                q_ptrs_trans=q_ptrs_trans,
                dq_ptrs_trans=dq_ptrs_trans,
                mask_n=mask_n,
                do_ptrs=do_ptrs,
                dk=dk,
                dv=dv,
                k=k,
                v=v,
                pos_offs_n=pos_offs_n,
                seq_len=seq_len,
                n_targets=n_targets,
                max_ids=max_ids,
                contextual_seq_len=contextual_seq_len,
                LOCK=LOCK,
                stride_qm=stride_qm,
                stride_dom=stride_dom,
                stride_dqm=stride_dqm,
                alpha=alpha,
                MAX_SEQ_LEN=MAX_SEQ_LEN,
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                ATOMIC_ADD=ATOMIC_ADD,
            )
    for start_m in tl.range(low, high, BLOCK_M, loop_unroll_factor=UNROLL):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        dk, dv = _hstu_attn_bwd_one_block(
            start_m=start_m,
            offs_n=offs_n,
            offs_m=offs_m,
            q_ptrs_trans=q_ptrs_trans,
            dq_ptrs_trans=dq_ptrs_trans,
            mask_n=mask_n,
            do_ptrs=do_ptrs,
            dk=dk,
            dv=dv,
            k=k,
            v=v,
            pos_offs_n=pos_offs_n,
            seq_len=seq_len,
            n_targets=n_targets,
            max_ids=max_ids,
            contextual_seq_len=contextual_seq_len,
            LOCK=LOCK,
            stride_qm=stride_qm,
            stride_dom=stride_dom,
            stride_dqm=stride_dqm,
            alpha=alpha,
            MAX_SEQ_LEN=MAX_SEQ_LEN,
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
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
    ],
)
@triton.jit
def _hstu_attn_bwd(  # noqa C901
    Q,
    K,
    V,
    sort_by_length_indices,
    seq_offsets,
    num_targets,
    DOut,
    DQ,
    DK,
    DV,
    LOCK,
    stride_qm,
    stride_qh,
    stride_kn,
    stride_kh,
    stride_vn,
    stride_vh,
    stride_dom,
    stride_doh,
    stride_dqm,
    stride_dqh,
    stride_dkn,
    stride_dkh,
    stride_dvn,
    stride_dvh,
    alpha,
    contextual_seq_len,
    Z,
    AUTOTUNE_Z,
    H,
    MAX_SEQ_LEN,
    AUTOTUNE_MAX_SEQ_LEN,  # Quantized MAX_SEQ_LEN used as an autotuning key
    DimQ,
    DimV,
    MAX_ATTN_LEN: tl.constexpr,
    CAUSAL: tl.constexpr,
    HAS_MULTIPLE_TARGETS: tl.constexpr,
    HAS_CONTEXTUAL_SEQ_LEN: tl.constexpr,
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
    if SEQUENCE_PARALLEL:
        start_n = tl.program_id(1) * BLOCK_N
        if start_n >= seq_len:
            return
        _hstu_attn_bwd_one_col_block(
            start_n=start_n,
            seq_len=seq_len,
            n_targets=n_targets,
            contextual_seq_len=contextual_seq_len,
            Q=Q,
            K=K,
            V=V,
            DOut=DOut,
            DQ=DQ,
            DK=DK,
            DV=DV,
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
            MAX_ATTN_LEN=MAX_ATTN_LEN,
            CAUSAL=CAUSAL,
            HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
            HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
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
            _hstu_attn_bwd_one_col_block(
                start_n=start_n,
                seq_len=seq_len,
                n_targets=n_targets,
                contextual_seq_len=contextual_seq_len,
                Q=Q,
                K=K,
                V=V,
                DOut=DOut,
                DQ=DQ,
                DK=DK,
                DV=DV,
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
                MAX_ATTN_LEN=MAX_ATTN_LEN,
                CAUSAL=CAUSAL,
                HAS_MULTIPLE_TARGETS=HAS_MULTIPLE_TARGETS,
                HAS_CONTEXTUAL_SEQ_LEN=HAS_CONTEXTUAL_SEQ_LEN,
                ALLOW_TF32=ALLOW_TF32,
                BLOCK_D_Q=BLOCK_D_Q,
                BLOCK_D_V=BLOCK_D_V,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                UNROLL=UNROLL,
                ATOMIC_ADD=False,
            )


def triton_hstu_attention_fwd(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    Z = seq_offsets.numel() - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = q.shape
    _, _, DimV = v.shape
    out = torch.empty_like(v)
    has_multiple_targets = num_targets is not None
    has_contextual_seq_len = contextual_seq_len > 0
    has_sort_by_length_indices = sort_by_length_indices is not None
    if L == 0:
        return out

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_M"]),
        Z * H,
    )

    _hstu_attn_fwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
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
        contextual_seq_len=contextual_seq_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=has_multiple_targets,
        IS_DELTA_Q=False,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        MAX_ATTN_LEN=max_attn_len,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_SORT_BY_LENGTH_INDICES=has_sort_by_length_indices,
    )
    return out


def triton_hstu_attention_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    N: int,
    alpha: float,
    max_attn_len: int,
    causal: float,
    contextual_seq_len: int,
    sort_by_length_indices: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dout = switch_to_contiguous_if_needed(dout)
    dq = switch_to_contiguous_if_needed(dq)
    dk = switch_to_contiguous_if_needed(dk)
    dv = switch_to_contiguous_if_needed(dv)
    if dout.shape[0] == 0:
        return torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
    Z = seq_offsets.numel() - 1
    _, H, DimQ = q.shape
    _, _, DimV = v.shape
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
    _hstu_attn_bwd[grid](
        Q=q,
        K=k,
        V=v,
        sort_by_length_indices=sort_by_length_indices,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        DOut=dout,
        DQ=dq,
        DK=dk,
        DV=dv,
        LOCK=lock,
        stride_qm=q.stride(0),
        stride_qh=q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_dom=dout.stride(0),
        stride_doh=dout.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqh=dq.stride(1),
        stride_dkn=dk.stride(0),
        stride_dkh=dk.stride(1),
        stride_dvn=dv.stride(0),
        stride_dvh=dv.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        MAX_ATTN_LEN=max_attn_len,
        CAUSAL=causal,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        HAS_CONTEXTUAL_SEQ_LEN=contextual_seq_len > 0,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_SORT_BY_LENGTH_INDICES=sort_by_length_indices is not None,
    )

    return dq, dk, dv


class _AttentionFunction(torch.autograd.Function):
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
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int,
        contextual_seq_len: int,
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
        if sort_by_length_indices is not None:
            saved_tensors.append(sort_by_length_indices)
        ctx.save_for_backward(*saved_tensors)
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.has_multiple_targets = num_targets is not None
        ctx.max_attn_len = max_attn_len
        ctx.N = N
        ctx.contextual_seq_len = contextual_seq_len
        ctx.sort_by_length = sort_by_length
        return triton_hstu_attention_fwd(
            N=N,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length_indices=sort_by_length_indices,
        )

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor
    ) -> Tuple[
        None,
        None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
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
            if ctx.sort_by_length:
                sort_by_length_indices = ctx.saved_tensors[idx]
            else:
                sort_by_length_indices = None

            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            dq, dk, dv = triton_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=seq_offsets,
                num_targets=num_targets,
                N=ctx.N,
                alpha=ctx.alpha,
                max_attn_len=ctx.max_attn_len,
                causal=ctx.causal,
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
                None,
                None,
                None,
            )


@torch.fx.wrap
def triton_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
) -> torch.Tensor:
    return _AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
    )


@torch.fx.wrap
def triton_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    Z = seq_offsets.size(0) - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = delta_q.shape
    DeltaSize = L // Z
    _, _, DimV = v.shape
    out = torch.empty((L, H, DimV), dtype=delta_q.dtype, device=delta_q.device)
    grid = lambda meta: (  # noqa E731
        triton.cdiv(DeltaSize, meta["BLOCK_M"]),
        Z * H,
    )
    has_contextual_seq_len = contextual_seq_len > 0
    _hstu_attn_fwd[grid](
        Q=delta_q,
        K=k,
        V=v,
        sort_by_length_indices=None,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        Out=out,
        stride_qm=delta_q.stride(0),
        stride_qh=delta_q.stride(1),
        stride_kn=k.stride(0),
        stride_kh=k.stride(1),
        stride_vn=v.stride(0),
        stride_vh=v.stride(1),
        stride_om=out.stride(0),
        stride_oh=out.stride(1),
        alpha=alpha,
        contextual_seq_len=contextual_seq_len,
        Z=Z,
        AUTOTUNE_Z=AUTOTUNE_Z,
        H=H,
        MAX_SEQ_LEN=N,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
        DimQ=DimQ,
        DimV=DimV,
        DeltaSize=DeltaSize,
        MAX_ATTN_LEN=max_attn_len,
        CAUSAL=True,
        HAS_MULTIPLE_TARGETS=num_targets is not None,
        IS_DELTA_Q=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_D_Q=DimQ,
        BLOCK_D_V=DimV,
        HAS_CONTEXTUAL_SEQ_LEN=has_contextual_seq_len,
        HAS_SORT_BY_LENGTH_INDICES=False,
    )
    return out

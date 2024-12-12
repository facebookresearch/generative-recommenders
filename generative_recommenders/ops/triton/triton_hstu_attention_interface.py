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

from typing import Optional

import torch

# @manual=//triton:triton
import triton

from generative_recommenders.common import (
    autotune_max_seq_len,
    prev_power_of_2,
    switch_to_contiguous_if_needed,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    _hstu_attn_fwd,
    AttentionFunction,
)

torch.fx.wrap("switch_to_contiguous_if_needed")


@torch.fx.wrap
def _fancy_empty(
    n: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    # We need to trick fx tracer to falsely think fancy_emtpy is dependent on input,
    # so that it does not create "_tensor_constant*".
    return torch.empty(0, dtype=dtype)


@torch.fx.wrap
def native_triton_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    seq2_offsets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
) -> torch.Tensor:
    return AttentionFunction.apply(
        N,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        invalid_attn_mask_type,
        num_targets,
        attn_bias,
        seq2_offsets,
        max_attn_len,
        contextual_seq_len,
        sort_by_length,
    )


def triton_hstu_mha(
    N: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    seq2_offsets: Optional[torch.Tensor] = None,
    triton_cc: bool = False,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
) -> torch.Tensor:
    q = switch_to_contiguous_if_needed(q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)
    seq_offsets = seq_offsets.contiguous()
    if attn_bias is not None:
        attn_bias = attn_bias.contiguous()
    if seq2_offsets is not None:
        seq2_offsets = seq2_offsets.contiguous()

    if triton_cc:
        assert contextual_seq_len == 0
        return native_triton_hstu_mha(
            N,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            num_targets,
            attn_bias,
            seq2_offsets,
            max_attn_len,
            contextual_seq_len,
            sort_by_length,
        )
    else:
        return native_triton_hstu_mha(
            N,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            invalid_attn_mask_type,
            num_targets,
            attn_bias,
            seq2_offsets,
            max_attn_len,
            contextual_seq_len,
            sort_by_length,
        )


def triton_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    triton_cc: bool = False,
    max_attn_len: Optional[int] = None,
) -> torch.Tensor:
    seq_offsets = seq_offsets.contiguous()
    delta_x_offsets = delta_x_offsets.contiguous()
    delta_q = switch_to_contiguous_if_needed(delta_q)
    k = switch_to_contiguous_if_needed(k)
    v = switch_to_contiguous_if_needed(v)

    Z = seq_offsets.size(0) - 1
    AUTOTUNE_Z = prev_power_of_2(Z)
    L, H, DimQ = delta_q.shape
    DeltaSize = L // Z

    if triton_cc:
        L, _, _ = delta_q.shape
        _, H, DimV = v.shape
        out = torch.empty((L, H, DimV), dtype=delta_q.dtype, device=delta_q.device)
        return out
    else:
        L, _, _ = delta_q.shape
        _, H, DimV = v.shape
        out = torch.empty((L, H, DimV), dtype=delta_q.dtype, device=delta_q.device)
        grid = lambda meta: (  # noqa E731
            triton.cdiv(DeltaSize, meta["BLOCK_M"]),
            Z * H,
        )

        _hstu_attn_fwd[grid](
            Q=delta_q,
            K=k,
            V=v,
            sort_by_length_indices=None,
            seq_offsets=seq_offsets,
            TS=None,
            TW=None,
            PW=None,
            Bias=None,
            seq2_offsets=None,
            delta_x_offsets=delta_x_offsets,
            num_targets=num_targets,
            Out=out,
            stride_qm=delta_q.stride(0),
            stride_qh=delta_q.stride(1),
            stride_kn=k.stride(0),
            stride_kh=k.stride(1),
            stride_vn=v.stride(0),
            stride_vh=v.stride(1),
            stride_ts=None,
            stride_om=out.stride(0),
            stride_oh=out.stride(1),
            alpha=alpha,
            CONTEXTUAL_SEQ_LEN=0,
            Z=Z,
            AUTOTUNE_Z=AUTOTUNE_Z,
            H=H,
            MAX_SEQ_LEN=N,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(N),
            DimQ=DimQ,
            DimV=DimV,
            DeltaSize=DeltaSize,
            num_buckets=None,
            max_pos_ind=None,
            MAX_ATTN_LEN=max_attn_len or 0,
            time_bucket_incr=None,
            time_bucket_div=None,
            time_delta=None,
            INVALID_MASK_TYPE="lower_triangular",
            CAUSAL=None,
            BUCKET_FN="none",
            ATTN_BIAS_TYPE="none",
            USE_TIME_BIAS=False,
            USE_POS_BIAS=False,
            HAS_MAX_POS_IND=False,
            HAS_MULTIPLE_TARGETS=num_targets is not None,
            IS_DELTA_Q=True,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
            BLOCK_D_Q=DimQ,
            BLOCK_D_V=DimV,
            HAS_SORT_BY_LENGTH_INDICES=False,
        )
        return out

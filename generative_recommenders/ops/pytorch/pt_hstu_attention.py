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

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@torch.fx.wrap
def get_invalid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    invalid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    invalid_attn_mask = torch.logical_or(invalid_attn_mask, row_col_dist > 0)
    if max_attn_len is not None and max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            invalid_attn_mask = torch.logical_and(
                invalid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            invalid_attn_mask = torch.logical_and(
                invalid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        invalid_attn_mask = torch.logical_or(
            invalid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return invalid_attn_mask


@torch.fx.wrap
def time_bucketization_fn(
    x: torch.Tensor, bucket_function: str, bucket_incr: float, final_div: float
) -> torch.Tensor:
    x = x.clamp(min=1e-6) / bucket_incr
    if bucket_function == "log":
        x = torch.log(x)
    elif bucket_function == "sqrt":
        x = torch.sqrt(x)
    else:
        raise Exception(f"Invalid time bucket function {bucket_function}.")
    return (x / final_div).clamp(min=0).int()


@torch.fx.wrap
def get_time_weights(
    ts: torch.Tensor,
    ts_weights: torch.Tensor,
    bucket_function: str,
    bucket_incr: float,
    final_div: float,
    delta: float,
    num_buckets: int,
) -> torch.Tensor:
    ts = time_bucketization_fn(ts + delta, bucket_function, bucket_incr, final_div)
    ts = torch.clamp(
        ts,
        min=0,
        max=num_buckets,
    )
    return torch.index_select(ts_weights.view(-1), index=ts.view(-1), dim=0)


@torch.fx.wrap
def time_bias_fn(
    ts: torch.Tensor,
    ts_weights: torch.Tensor,
    causal: bool,
    bucket_function: str,
    bucket_incr: float,
    final_div: float,
    delta: float,
    num_buckets: int,
    N: int,
) -> torch.Tensor:
    if causal:
        ts = ts[:, 1:].unsqueeze(2) - ts[:, :-1].unsqueeze(1)
    else:
        ts = ts[:, :-1].unsqueeze(2) - ts[:, 1:].unsqueeze(1)
    return get_time_weights(
        ts.view(-1),
        ts_weights,
        bucket_function,
        bucket_incr,
        final_div,
        delta,
        num_buckets,
    ).view(-1, N, N)


@torch.fx.wrap
def get_pos_weights(
    N: int,
    pos_ids: torch.Tensor,
    pos_weights: torch.Tensor,
    max_pos_ind: Optional[int] = None,
) -> torch.Tensor:
    if max_pos_ind is not None:
        pos_ids = pos_ids + max_pos_ind - 1
        pos_ids = torch.clamp(pos_ids, min=0, max=2 * max_pos_ind - 2)
    else:
        pos_ids = pos_ids + N - 1
    return torch.index_select(pos_weights, 0, pos_ids.view(-1))


@torch.fx.wrap
def pos_bias_fn(
    pos_weights: torch.Tensor,
    N: int,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    num_targets: Optional[torch.Tensor] = None,
    max_pos_ind: Optional[int] = None,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=pos_weights.device)
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    max_ids = seq_lengths.view(-1, 1, 1) - 1
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids + 1,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    pos_ids = col_ids - row_ids
    return get_pos_weights(N, pos_ids, pos_weights, max_pos_ind).view(-1, N, N)


@torch.fx.wrap
def pytorch_relative_bias(
    N: int,
    seq_offsets: torch.Tensor,
    ts: torch.Tensor,
    ts_weights: torch.Tensor,
    pos_weights: torch.Tensor,
    num_buckets: int,
    causal: bool,
    bucket_function: str,
    bucket_incr: float,
    final_div: float,
    delta: float = 0.0,
    invalid_attn_mask_type: str = "lower_triangular",
    num_targets: Optional[torch.Tensor] = None,
    max_pos_ind: Optional[int] = None,
    relative_bias_type: str = "ALL",
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    if relative_bias_type == "ALL":
        bias = time_bias_fn(
            ts=ts,
            ts_weights=ts_weights,
            causal=causal,
            bucket_function=bucket_function,
            bucket_incr=bucket_incr,
            final_div=final_div,
            delta=delta,
            num_buckets=num_buckets,
            N=N,
        ) + pos_bias_fn(
            pos_weights=pos_weights,
            N=N,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            num_targets=num_targets,
            max_pos_ind=max_pos_ind,
        )
    elif relative_bias_type == "TIME":
        bias = time_bias_fn(
            ts=ts,
            ts_weights=ts_weights,
            causal=causal,
            bucket_function=bucket_function,
            bucket_incr=bucket_incr,
            final_div=final_div,
            delta=delta,
            num_buckets=num_buckets,
            N=N,
        )
    else:
        bias = pos_bias_fn(
            pos_weights=pos_weights,
            N=N,
            seq_offsets=seq_offsets,
            invalid_attn_mask_type=invalid_attn_mask_type,
            num_targets=num_targets,
            max_pos_ind=max_pos_ind,
        )
    return bias.unsqueeze(1)


def pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    invalid_attn_mask_type: str,
    dropout_pr: float,
    training: bool,
    num_targets: Optional[torch.Tensor] = None,
    seq2_offsets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    attn_scale: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    if attn_bias is not None:
        qk_attn = qk_attn + attn_bias
    qk_attn = F.silu(qk_attn) / max_seq_len
    invalid_attn_mask = get_invalid_attn_mask(
        device=q.device,
        causal=(invalid_attn_mask_type == "lower_triangular"),
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    qk_attn = qk_attn * invalid_attn_mask.unsqueeze(1)
    if attn_scale is not None:
        if attn_scale.dim() == 1:
            Z = seq_offsets.numel() - 1
            attn_scale = attn_scale.expand(Z, max_seq_len)
        qk_attn = qk_attn * attn_scale.unsqueeze(1).unsqueeze(-1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def get_delta_invalid_attn_mask(
    max_seq_len: int,
    delta_x_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
) -> torch.Tensor:
    B = seq_lengths.size(0)
    ids = torch.arange(0, max_seq_len, device=delta_x_offsets.device)
    col_ids = ids.view(1, 1, max_seq_len)
    row_ids = delta_x_offsets.view(B, -1) - seq_offsets[:-1].view(-1, 1)
    row_ids = row_ids.view(B, -1, 1)
    invalid_attn_mask = col_ids == row_ids
    if num_targets is not None:
        seq_lengths = seq_lengths.view(-1, 1, 1)
        num_targets = num_targets.view(-1, 1, 1)
        row_ids = torch.clamp(row_ids, max=seq_lengths - num_targets)
        col_ids = torch.clamp(col_ids, max=seq_lengths - num_targets)
    row_col_dist = row_ids - col_ids
    invalid_attn_mask = torch.logical_or(invalid_attn_mask, row_col_dist > 0)
    if max_attn_len is not None:
        invalid_attn_mask = torch.logical_and(
            invalid_attn_mask, row_col_dist <= max_attn_len
        )
    return invalid_attn_mask.unsqueeze(1)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    N: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    attn_bias: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2)
    full_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    if attn_bias is not None:
        qk_attn = qk_attn + attn_bias
    qk_attn = F.silu(qk_attn) / N
    invalid_attn_mask = get_delta_invalid_attn_mask(
        max_seq_len=N,
        delta_x_offsets=delta_x_offsets,
        seq_lengths=seq_lengths,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
    )
    qk_attn = qk_attn * invalid_attn_mask
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)

    return attn_output.transpose(1, 2).reshape(-1, H, V)

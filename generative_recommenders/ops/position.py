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
from generative_recommenders.ops.pytorch.pt_position import (
    pytorch_add_position_embeddings,
    pytorch_add_timestamp_positional_embeddings,
)

try:
    from hammer.ops.triton.cc.add_timestamp_position_embeddings.triton_cc_add_timestamp_position_embeddings import (
        triton_cc_add_timestamp_position_embeddings,
    )
except ImportError:
    pass
from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.triton.triton_position import (
    triton_add_position_embeddings,
    triton_add_timestamp_positional_embeddings,
)
from torch.fx._symbolic_trace import is_fx_tracing


@torch.fx.wrap
def _get_high_inds(
    high_inds: torch.Tensor,
    position_embeddings_weight: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    max_pos_ind = position_embeddings_weight.size(0)
    if num_targets is not None:
        if interleave_targets:
            high_inds = high_inds - num_targets * 2
        else:
            high_inds = high_inds - num_targets
    high_inds = torch.clamp(high_inds, max=max_pos_ind - 1)
    return high_inds


def add_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    high_inds = _get_high_inds(
        seq_lengths, position_embeddings_weight, num_targets, interleave_targets
    )
    if not is_fx_tracing():
        _, D = seq_embeddings.shape
        torch._assert(
            seq_offsets.size(0) - 1 == high_inds.size(0),
            "wrong jagged_offsets shape[0]",
        )
        _, D2 = position_embeddings_weight.shape
        torch._assert(D2 == D, "wrong dense shape[1]")

    if kernel == HammerKernel.TRITON:
        return triton_add_position_embeddings(
            jagged=seq_embeddings,
            jagged_offsets=seq_offsets,
            high_inds=high_inds,
            max_seq_len=max_seq_len,
            dense=position_embeddings_weight,
            scale=alpha,
        )
    else:
        return pytorch_add_position_embeddings(
            jagged=seq_embeddings,
            jagged_offsets=seq_offsets,
            high_inds=high_inds,
            max_seq_len=max_seq_len,
            dense=position_embeddings_weight,
            scale=alpha,
        )


def add_timestamp_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    assert time_bucket_fn in ["sqrt", "log"]
    seq_embeddings = seq_embeddings * alpha
    if kernel == HammerKernel.TRITON:
        return triton_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    elif kernel == HammerKernel.TRITON_CC:
        return triton_cc_add_timestamp_position_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    else:
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )

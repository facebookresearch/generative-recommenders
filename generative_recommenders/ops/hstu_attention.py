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

from generative_recommenders.common import HammerKernel
from generative_recommenders.ops.pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)
from generative_recommenders.ops.triton.triton_hstu_attention import (
    triton_cached_hstu_mha,
    triton_hstu_mha,
)
from torch.fx._symbolic_trace import is_fx_tracing


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: Optional[int] = None,
    contextual_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    _, H, _ = q.shape
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(q.dim() == 3, "q must be 3-D")
        torch._assert(k.shape == q.shape, "k must be the same shape as q")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")
        if max_attn_len is not None:
            torch._assert(max_attn_len > 0, "max_attn_len must be larger than 0")
        torch._assert(causal, "only support causal attention")

    if kernel in [HammerKernel.TRITON, HammerKernel.TRITON_CC]:
        if not is_fx_tracing() and kernel == HammerKernel.TRITON:
            torch._assert(q.is_cuda, "q must be CUDA tensor")
            torch._assert(k.is_cuda, "k must be CUDA tensor")
            torch._assert(v.is_cuda, "v must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")
        return triton_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
            triton_cc=kernel == HammerKernel.TRITON_CC,
        )
    else:
        return pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            dropout_pr=dropout_pr,
            training=training,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )


def cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    delta_x_offsets: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    B = seq_offsets.size(0) - 1
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(delta_q.dim() == 3, "delta_q must be 3-D")
        torch._assert(L % B == 0, "delta_q must be padded")
        torch._assert(
            delta_x_offsets.numel() == L,
            "wrong delta_x_offsets shape",
        )
        torch._assert(k.dim() == 3, "k must be 3-D")
        torch._assert(k.shape[1] == H, "wrong k shape[1]")
        torch._assert(k.shape[2] == D, "wrong k shape[2]")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")

    if kernel in [HammerKernel.TRITON, HammerKernel.TRITON_CC]:
        if not is_fx_tracing() and kernel == HammerKernel.TRITON:
            torch._assert(delta_q.is_cuda, "q must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            torch._assert(
                delta_x_offsets.is_cuda, "delta_x_offsets must be CUDA tensor"
            )
            if num_targets is not None:
                torch._assert(num_targets.is_cuda, "num_targets must be CUDA tensor")
        return triton_cached_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            delta_x_offsets=delta_x_offsets,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            triton_cc=(kernel == HammerKernel.TRITON_CC),
        )
    else:
        return pytorch_cached_hstu_mha(
            N=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            delta_x_offsets=delta_x_offsets,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            attn_bias=None,
        )
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

# pyre-strict

from typing import Optional, Tuple

import hstu_flash_attention  # @manual  # pyre-ignore[21]
import torch


class HSTUFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,  # pyre-ignore[2]
        max_seq_len: int,
        alpha: float,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        seq_offsets: Optional[torch.Tensor],
        causal: bool,
        num_targets: Optional[torch.Tensor],
        max_attn_len: int = 0,
        contextual_seq_len: int = 0,
        q_descale: Optional[torch.Tensor] = None,
        k_descale: Optional[torch.Tensor] = None,
        v_descale: Optional[torch.Tensor] = None,
        sort_by_length: bool = False,
        deterministic: bool = False,
        sm_margin: int = 0,
    ) -> torch.Tensor:
        out = hstu_flash_attention.forward(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            seq_offsets,
            causal,
            num_targets,
            max_attn_len,
            contextual_seq_len,
            q_descale,
            k_descale,
            v_descale,
            sm_margin,
        )
        saved_tensors = [q, k, v]
        if seq_offsets is not None:
            saved_tensors.append(seq_offsets)
        if num_targets is not None:
            saved_tensors.append(num_targets)
        if q_descale is not None:
            saved_tensors.append(q_descale)
        if k_descale is not None:
            saved_tensors.append(k_descale)
        if v_descale is not None:
            saved_tensors.append(v_descale)
        ctx.save_for_backward(*saved_tensors)
        ctx.max_seq_len = max_seq_len
        ctx.alpha = alpha
        ctx.causal = causal
        ctx.max_attn_len = max_attn_len
        ctx.contextual_seq_len = contextual_seq_len
        ctx.deterministic = deterministic
        ctx.sm_margin = sm_margin
        ctx.has_seq_offsets = seq_offsets is not None
        ctx.has_num_targets = num_targets is not None
        ctx.has_q_descale = q_descale is not None
        ctx.has_k_descale = k_descale is not None
        ctx.has_v_descale = v_descale is not None
        ctx.sort_by_length = sort_by_length
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx,  # pyre-ignore[2]
        dout: torch.Tensor,
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
        None,
        None,
        None,
        None,
        None,
    ]:
        idx = 3
        seq_offsets: Optional[torch.Tensor] = None
        num_targets: Optional[torch.Tensor] = None
        q_descale: Optional[torch.Tensor] = None
        k_descale: Optional[torch.Tensor] = None
        v_descale: Optional[torch.Tensor] = None
        q, k, v = ctx.saved_tensors[:idx]
        if ctx.has_seq_offsets:
            seq_offsets = ctx.saved_tensors[idx]
            idx += 1
        if ctx.has_num_targets:
            num_targets = ctx.saved_tensors[idx]
            idx += 1
        if ctx.has_q_descale:
            q_descale = ctx.saved_tensors[idx]
            idx += 1
        if ctx.has_k_descale:
            k_descale = ctx.saved_tensors[idx]
            idx += 1
        if ctx.has_v_descale:
            v_descale = ctx.saved_tensors[idx]
            idx += 1

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        dq, dk, dv = hstu_flash_attention.backward(
            ctx.max_seq_len,
            ctx.alpha,
            dout,
            q,
            k,
            v,
            dq,
            dk,
            dv,
            seq_offsets,
            ctx.causal,
            num_targets,
            ctx.max_attn_len,
            ctx.contextual_seq_len,
            ctx.sort_by_length,
            ctx.deterministic,
            ctx.sm_margin,
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
            None,
            None,
            None,
            None,
            None,
        )


def cuda_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: Optional[torch.Tensor] = None,
    causal: bool = False,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    sort_by_length: bool = False,
    deterministic: bool = False,
    sm_margin: int = 0,
) -> torch.Tensor:
    """
    Arguments:
        q, k, v: (batch_size, seqlen, nheads, headdim) or (total_seqlen, nheads, headdim)
    """
    return HSTUFlashAttentionFunction.apply(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        q_descale,
        k_descale,
        v_descale,
        sort_by_length,
        deterministic,
        sm_margin,
    )

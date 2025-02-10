from typing import Tuple

import torch
from hammer.ops.layer_norm import layer_norm
from hammer.utils import HammerKernel


def select_experts_with_prob(
    x: torch.Tensor,  # [BN, D]
    wg: torch.Tensor,  # [D, E]
    topk: int,
    group_scale: int = 2,
):
    BN, D = x.shape
    _, E = wg.shape
    E2 = group_scale * E
    G = (BN + E2 - 1) // E2
    pad_size = G * E2 - BN
    padded_x = torch.cat(
        [
            x,
            torch.zeros([pad_size, x.size(1)], dtype=x.dtype, device=x.device),
        ],
        dim=0,
    ).view(G, E2, D)
    score = torch.matmul(padded_x, wg)
    _, indices = torch.topk(
        score, dim=1, k=topk * group_scale, sorted=True
    )  # [G, topk*group_scale, E]
    threshold = torch.gather(
        score, 1, torch.transpose(indices, 0, 1)[-1].unsqueeze(1)
    )  # [G, 1, E]
    filtered_prob = torch.where(score >= threshold, torch.softmax(score, 2), 0).view(
        -1, E
    )[:BN, :]
    return torch.topk(filtered_prob, k=topk, dim=1, sorted=False)


def apply_base_and_experts_v2(
    x: torch.Tensor,  # [BN, D]
    we: torch.Tensor,  # [E, D, inner_D]
    bias_e: torch.Tensor,  # [E, inner_D]
    w: torch.Tensor,  # [D, inner_D]
    bias: torch.Tensor,  # [inner_D]
    probs: torch.Tensor,  # [BN, topk]
    indices: torch.Tensor,  # [BN, topk]
    topk: int,
) -> torch.Tensor:
    BN, _ = x.shape
    E, _, inner_D = we.shape
    seq_id = torch.arange(BN, device=x.device).unsqueeze(1).expand(-1, topk)
    sum_prob = torch.sum(probs, dim=1) + 1.0
    out = torch.mm(x.to(torch.float32), w.to(torch.float32)) + bias.to(
        torch.float32
    )  # [BN, inner_D]
    for idx in range(0, E):
        mask = indices == idx
        selected_seq_id = torch.masked_select(seq_id, mask=mask)
        selected_probs = torch.masked_select(probs, mask=mask)
        selected_x = torch.index_select(x, 0, selected_seq_id)
        result = torch.mm(
            selected_x.to(torch.float32), we[idx].to(torch.float32)
        ) + bias_e[idx].to(torch.float32)
        out.scatter_add_(
            0,
            selected_seq_id.unsqueeze(-1).expand(-1, inner_D),
            result * selected_probs.unsqueeze(-1),
        )
    return (out / sum_prob.unsqueeze(1)).to(x.dtype)


def apply_base_and_experts(
    x: torch.Tensor,  # [BN, D]
    we: torch.Tensor,  # [E, D, inner_D]
    bias_e: torch.Tensor,  # [E, inner_D]
    w: torch.Tensor,  # [D, inner_D]
    bias: torch.Tensor,  # [inner_D]
    probs: torch.Tensor,  # [BN, topk]
    indices: torch.Tensor,  # [BN, topk]
    topk: int,
) -> torch.Tensor:
    BN, D = x.shape
    E, _, inner_D = we.shape
    we_expanded = torch.concat(
        [
            w.unsqueeze(0).expand(BN, -1, -1),
            torch.gather(
                we.transpose(0, 1).unsqueeze(0).broadcast_to(BN, D, E, inner_D),
                dim=2,
                index=indices.view(BN, 1, -1, 1).expand(-1, D, -1, inner_D),
            ).view(BN, D, -1),
        ],
        dim=2,
    )  # [BN, D, (topk+1)*inner_D]
    bias_expanded = torch.concat(
        [
            bias.view(1, 1, -1).expand(BN, 1, -1),
            torch.gather(
                bias_e.view(1, 1, E, inner_D).broadcast_to(BN, 1, E, inner_D),
                dim=2,
                index=indices.view(BN, 1, -1, 1).expand(-1, 1, -1, inner_D),
            ).view(BN, 1, -1),
        ],
        dim=2,
    )  # [BN, 1, (topk+1)*inner_D]
    gating_prob_with_shared = torch.concat(
        [
            torch.ones([BN, 1], dtype=x.dtype, device=x.device),
            probs,
        ],
        dim=1,
    )
    # L1 norm
    prob = gating_prob_with_shared / torch.sum(
        gating_prob_with_shared, dim=1, keepdim=True
    )  # [BN, (topk+1)]
    return torch.bmm(
        prob.unsqueeze(1),
        torch.baddbmm(bias_expanded, x.unsqueeze(1), we_expanded).view(
            BN, topk + 1, inner_D
        ),
    ).squeeze(1)


def pytroch_hstu_moe_linear(
    x: torch.Tensor,  # [BN, D]
    wg: torch.Tensor,  # [D, E]
    we: torch.Tensor,  # [E, D, inner_D]
    bias_e: torch.Tensor,  # [E, inner_D]
    w: torch.Tensor,  # [D, inner_D]
    bias: torch.Tensor,  # [inner_D]
    topk: int,
    group_scale: int = 2,
    kernel: HammerKernel = HammerKernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Performs a MOE (Mixture of Experts) linear layer.

    Args:
        x: Input tensor of shape [BN, D].
        wg: Weight tensor of shape [D, E] for gating.
        we: Weight tensor of shape [D, E, inner_D] for gated experts.
        w: Weight tensor of shape [D, inner_D] for the base expert.
        topk: Number of experts to select.
        group_scale: Group scale factor.
        kernel: Hammer kernel to use.

    Returns:
        Output tensor of shape [BN, topk, inner_D].
    """
    probs, indices = select_experts_with_prob(
        x=x, wg=wg, topk=topk, group_scale=group_scale
    )
    out = apply_base_and_experts_v2(
        x=x,
        we=we,
        bias_e=bias_e,
        w=w,
        bias=bias,
        probs=probs,
        indices=indices,
        topk=topk,
    )

    return out, probs.detach().clone(), indices.detach().clone()

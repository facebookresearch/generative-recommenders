import logging
from typing import Tuple

import torch

# @manual=//triton:triton
import triton

from generative_recommenders.common import (
    switch_to_contiguous_if_needed,
    triton_autotune,
)
from generative_recommenders.ops.pytorch.pt_hstu_moe_linear import (
    select_experts_with_prob,
)
from generative_recommenders.ops.triton.triton_addmm import _addmm_fwd

# @manual=//triton:triton
from triton import language as tl

DEBUG = False


def _get_jagged_linear_fwd_configs():
    configs = []
    for BLOCK_M in [16, 32]:
        for BLOCK_N in [16, 32, 64]:
            for BLOCK_K in [16, 32, 64]:
                configs.append(
                    triton.Config(
                        {
                            "BLOCK_M": BLOCK_M,
                            "BLOCK_N": BLOCK_N,
                            "BLOCK_K": BLOCK_K,
                        },
                        num_stages=2,
                        num_warps=4,
                    )
                )
    return configs


@triton.autotune(
    configs=_get_jagged_linear_fwd_configs(),
    key=["INNER_D", "D", "GROUP_SCALE", "M", "ALLOW_TF32"],
)
@triton.jit
def _jagged_linear_fwd(
    X,
    SEQ_OFFSETS,
    WE,
    BIAS_E,
    Z,
    stride_x: tl.constexpr,
    stride_seq_offsets: tl.constexpr,
    stride_we: tl.constexpr,
    stride_bias_e: tl.constexpr,
    stride_z: tl.constexpr,
    M: tl.constexpr,
    INNER_D: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    OFFSET_M = tl.program_id(0).to(tl.int64)
    EXPERT = tl.program_id(1).to(tl.int64)
    SEQ_OFFSETS += stride_seq_offsets * EXPERT

    m_start = (OFFSET_M * BLOCK_M).to(tl.int32)
    m_end = m_start + BLOCK_M
    if M < m_end:
        m_end = M
    expert_start = tl.load(SEQ_OFFSETS)
    expert_end = tl.load(SEQ_OFFSETS + stride_seq_offsets)
    if m_start >= expert_end or m_end <= expert_start:
        return

    OFFSET_N = tl.program_id(2).to(tl.int64)
    X += stride_x * D * BLOCK_M * OFFSET_M
    WE += stride_we * (INNER_D * D * EXPERT + BLOCK_N * OFFSET_N)
    BIAS_E += stride_bias_e * (INNER_D * EXPERT + BLOCK_N * OFFSET_N)
    Z += stride_z * (INNER_D * BLOCK_M * OFFSET_M + BLOCK_N * OFFSET_N)

    m_offsets = tl.arange(0, BLOCK_M) + m_start
    m_mask = (m_offsets >= expert_start) & (m_offsets < expert_end)
    n_offsets = tl.arange(0, BLOCK_N) + BLOCK_N * OFFSET_N
    n_mask = n_offsets < INNER_D
    bias = tl.load(
        BIAS_E + stride_bias_e * tl.arange(0, BLOCK_N), mask=n_mask, other=0.0
    )
    z = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, D, BLOCK_K):
        k_offsets = tl.arange(0, BLOCK_K) + k
        k_mask = k_offsets < D
        x = tl.load(
            X + stride_x * (tl.arange(0, BLOCK_M)[:, None] * D + k_offsets[None, :]),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        w = tl.load(
            WE
            + stride_we
            * (k_offsets[:, None] * INNER_D + tl.arange(0, BLOCK_N)[None, :]),
            mask=k_mask[:, None] & n_mask[None, :],
            other=0.0,
        )
        z += tl.dot(x, w, allow_tf32=ALLOW_TF32)
    z += bias.to(tl.float32)
    tl.store(
        Z
        + stride_z
        * (tl.arange(0, BLOCK_M)[:, None] * INNER_D + tl.arange(0, BLOCK_N)[None, :]),
        z.to(Z.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


_jagged_linear_fwd = triton_autotune(
    configs=_get_jagged_linear_fwd_configs(),
    key=["INNER_D", "D", "M"],
)(_jagged_linear_fwd.fn)


def triton_apply_base_and_experts_v2_fwd(
    x: torch.Tensor,
    we: torch.Tensor,
    bias_e: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    gating_probs: torch.Tensor,
    gating_indices: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    BN, D = x.shape
    E, _, inner_D = we.shape

    z = torch.zeros([BN, inner_D], device=x.device, dtype=torch.float32)
    z = switch_to_contiguous_if_needed(z)
    _addmm_fwd[
        lambda meta: (
            triton.cdiv(BN, meta["BLOCK_M"]),
            triton.cdiv(inner_D, meta["BLOCK_N"]),
        )
    ](
        x,
        w,
        bias,
        z,
        M=BN,
        N=inner_D,
        K=D,
        stride_xm=x.stride(0),
        stride_xk=x.stride(-1),
        stride_wk=w.stride(0),
        stride_wn=w.stride(-1),
        stride_ym=0,
        stride_yn=bias.stride(-1),
        stride_zm=z.stride(0),
        stride_zn=z.stride(-1),
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BROADCAST_Y=True,
    )

    seq_id = torch.arange(BN, device=x.device, dtype=torch.int64).view(-1, 1)
    x_list = []
    probs_list = []
    selected_seq_ids_list = []
    seq_lens = []
    for idx in range(0, E):
        mask = gating_indices == idx
        selected_seq_id = torch.masked_select(seq_id, mask=mask)
        probs_list.append(torch.masked_select(gating_probs, mask=mask))
        x_list.append(torch.index_select(x, 0, selected_seq_id))
        selected_seq_ids_list.append(selected_seq_id)
        seq_lens.append(selected_seq_id.shape[0])
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(seq_lens, device=x.device, dtype=torch.int64)
    )
    jagged_x = torch.cat(x_list, dim=0)
    experts_z = torch.zeros([BN * topk, inner_D], device=x.device, dtype=torch.float32)
    _jagged_linear_fwd[
        lambda meta: (  # noqa E731
            triton.cdiv(BN * topk, meta["BLOCK_M"]),
            E,
            triton.cdiv(inner_D, meta["BLOCK_N"]),
        )
    ](
        jagged_x,
        seq_offsets,
        we,
        bias_e,
        experts_z,
        stride_x=jagged_x.stride(-1),
        stride_seq_offsets=seq_offsets.stride(-1),
        stride_we=we.stride(-1),
        stride_bias_e=bias_e.stride(-1),
        stride_z=experts_z.stride(-1),
        M=BN * topk,
        INNER_D=inner_D,
        D=D,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )

    jagged_probs = torch.cat(probs_list, dim=0)
    selected_seq_ids = torch.cat(selected_seq_ids_list, dim=0)
    z.scatter_add_(
        0,
        selected_seq_ids.unsqueeze(-1).expand(-1, inner_D),
        experts_z * jagged_probs.unsqueeze(-1),
    )
    sum_probs = torch.sum(gating_probs, dim=1, keepdim=True) + 1.0
    return (z / sum_probs).to(x.dtype)


def triton_moe_linear_fwd(
    x: torch.Tensor,
    wg: torch.Tensor,
    we: torch.Tensor,
    bias_e: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    group_scale: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = switch_to_contiguous_if_needed(x)
    wg = switch_to_contiguous_if_needed(wg)
    we = switch_to_contiguous_if_needed(we)
    w = switch_to_contiguous_if_needed(w)

    gating_probs, gating_indices = select_experts_with_prob(
        x=x, wg=wg, topk=topk, group_scale=group_scale
    )
    gating_probs = switch_to_contiguous_if_needed(gating_probs)
    gating_indices = switch_to_contiguous_if_needed(gating_indices)

    out = triton_apply_base_and_experts_v2_fwd(
        x=x,
        we=we,
        bias_e=bias_e,
        w=w,
        bias=bias,
        gating_probs=gating_probs,
        gating_indices=gating_indices,
        topk=topk,
    )
    if DEBUG:
        logging.info(
            f"[fwd] out: {out.detach().clone().cpu()[:2, :16]}, std: {out.detach().clone().std()}"
        )
    return out, gating_probs, gating_indices


def _get_jagged_linear_bwd_configs():
    configs = []
    for BLOCK_M in [16, 32]:
        for BLOCK_K in [32, 64, 128]:
            configs.append(
                triton.Config(
                    {
                        "BLOCK_M": BLOCK_M,
                        "BLOCK_K": BLOCK_K,
                    },
                    num_stages=2,
                    num_warps=4,
                )
            )
    return configs


@triton.autotune(
    configs=_get_jagged_linear_bwd_configs(),
    key=["INNER_D", "D", "M", "ALLOW_TF32"],
)
@triton.jit
def _jagged_linear_bwd(
    DOUT,
    X,
    PROB,
    WE,
    BIAS_E,
    SEQ_OFFSETS,
    DX,
    DWE,
    DBIAS_E,
    OUT_XW,
    OUT_BIAS,
    stride_dout: tl.constexpr,
    stride_x: tl.constexpr,
    stride_prob: tl.constexpr,
    stride_we: tl.constexpr,
    stride_bias_e: tl.constexpr,
    stride_seq_offsets: tl.constexpr,
    stride_dx: tl.constexpr,
    stride_dwe: tl.constexpr,
    stride_dbias_e: tl.constexpr,
    stride_out_xw: tl.constexpr,
    stride_out_bias: tl.constexpr,
    M: tl.constexpr,
    INNER_D: tl.constexpr,
    D: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    OFFSET_M = tl.program_id(0).to(tl.int64)
    EXPERT = tl.program_id(1).to(tl.int64)
    SEQ_OFFSETS += stride_seq_offsets * EXPERT

    m_start = (OFFSET_M * BLOCK_M).to(tl.int32)
    m_end = m_start + BLOCK_M
    if M < m_end:
        m_end = M
    expert_start = tl.load(SEQ_OFFSETS)
    expert_end = tl.load(SEQ_OFFSETS + stride_seq_offsets)
    if m_start >= expert_end or m_end <= expert_start:
        return

    OFFSET_N = tl.program_id(2).to(tl.int64)
    X += stride_x * D * BLOCK_M * OFFSET_M
    WE += stride_we * INNER_D * D * EXPERT
    BIAS_E += stride_bias_e * INNER_D * EXPERT
    DOUT += stride_dout * (INNER_D * BLOCK_M * OFFSET_M)
    DX += stride_dx * D * BLOCK_M * OFFSET_M
    DWE += stride_dwe * INNER_D * D * EXPERT
    DBIAS_E += stride_dbias_e * INNER_D * EXPERT
    PROB += stride_prob * BLOCK_M * OFFSET_M

    THEAD_N_NUM = tl.cdiv(D, BLOCK_N)
    OUT_XW += stride_out_xw * (INNER_D * (THEAD_N_NUM * BLOCK_M * OFFSET_M + OFFSET_N))
    OUT_BIAS += stride_out_bias * INNER_D * OFFSET_M * BLOCK_M

    m_offsets = tl.arange(0, BLOCK_M) + m_start
    m_mask = (m_offsets >= expert_start) & (m_offsets < expert_end)
    n_offsets = tl.arange(0, BLOCK_N) + BLOCK_N * OFFSET_N
    n_mask = n_offsets < D

    probs = tl.load(PROB + stride_prob * tl.arange(0, BLOCK_M), mask=m_mask, other=0.0)
    if OFFSET_N == 0:
        full_dout = tl.load(
            DOUT
            + stride_dout
            * (tl.arange(0, BLOCK_M)[:, None] * INNER_D + tl.arange(0, INNER_D)),
            mask=m_mask[:, None],
            other=0.0,
        )
        tl.atomic_add(
            DBIAS_E + stride_bias_e * tl.arange(0, INNER_D),
            tl.sum(full_dout * probs[:, None], axis=0),
        )
        full_bias = tl.load(BIAS_E + stride_bias_e * tl.arange(0, INNER_D))
        tl.store(
            OUT_BIAS
            + stride_out_bias
            * (
                tl.arange(0, BLOCK_M)[:, None] * INNER_D
                + tl.arange(0, INNER_D)[None, :]
            ),
            full_bias[None, :],
            mask=m_mask[:, None],
        )

    x = tl.load(
        X + stride_x * (tl.arange(0, BLOCK_M)[:, None] * D + n_offsets[None, :]),
        mask=m_mask[:, None] & n_mask[None, :],
        other=0.0,
    )
    dx = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, INNER_D, BLOCK_K):
        k_offsets = tl.arange(0, BLOCK_K) + k
        k_mask = k_offsets < INNER_D
        dout = tl.load(
            DOUT
            + stride_dout
            * (tl.arange(0, BLOCK_M)[:, None] * INNER_D + k_offsets[None, :]),
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        norm_dout = dout * probs[:, None]
        w = tl.load(
            WE + stride_we * (n_offsets[:, None] * INNER_D + k_offsets[None, :]),
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        tl.atomic_add(
            DWE + stride_dwe * (n_offsets[:, None] * INNER_D + k_offsets[None, :]),
            tl.dot(x.T.to(norm_dout.dtype), norm_dout, allow_tf32=ALLOW_TF32).to(
                DWE.dtype.element_ty
            ),
            mask=n_mask[:, None] & k_mask[None, :],
        )
        dx += tl.dot(norm_dout, w.T.to(norm_dout.dtype), allow_tf32=ALLOW_TF32)

        out_xw = tl.dot(x, w, allow_tf32=ALLOW_TF32)
        tl.store(
            OUT_XW
            + stride_out_xw
            * (
                tl.arange(0, BLOCK_M)[:, None] * INNER_D * THEAD_N_NUM
                + k_offsets[None, :]
            ),
            out_xw.to(OUT_XW.dtype.element_ty),
            mask=m_mask[:, None] & k_mask[None, :],
        )
    tl.store(
        DX + stride_dx * (tl.arange(0, BLOCK_M)[:, None] * D + n_offsets[None, :]),
        dx.to(DX.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


_jagged_linear_bwd = triton_autotune(
    configs=_get_jagged_linear_bwd_configs(),
    key=["INNER_D", "D", "M", "ALLOW_TF32"],
)(_jagged_linear_bwd.fn)


def triton_apply_base_and_experts_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    we: torch.Tensor,
    bias_e: torch.Tensor,
    w: torch.Tensor,
    gating_probs: torch.Tensor,
    gating_indices: torch.Tensor,
    topk: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    BN, D = x.shape
    E, _, inner_D = we.shape
    dwe = torch.zeros([E, D, inner_D], device=x.device, dtype=dout.dtype)
    dbias_e = torch.zeros([E, inner_D], device=x.device, dtype=dout.dtype)

    prob_sums = torch.sum(gating_probs, dim=1, keepdim=True) + 1.0
    norm_dout = (dout / prob_sums).to(dtype=dout.dtype)
    dw = torch.matmul(x.to(dtype=dout.dtype).t(), norm_dout)
    dbias = torch.sum(norm_dout, dim=0)
    dx = torch.matmul(norm_dout, w.to(dtype=dout.dtype).t())

    seq_id = torch.arange(BN, device=x.device, dtype=torch.int64).view(-1, 1)
    pos = torch.arange(topk, device=x.device, dtype=torch.int64).view(1, -1)
    x_list = []
    dout_list = []
    probs_list = []
    selected_seq_ids_list = []
    seq_lens = []
    pos_list = []
    for idx in range(0, E):
        mask = gating_indices == idx
        selected_seq_id = torch.masked_select(seq_id, mask=mask)
        probs_list.append(torch.masked_select(gating_probs / prob_sums, mask=mask))
        dout_list.append(torch.index_select(dout, 0, selected_seq_id))
        x_list.append(torch.index_select(x, 0, selected_seq_id))
        selected_seq_ids_list.append(selected_seq_id)
        pos_list.append(torch.masked_select(pos, mask=mask))
        seq_lens.append(selected_seq_id.shape[0])
    seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
        torch.tensor(seq_lens, device=x.device, dtype=torch.int64)
    )
    jagged_x = torch.cat(x_list, dim=0)
    jagged_dout = torch.cat(dout_list, dim=0)
    jagged_prob = torch.cat(probs_list, dim=0).unsqueeze(-1)
    djagged_x = torch.zeros([BN * topk, D], device=x.device, dtype=dout.dtype)
    BLOCK_N = 64
    out_xw = torch.zeros(
        [BN * topk, triton.cdiv(D, BLOCK_N), inner_D],
        device=x.device,
        dtype=x.dtype,
    )
    out_bias = torch.zeros([BN * topk, inner_D], device=x.device, dtype=x.dtype)
    _jagged_linear_bwd[
        lambda meta: (  # noqa E731
            triton.cdiv(BN * topk, meta["BLOCK_M"]),
            E,
            triton.cdiv(D, BLOCK_N),
        )
    ](
        jagged_dout,
        jagged_x,
        jagged_prob,
        we,
        bias_e,
        seq_offsets,
        djagged_x,
        dwe,
        dbias_e,
        out_xw,
        out_bias,
        stride_dout=jagged_dout.stride(-1),
        stride_x=jagged_x.stride(-1),
        stride_prob=jagged_prob.stride(-1),
        stride_we=we.stride(-1),
        stride_bias_e=bias_e.stride(-1),
        stride_seq_offsets=seq_offsets.stride(-1),
        stride_dx=djagged_x.stride(-1),
        stride_dwe=dwe.stride(-1),
        stride_dbias_e=dbias_e.stride(-1),
        stride_out_xw=out_xw.stride(-1),
        stride_out_bias=out_bias.stride(-1),
        M=BN * topk,
        INNER_D=inner_D,
        D=D,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        BLOCK_N=BLOCK_N,
    )
    selected_seq_ids = torch.cat(selected_seq_ids_list, dim=0)
    pos_ids = torch.cat(pos_list, dim=0)
    dx.scatter_add_(
        0,
        selected_seq_ids.unsqueeze(-1).expand(-1, inner_D),
        djagged_x,
    )
    djagged_prob = torch.sum(
        (torch.sum(out_xw, dim=1) + out_bias).to(dtype=dout.dtype) * jagged_dout, dim=-1
    )

    dprob = torch.sparse_coo_tensor(
        indices=torch.stack([selected_seq_ids, pos_ids], dim=0),
        values=djagged_prob,
        size=[BN, topk],
    ).to_dense()
    return dx, dwe, dbias_e, dw, dbias, dprob


def _get_gating_softmax_bwd_configs():
    configs = []
    configs.append(
        triton.Config(
            {},
            num_stages=2,
            num_warps=4,
        )
    )
    return configs


@triton.autotune(
    configs=_get_gating_softmax_bwd_configs(),
    key=["E", "TOP_K", "D"],
)
@triton.jit
def _gating_softmax_bwd(
    DPROBS,
    PROBS,
    INDICES,
    X,
    WG,
    DX,
    DWG,
    DSCORES,
    stride_dprobs: tl.constexpr,
    stride_probs: tl.constexpr,
    stride_indices: tl.constexpr,
    stride_x: tl.constexpr,
    stride_wg: tl.constexpr,
    stride_dx: tl.constexpr,
    stride_dwg: tl.constexpr,
    stride_dscores: tl.constexpr,
    E: tl.constexpr,
    TOP_K: tl.constexpr,
    D: tl.constexpr,
):
    OFFSET_M = tl.program_id(0).to(tl.int64)
    DPROBS += stride_dprobs * TOP_K * OFFSET_M
    PROBS += stride_probs * TOP_K * OFFSET_M
    INDICES += stride_indices * TOP_K * OFFSET_M
    X += stride_x * D * OFFSET_M
    DX += stride_dx * D * OFFSET_M
    DSCORES += stride_dscores * TOP_K * OFFSET_M

    probs = tl.load(PROBS + stride_probs * tl.arange(0, TOP_K))
    prob_sum = tl.sum(probs) + 1.0
    probs /= prob_sum
    pos = tl.arange(0, TOP_K)
    jacobian = tl.where(
        pos[:, None] == pos[None, :],
        (probs * (1.0 - probs))[:, None],
        -probs[:, None] * probs[None, :],
    )
    dprobs = tl.load(DPROBS + stride_dprobs * tl.arange(0, TOP_K))
    dscores = tl.sum(dprobs[:, None] * jacobian, axis=1)
    tl.store(
        DSCORES + stride_dscores * tl.arange(0, TOP_K), dscores, cache_modifier=".wb"
    )
    x = tl.load(X + stride_x * tl.arange(0, D))
    dx = tl.zeros([D], dtype=dprobs.dtype)
    dwg = tl.zeros([D, E], dtype=dprobs.dtype)
    wg_offsets = tl.arange(0, E)[None, :]
    wg_mask = tl.zeros([D, E], dtype=tl.int1)
    for i in range(0, TOP_K):
        e_id = tl.load(INDICES + stride_indices * i)
        dscore = tl.load(DSCORES + stride_dscores * i)
        w = tl.load(WG + stride_wg * (e_id + tl.arange(0, D) * E))
        dx += dscore * w
        dwg = tl.where(wg_offsets == e_id, dscore * x[:, None], dwg)
        wg_mask |= wg_offsets == e_id
    tl.atomic_add(DX + stride_dx * tl.arange(0, D), dx, sem="relaxed")
    tl.atomic_add(
        DWG + stride_dwg * (tl.arange(0, D)[:, None] * E + tl.arange(0, E)[None, :]),
        dwg,
        mask=wg_mask,
    )


_gating_softmax_bwd = triton_autotune(
    configs=_get_gating_softmax_bwd_configs(),
    key=["E", "TOP_K", "D"],
)(_gating_softmax_bwd.fn)


def select_experts_with_prob_bwd(
    dprob: torch.Tensor,  # [BN, topk]
    gating_probs: torch.Tensor,  # [BN, topk]
    gating_indices: torch.Tensor,  # [BN, topk]
    x: torch.Tensor,  # [BN, D]
    wg: torch.Tensor,  # [D, E]
    out_dx: torch.Tensor,  # [BN, D]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    BN, D = x.shape
    _, E = wg.shape
    _, TOP_K = dprob.shape
    dwg = torch.zeros([D, E], dtype=dprob.dtype, device=x.device)
    dscores = torch.zeros([BN, TOP_K], dtype=dprob.dtype, device=x.device)

    _gating_softmax_bwd[(BN,)](
        dprob,
        gating_probs,
        gating_indices,
        x,
        wg,
        out_dx,
        dwg,
        dscores,
        stride_dprobs=dprob.stride(-1),
        stride_probs=gating_probs.stride(-1),
        stride_indices=gating_indices.stride(-1),
        stride_x=x.stride(-1),
        stride_wg=wg.stride(-1),
        stride_dx=out_dx.stride(-1),
        stride_dwg=dwg.stride(-1),
        stride_dscores=dscores.stride(-1),
        E=E,
        TOP_K=TOP_K,
        D=D,
    )
    return out_dx, dwg, dscores


def triton_moe_linear_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    wg: torch.Tensor,
    we: torch.Tensor,
    bias_e: torch.Tensor,
    w: torch.Tensor,
    gating_probs: torch.Tensor,
    gating_indices: torch.Tensor,
    topk: int,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    dx, dwe, dbias_e, dw, dbias, dprob = triton_apply_base_and_experts_bwd(
        dout=dout,
        x=x,
        we=we,
        bias_e=bias_e,
        w=w,
        gating_probs=gating_probs,
        gating_indices=gating_indices,
        topk=topk,
    )
    if DEBUG:
        intermediate_dx = dx.detach().clone()

    dx, dwg, dscore = select_experts_with_prob_bwd(
        dprob=dprob,
        gating_probs=gating_probs,
        gating_indices=gating_indices,
        x=x,
        wg=wg,
        out_dx=dx,
    )
    if DEBUG:
        logging.info(
            f"dout: {dout.detach().clone().cpu()[:2, :16]}, std={dout.detach().clone().std()}\n"
            + f"x: {x.detach().clone().cpu()[:2, :16]}, std={x.detach().clone().std()}\n"
            + f"intermediate_dx: {intermediate_dx.cpu()[:2, :16]}, std={intermediate_dx.std()}\n"
            + f"dwe: {dwe.detach().clone().cpu()[:2, 0, :8]}, std={dwe.detach().clone().std()}\n"
            + f"dbias_e: {dbias_e.detach().clone().cpu()[:2, :8]}, std={dbias_e.detach().clone().std()}\n"
            + f"dw: {dw.detach().clone().cpu()[:2, :8]}, std={dw.detach().clone().std()}\n"
            + f"dbias: {dbias.detach().clone().cpu()[:8]}, std={dbias.detach().clone().std()}\n"
            + f"dprob: {dprob.detach().clone().cpu()[:2]}, std={dprob.detach().clone().std()}\n"
            + f"gating_probs: {gating_probs.detach().clone().cpu()[:2]}, std={gating_probs.detach().clone().std()}\n"
            + f"dscore: {dscore.detach().clone().cpu()[:2, :16]}, std={dscore.detach().clone().std()}\n"
            + f"out_dx: {dx.detach().clone().cpu()[:2, :16]}, std={dx.detach().clone().std()}\n"
            + f"dwg: {dwg.detach().clone().cpu()[:2, :16]}, std={dwg.detach().clone().std()}"
        )

    return (
        dx,
        dwg,
        dwe,
        dbias_e,
        dw,
        dbias,
    )


class HstuMoeLinearFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        x: torch.Tensor,  # [BN, D]
        wg: torch.Tensor,  # [D, E]
        we: torch.Tensor,  # [E, D, inner_D]
        bias_e: torch.Tensor,  # [E, inner_D]
        w: torch.Tensor,  # [D, inner_D]
        bias: torch.Tensor,  # [inner_D]
        topk: int,
        group_scale: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, probs, gating_indices = triton_moe_linear_fwd(
            x=x,
            wg=wg,
            we=we,
            bias_e=bias_e,
            w=w,
            bias=bias,
            topk=topk,
            group_scale=group_scale,
        )
        saved_tensors = [
            x,
            wg,
            we,
            bias_e,
            w,
            probs,
            gating_indices,
        ]
        ctx.save_for_backward(*saved_tensors)
        ctx.topk = topk
        return out, probs, gating_indices

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, dout: torch.Tensor, dprobs: torch.Tensor, dgating_indices: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        None,
        None,
        None,
    ]:
        del dprobs  # unused
        del dgating_indices  # unused
        (
            x,
            wg,
            we,
            bias_e,
            w,
            gating_probs,
            gating_indices,
        ) = ctx.saved_tensors
        dx, dwg, dwe, dbias_e, dw, dbias = triton_moe_linear_bwd(
            dout=dout,
            x=x,
            wg=wg,
            we=we,
            bias_e=bias_e,
            w=w,
            gating_probs=gating_probs,
            gating_indices=gating_indices,
            topk=ctx.topk,
        )
        # pyre-ignore[7]
        return (
            dx,
            dwg,
            dwe,
            dbias_e,
            dw,
            dbias,
            None,
            None,
            None,
        )


@torch.fx.wrap
def triton_hstu_moe_linear(
    x: torch.Tensor,  # [BN, D]
    wg: torch.Tensor,  # [D, E]
    we: torch.Tensor,  # [E, D, inner_D]
    bias_e: torch.Tensor,  # [E, inner_D]
    w: torch.Tensor,  # [D, inner_D]
    bias: torch.Tensor,  # [inner_D]
    topk: int,
    group_scale: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return HstuMoeLinearFunction.apply(
        x,
        wg,
        we,
        bias_e,
        w,
        bias,
        topk,
        group_scale,
    )

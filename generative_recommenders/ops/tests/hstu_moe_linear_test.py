import logging
import unittest
from typing import Optional

import torch
from generative_recommenders.common import (
    generate_sparse_seq_len,
    gpu_unavailable,
    set_dev_mode,
)
from hammer.utils import HammerKernel
from hypothesis import given, settings, strategies as st, Verbosity


def test_moe_linear(
    batch_size: int,
    max_seq_leng: int,
    model_dim: int,
    latent_dim: int,
    num_experts: int,
    topk: int,
    group_scale: int,
    dtype: torch.dtype,
    test_backward: bool,
    ref_kernel: HammerKernel,
    real_kernel: HammerKernel,
    skip_comparisons: bool = False,
    sparsity: float = -1.0,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
) -> None:
    set_dev_mode(True)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    from hammer.v2.ops.hstu_moe_linear import hstu_moe_linear

    if sparsity > 0.0:
        lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_seq_leng,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        lengths = torch.randint(
            max_seq_leng + 1, size=(batch_size,), device=torch.device("cuda")
        )
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())
    x = (
        torch.empty((L, model_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )

    gate_weight = (
        torch.empty(
            [model_dim, num_experts],
            dtype=dtype,
            device=torch.device("cuda"),
        )
        .normal_(mean=0, std=0.02)
        .requires_grad_()
    )
    weights = (
        torch.empty(
            [model_dim, latent_dim],
            dtype=dtype,
            device=torch.device("cuda"),
        )
        .normal_(mean=0, std=0.02)
        .requires_grad_()
    )
    bias = (
        torch.empty([latent_dim], dtype=dtype, device=torch.device("cuda"))
        .normal_()
        .requires_grad_()
    )
    expert_weights = (
        torch.empty(
            [num_experts, model_dim, latent_dim],
            dtype=dtype,
            device=torch.device("cuda"),
        )
        .normal_(mean=0, std=0.02)
        .requires_grad_()
    )
    expert_bias = (
        torch.empty(
            [num_experts, latent_dim],
            dtype=dtype,
            device=torch.device("cuda"),
        )
        .normal_()
        .requires_grad_()
    )

    # ref implementation
    ref_out, ref_probs, ref_indices = hstu_moe_linear(
        x=x,
        wg=gate_weight,
        we=expert_weights,
        bias_e=expert_bias,
        w=weights,
        bias=bias,
        topk=topk,
        group_scale=group_scale,
        kernel=ref_kernel,
    )
    dout = torch.randn_like(ref_out, dtype=torch.float32)
    ref_out.backward(dout)

    if skip_comparisons:
        return

    # pyre-ignore
    ref_dx, x.grad = x.grad.clone(), None

    ref_dgate_weight, gate_weight.grad = gate_weight.grad.clone(), None
    ref_dexpert_weights, expert_weights.grad = expert_weights.grad.clone(), None
    ref_dexpert_bias, expert_bias.grad = expert_bias.grad.clone(), None
    ref_dweights, weights.grad = weights.grad.clone(), None
    ref_dbias, bias.grad = bias.grad.clone(), None

    # triton implementation
    x = x.detach().clone().requires_grad_()
    gate_weight = gate_weight.detach().clone().requires_grad_()
    expert_weights = expert_weights.detach().clone().requires_grad_()
    expert_bias = expert_bias.detach().clone().requires_grad_()
    weights = weights.detach().clone().requires_grad_()
    bias = bias.detach().clone().requires_grad_()
    dout = dout.detach().clone()
    real_out, real_probs, real_indices = hstu_moe_linear(
        x=x,
        wg=gate_weight,
        we=expert_weights,
        bias_e=expert_bias,
        w=weights,
        bias=bias,
        topk=topk,
        group_scale=group_scale,
        kernel=real_kernel,
    )

    torch.testing.assert_close(ref_probs, real_probs, atol=atol, rtol=rtol)
    torch.testing.assert_close(ref_indices, real_indices, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        ref_out,
        real_out,
        atol=atol,
        rtol=rtol,
    )
    if test_backward:
        real_out.backward(dout)
        real_dx = x.grad.clone()
        real_dgate_weight = gate_weight.grad.clone()
        real_dexpert_weights = expert_weights.grad.clone()
        real_dexpert_bias = expert_bias.grad.clone()
        real_dweights = weights.grad.clone()
        real_dbias = bias.grad.clone()

        torch.testing.assert_close(ref_dweights, real_dweights, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_dbias, real_dbias, atol=atol, rtol=rtol)
        torch.testing.assert_close(
            ref_dexpert_weights, real_dexpert_weights, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            ref_dexpert_bias, real_dexpert_bias, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(
            ref_dgate_weight, real_dgate_weight, atol=atol, rtol=rtol
        )
        torch.testing.assert_close(ref_dx, real_dx, atol=atol, rtol=rtol)


class HSTUMoELinearTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @given(
        batch_size=st.integers(4, 8),
        max_seq_leng=st.sampled_from([20, 100, 128, 256]),
        model_dim=st.sampled_from([16, 32, 64, 128, 512]),
        latent_dim=st.sampled_from([16, 32, 64]),
        num_experts=st.sampled_from([16, 32]),
        topk=st.sampled_from([1, 2]),
        group_scale=st.sampled_from([2, 4]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=200,
        deadline=None,
    )
    def test_moe_linear_triton(self, *args, **kwargs) -> None:
        test_moe_linear(
            *args,
            **kwargs,
            test_backward=False,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

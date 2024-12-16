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

import unittest

from typing import Optional

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel, set_dev_mode
from generative_recommenders.ops.layer_norm import layer_norm, LayerNorm, SwishLayerNorm

from hypothesis import given, settings, strategies as st, Verbosity


class LayerNormTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.sampled_from([4200000]),
        D=st.sampled_from([512]),
        dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32]),
        learnable=st.sampled_from([False, True]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_layernorm_large_tensor(self, *args, **kwargs) -> None:
        self._test_layernorm_base(*args, **kwargs)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=32, max_value=256),
        D=st.integers(min_value=32, max_value=256),
        dtype=st.sampled_from([torch.float16, torch.bfloat16, torch.float32]),
        learnable=st.sampled_from([False, True]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_layernorm(self, *args, **kwargs) -> None:
        self._test_layernorm_base(*args, **kwargs)

    def _test_layernorm_base(
        self, N: int, D: int, learnable: bool, dtype: torch.dtype
    ) -> None:
        set_dev_mode(True)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        if learnable:
            weight = (
                torch.empty((D,), dtype=dtype, device=torch.device("cuda"))
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias = (
                torch.empty((D,), dtype=dtype, device=torch.device("cuda"))
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
        else:
            weight = None
            bias = None
        # ref
        ref_out = layer_norm(x, weight, bias, eps=1e-6, kernel=HammerKernel.PYTORCH)
        dout = torch.randn_like(ref_out) * 0.1
        ref_out.backward(dout)
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        if learnable:
            assert bias is not None and weight is not None
            ref_dw, weight.grad = weight.grad.detach().clone(), None
            ref_db, bias.grad = bias.grad.detach().clone(), None
        else:
            ref_dw = None
            ref_db = None
        # opt
        x = x.detach().clone().requires_grad_()
        if learnable:
            weight = weight.detach().clone().requires_grad_()
            bias = bias.detach().clone().requires_grad_()
        opt_out = layer_norm(x, weight, bias, eps=1e-6, kernel=HammerKernel.TRITON)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        opt_dx, x.grad = x.grad.detach().clone(), None
        if learnable:
            opt_dw, weight.grad = weight.grad.detach().clone(), None
            opt_db, bias.grad = bias.grad.detach().clone(), None
        else:
            opt_dw = None
            opt_db = None
        torch.testing.assert_close(ref_out, opt_out)
        torch.testing.assert_close(ref_dx, opt_dx)
        if learnable:
            torch.testing.assert_close(ref_dw, opt_dw)
            torch.testing.assert_close(ref_db, opt_db)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=500, max_value=1000),
        D=st.integers(min_value=256, max_value=512),
        dtype=st.sampled_from([torch.float32]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    def test_layer_norm_mul_dropout(self, N: int, D: int, dtype: torch.dtype) -> None:
        set_dev_mode(True)
        from hammer.ops.triton.triton_hstu_linear import triton_norm_mul_dropout

        dropout_ratio = 0.3
        attn = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        u = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        norm_weight = (
            torch.empty((D,), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        norm_bias = (
            torch.empty((D,), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        norm_eps = 1e-6
        # ref
        out = triton_norm_mul_dropout(
            x=attn,
            u=u,
            weight=norm_weight,
            bias=norm_bias,
            eps=norm_eps,
            dropout_ratio=dropout_ratio,
            training=True,
            group_norm=False,
        )
        # make sure dout does not have zero values
        dout = torch.randn_like(out) * 0.1 + 1e-6
        out.backward(dout)
        # pyre-ignore[16]
        du, u.grad = u.grad.detach().clone(), None
        mask_1 = out == 0
        mask_2 = du == 0
        torch.testing.assert_close(mask_1, mask_2)
        real_dropout_ratio = torch.sum(mask_1).cpu().item() / (N * D)
        assert abs(real_dropout_ratio - dropout_ratio) / dropout_ratio < 0.3

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=32, max_value=10000),
        D=st.integers(min_value=256, max_value=512),
        dtype=st.sampled_from(
            [torch.float32]
        ),  # currently we convert everything to float32 in the triton kernel anyways
        is_swish=st.sampled_from([True]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_swish_norm_triton(self, *args, **kwargs) -> None:
        self._test_layer_norm(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_layer_norm(
        self,
        N: int,
        D: int,
        dtype: torch.dtype,
        is_swish: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> None:
        set_dev_mode(True)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        # ref
        if is_swish:
            ref_layer = SwishLayerNorm(
                dim=D,
                eps=1e-6,
                dtype=dtype,
            ).to(device="cuda")
            ref_layer._hammer_kernel = HammerKernel.PYTORCH
        else:
            ref_layer = LayerNorm(
                dim=D,
                eps=1e-6,
                dtype=dtype,
            ).to(device="cuda")
            ref_layer._hammer_kernel = HammerKernel.PYTORCH
        ref_out = ref_layer(x.to(torch.float32))
        if test_backward:
            dout = torch.randn_like(ref_out) * 0.1
            ref_out.backward(dout)
            # pyre-ignore[16]
            ref_dx, x.grad = x.grad.detach().clone(), None
            ref_dw = ref_layer.weight.grad.detach().clone()
            ref_db = ref_layer.bias.grad.detach().clone()
        else:
            dout = None
            ref_dx = None
            ref_dw = None
            ref_db = None
        # opt
        x = x.detach().clone().requires_grad_()
        if is_swish:
            opt_layer = SwishLayerNorm(
                dim=D,
                eps=1e-6,
                dtype=dtype,
            ).to(device="cuda")
        else:
            opt_layer = LayerNorm(
                dim=D,
                eps=1e-6,
                dtype=dtype,
            ).to(device="cuda")
        opt_out = opt_layer(x)
        torch.testing.assert_close(ref_out, opt_out, atol=atol, rtol=rtol)
        if test_backward:
            dout = dout.detach().clone()
            opt_out.backward(dout)
            opt_dx, x.grad = x.grad.detach().clone(), None
            opt_dw = opt_layer.weight.grad.detach().clone()
            opt_db = opt_layer.bias.grad.detach().clone()
            torch.testing.assert_close(
                ref_dx,
                opt_dx,
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                ref_dw,
                opt_dw,
                atol=atol,
                rtol=rtol,
            )
            torch.testing.assert_close(
                ref_db,
                opt_db,
                atol=atol,
                rtol=rtol,
            )

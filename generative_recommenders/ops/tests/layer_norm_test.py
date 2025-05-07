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

import copy
import unittest

import torch
from generative_recommenders.common import gpu_unavailable, HammerKernel, set_dev_mode
from generative_recommenders.ops.layer_norm import (
    layer_norm,
    LayerNorm,
    swish_layer_norm,
    SwishLayerNorm,
)

from hypothesis import given, settings, strategies as st, Verbosity


class LayerNormTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.sampled_from([4200000]),
        D=st.sampled_from([512]),
        is_swish=st.sampled_from([False]),
        dtype=st.sampled_from(
            [torch.bfloat16]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=1,
    )
    # pyre-ignore[2]
    def test_large_tensors(self, *args, **kwargs) -> None:
        self._test_layernorm(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
            skip_comparisons=True,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=0, max_value=10000),
        D=st.integers(min_value=32, max_value=512),
        is_swish=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_ln(self, *args, **kwargs) -> None:
        self._test_layernorm(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_layernorm(
        self,
        N: int,
        D: int,
        is_swish: bool,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        skip_comparisons: bool = False,
    ) -> None:
        N = N // 4 * 4
        set_dev_mode(True)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        weight = (
            torch.empty((D,), device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        bias = (
            torch.empty((D,), device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        if is_swish:
            layer_norm_func = swish_layer_norm
        else:
            layer_norm_func = layer_norm
        # ref
        ref_out = layer_norm_func(x, weight, bias, eps=1e-6, kernel=ref_kernel)
        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw, weight.grad = weight.grad.detach().clone(), None
        ref_db, bias.grad = bias.grad.detach().clone(), None
        # opt
        x = x.detach().clone().requires_grad_()
        weight = weight.detach().clone().requires_grad_()
        bias = bias.detach().clone().requires_grad_()
        opt_out = layer_norm_func(x, weight, bias, eps=1e-6, kernel=real_kernel)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        opt_dx, x.grad = x.grad.detach().clone(), None
        opt_dw, weight.grad = weight.grad.detach().clone(), None
        opt_db, bias.grad = bias.grad.detach().clone(), None
        torch.testing.assert_close(ref_out, opt_out)
        torch.testing.assert_close(ref_dx, opt_dx)
        torch.testing.assert_close(ref_dw, opt_dw)
        torch.testing.assert_close(ref_db, opt_db)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore[56]
    @given(
        N=st.integers(min_value=32, max_value=10000),
        D=st.integers(min_value=32, max_value=512),
        is_swish=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=20,
    )
    # pyre-ignore[2]
    def test_modules(self, *args, **kwargs) -> None:
        self._test_layer_norm_module(
            *args,
            **kwargs,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_layer_norm_module(
        self,
        N: int,
        D: int,
        is_swish: bool,
        dtype: torch.dtype,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        x = (
            torch.empty((N, D), dtype=dtype, device=torch.device("cuda"))
            .normal_(0.0, 1.0)
            .requires_grad_()
        )
        # ref
        if is_swish:
            ref_layer = SwishLayerNorm(
                dim=D,
                eps=1e-6,
            ).to(device="cuda")
            ref_layer._hammer_kernel = ref_kernel
        else:
            ref_layer = LayerNorm(
                dim=D,
                eps=1e-6,
            ).to(device="cuda")
            ref_layer._hammer_kernel = ref_kernel
        opt_layer = copy.deepcopy(ref_layer)
        opt_layer._hammer_kernel = real_kernel

        ref_out = ref_layer(x)
        dout = torch.randn_like(ref_out) * 0.05
        ref_out.backward(dout)
        if skip_comparisons:
            return
        # pyre-ignore[16]
        ref_dx, x.grad = x.grad.detach().clone(), None
        ref_dw = ref_layer.weight.grad.detach().clone()
        ref_db = ref_layer.bias.grad.detach().clone()
        # opt
        x = x.detach().clone().requires_grad_()
        opt_out = opt_layer(x)
        dout = dout.detach().clone()
        opt_out.backward(dout)
        opt_dx, x.grad = x.grad.detach().clone(), None
        opt_dw = opt_layer.weight.grad.detach().clone()
        opt_db = opt_layer.bias.grad.detach().clone()
        torch.testing.assert_close(ref_out, opt_out)
        torch.testing.assert_close(
            ref_dx,
            opt_dx,
        )
        torch.testing.assert_close(
            ref_dw,
            opt_dw,
        )
        torch.testing.assert_close(
            ref_db,
            opt_db,
        )

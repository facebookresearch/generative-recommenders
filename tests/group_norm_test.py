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

#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch

from hypothesis import given, settings, strategies as st, Verbosity

from ops.triton.triton_group_norm import GroupNorm as TritonGroupNorm


class GroupNormTest(unittest.TestCase):
    @given(
        N=st.sampled_from([32]),
        D=st.sampled_from([32]),
        num_groups=st.sampled_from([2]),
        channel_per_group=st.sampled_from([2]),
        dtype=st.sampled_from([torch.float32]),
        affine=st.sampled_from([False, True]),
    )
    @settings(
        deadline=None,
        verbosity=Verbosity.verbose,
        max_examples=30,
    )
    # pyre-ignore[2]
    def test_group_norm_triton(
        self,
        N: int,
        D: int,
        num_groups: int,
        channel_per_group: int,
        dtype: torch.dtype,
        affine: bool,
    ) -> None:
        num_channels = num_groups * channel_per_group

        ref_layer = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=1e-5,
            affine=affine,
        ).to(device="cuda")
        real_layer = TritonGroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=1e-5,
            affine=affine,
        ).to(device="cuda")
        if affine:
            weight = (
                torch.empty((num_channels,), dtype=dtype, device=torch.device("cuda"))
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            bias = (
                torch.empty((num_channels,), dtype=dtype, device=torch.device("cuda"))
                .uniform_(-1.0, 1.0)
                .requires_grad_()
            )
            real_layer.weight.data.copy_(weight)
            real_layer.bias.data.copy_(bias)
            ref_layer.weight.data.copy_(weight)
            ref_layer.bias.data.copy_(bias)

        x = (
            torch.empty(
                (N, num_channels, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        # ref
        ref_out = ref_layer(x)
        # real
        x = x.detach().clone().requires_grad_()
        real_out = real_layer(x)
        torch.testing.assert_close(ref_out, real_out)


if __name__ == "__main__":
    unittest.main()

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

import torch

from generative_recommenders.common import gpu_unavailable, HammerKernel, set_dev_mode

from hypothesis import given, settings, strategies as st, Verbosity


class JaggedTensorsTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_split_2D_jagged_triton(self, *args, **kwargs) -> None:
        self._test_split_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_split_2D_jagged(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        is_dense_a: bool,
        is_dense_b: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import split_2D_jagged

        max_seq_len = max_len_a + max_len_b
        if not is_dense_a:
            lengths_a = torch.randint(
                1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_a = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
            total_len_a = int(offsets_a[-1].item())
        else:
            offsets_a = None
            total_len_a = batch_size * max_len_a
            is_dense_b = False
        if not is_dense_b:
            lengths_b = torch.randint(
                1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_b = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
            total_len_b = int(offsets_b[-1].item())
        else:
            offsets_b = None
            total_len_b = batch_size * max_len_b
        values = (
            torch.empty(
                (total_len_a + total_len_b, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values_a, ref_values_b = split_2D_jagged(
            max_seq_len=max_len_a + max_len_b,
            values=values,
            max_len_left=max_len_a if is_dense_a else None,
            max_len_right=max_len_b if is_dense_b else None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        d_values_a = torch.randn_like(ref_values_a)
        d_values_b = torch.randn_like(ref_values_b)
        ref_values_a.backward(d_values_a, retain_graph=True)
        ref_values_b.backward(d_values_b)
        if skip_comparisons:
            return

        assert values.grad is not None
        ref_d_values, values.grad = values.grad.clone(), None

        values = values.detach().clone().requires_grad_()
        real_values_a, real_values_b = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            max_len_left=max_len_a if is_dense_a else None,
            max_len_right=max_len_b if is_dense_b else None,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values_a, real_values_a)
        torch.testing.assert_close(ref_values_b, real_values_b)

        if test_backward:
            d_values_a = d_values_a.detach().clone()
            d_values_b = d_values_b.detach().clone()
            real_values_a.backward(d_values_a, retain_graph=True)
            real_values_b.backward(d_values_b)
            real_d_values = values.grad.clone()
            torch.testing.assert_close(ref_d_values, real_d_values)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_len_a=st.integers(20, 100),
        max_len_b=st.integers(20, 100),
        D=st.integers(10, 30),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_concat_2D_jagged_triton(self, *args, **kwargs) -> None:
        self._test_concat_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=HammerKernel.PYTORCH,
            real_kernel=HammerKernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.sampled_from([130]),
        max_len_a=st.sampled_from([32768]),
        max_len_b=st.sampled_from([10]),
        D=st.sampled_from([512]),
        is_dense_a=st.sampled_from([True, False]),
        is_dense_b=st.sampled_from([True, False]),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_concat_2D_jagged_large_tensor(self, *args, **kwargs) -> None:
        self._test_concat_2D_jagged(
            *args,
            **kwargs,
            test_backward=True,
            skip_comparisons=True,
            ref_kernel=HammerKernel.TRITON,
            real_kernel=HammerKernel.TRITON,
        )

    def _test_concat_2D_jagged(
        self,
        batch_size: int,
        max_len_a: int,
        max_len_b: int,
        D: int,
        is_dense_a: bool,
        is_dense_b: bool,
        test_backward: bool,
        ref_kernel: HammerKernel,
        real_kernel: HammerKernel,
        dtype: torch.dtype = torch.float32,
        skip_comparisons: bool = False,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import concat_2D_jagged

        if not is_dense_a:
            lengths_a = torch.randint(
                1, max_len_a + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_a = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_a[1:] = torch.cumsum(lengths_a, dim=0)
            total_len_a = int(offsets_a[-1].item())
        else:
            offsets_a = None
            total_len_a = batch_size * max_len_a
        values_a = (
            torch.empty(
                (total_len_a, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        if not is_dense_b:
            lengths_b = torch.randint(
                1, max_len_b + 1, size=(batch_size,), device=torch.device("cuda")
            )
            offsets_b = torch.zeros(
                (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
            )
            offsets_b[1:] = torch.cumsum(lengths_b, dim=0)
            total_len_b = int(offsets_b[-1].item())
        else:
            offsets_b = None
            total_len_b = batch_size * max_len_b
        values_b = (
            torch.empty(
                (total_len_b, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        ref_values = concat_2D_jagged(
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_values)
        ref_values.backward(dout)
        if skip_comparisons:
            return

        assert values_a.grad is not None
        ref_d_a, values_a.grad = values_a.grad.clone(), None
        assert values_b.grad is not None
        ref_d_b, values_b.grad = values_b.grad.clone(), None

        values_a = values_a.detach().clone().requires_grad_()
        values_b = values_b.detach().clone().requires_grad_()
        dout = dout.detach().clone()
        real_values = concat_2D_jagged(
            values_left=values_a,
            values_right=values_b,
            max_len_left=max_len_a,
            max_len_right=max_len_b,
            offsets_left=offsets_a,
            offsets_right=offsets_b,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_values, real_values)

        if test_backward:
            real_values.backward(dout)
            real_d_a = values_a.grad.clone()
            real_d_b = values_b.grad.clone()
            torch.testing.assert_close(ref_d_a, real_d_a)
            torch.testing.assert_close(ref_d_b, real_d_b)

    # pyre-ignore
    @given(
        batch_size=st.integers(2, 8),
        max_uih_len=st.integers(20, 100),
        max_l2_len=st.integers(10, 30),
        contextual_seq_len=st.sampled_from([0, 10]),
        max_targets=st.sampled_from([10, 20]),
        D=st.integers(10, 30),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_hstu_split_l2_embeddings(
        self,
        batch_size: int,
        max_uih_len: int,
        max_l2_len: int,
        contextual_seq_len: int,
        max_targets: int,
        D: int,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import hstu_split_l2_embeddings

        max_seq_len = max_uih_len + max_targets + contextual_seq_len
        num_targets = torch.randint(
            1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        x_lengths = torch.randint(
            0,
            max_uih_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        x_lengths = num_targets + x_lengths + contextual_seq_len
        x_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        x_offsets[1:] = torch.cumsum(x_lengths, dim=0)
        total_len = int(x_offsets[-1].item())
        x = (
            torch.empty(
                (total_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        minus_l2_lengths = x_lengths - max_l2_len - num_targets - contextual_seq_len
        minus_l2_lengths = torch.clamp(minus_l2_lengths, min=0)
        minus_l2_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            minus_l2_lengths
        )
        l2_offsets = x_offsets - minus_l2_offsets
        ref_minus_l2_x, ref_l2_x = hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            minus_l2_offsets=minus_l2_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.PYTORCH,
        )
        d_minus_l2_x = torch.randn_like(ref_minus_l2_x)
        d_l2_x = torch.randn_like(ref_l2_x)
        ref_minus_l2_x.backward(d_minus_l2_x, retain_graph=True)
        ref_l2_x.backward(d_l2_x)
        assert x.grad is not None
        ref_d_x, x.grad = x.grad.clone(), None
        x = x.detach().clone().requires_grad_()
        real_minus_l2_x, real_l2_x = hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            minus_l2_offsets=minus_l2_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.TRITON,
        )
        print(ref_minus_l2_x.shape, real_minus_l2_x.shape)
        torch.testing.assert_close(ref_minus_l2_x, real_minus_l2_x)
        torch.testing.assert_close(ref_l2_x, real_l2_x)
        d_minus_l2_x = d_minus_l2_x.detach().clone()
        d_l2_x = d_l2_x.detach().clone()
        real_minus_l2_x.backward(d_minus_l2_x, retain_graph=True)
        real_l2_x.backward(d_l2_x)
        real_d_x = x.grad.clone()
        torch.testing.assert_close(ref_d_x, real_d_x)

    # pyre-ignore
    @given(
        batch_size=st.integers(1, 1),
        max_minus_l2_len=st.integers(10, 10),
        max_l2_len=st.integers(5, 5),
        contextual_seq_len=st.sampled_from([3]),
        max_targets=st.sampled_from([2]),
        D=st.integers(10, 10),
        dtype=st.sampled_from(
            [torch.bfloat16, torch.float32]
            if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8
            else [torch.float32]
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_hstu_concat_l2_embeddings(
        self,
        batch_size: int,
        max_minus_l2_len: int,
        max_l2_len: int,
        contextual_seq_len: int,
        max_targets: int,
        D: int,
        dtype: torch.dtype,
    ) -> None:
        set_dev_mode(True)
        from generative_recommenders.ops.jagged_tensors import hstu_concat_l2_embeddings

        num_targets = torch.randint(
            1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        l2_lengths = torch.randint(
            0,
            max_l2_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        l2_lengths = num_targets + l2_lengths + contextual_seq_len
        max_l2_len = max_l2_len + contextual_seq_len + max_targets
        l2_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        l2_offsets[1:] = torch.cumsum(l2_lengths, dim=0)
        total_l2_len = int(l2_offsets[-1].item())
        l2_x = (
            torch.empty(
                (total_l2_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        minus_l2_lengths = torch.randint(
            0,
            max_minus_l2_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        minus_l2_lengths = torch.randint(
            0,
            max_minus_l2_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        minus_l2_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        minus_l2_offsets[1:] = torch.cumsum(minus_l2_lengths, dim=0)
        total_minus_l2_len = int(minus_l2_offsets[-1].item())
        minus_l2_x = (
            torch.empty(
                (total_minus_l2_len, D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        ref_x = hstu_concat_l2_embeddings(
            max_minus_l2_len=max_minus_l2_len,
            minus_l2_x=minus_l2_x,
            minus_l2_offsets=minus_l2_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.PYTORCH,
        )
        dout = torch.randn_like(ref_x)
        ref_x.backward(dout)

        assert minus_l2_x.grad is not None
        ref_d_minus_l2_x, minus_l2_x.grad = minus_l2_x.grad.clone(), None
        assert l2_x.grad is not None
        ref_d_l2_x, l2_x.grad = l2_x.grad.clone(), None

        minus_l2_x = minus_l2_x.detach().clone().requires_grad_()
        l2_x = l2_x.detach().clone().requires_grad_()
        real_x = hstu_concat_l2_embeddings(
            max_minus_l2_len=max_minus_l2_len,
            minus_l2_x=minus_l2_x,
            minus_l2_offsets=minus_l2_offsets,
            max_l2_len=max_l2_len,
            l2_x=l2_x,
            l2_offsets=l2_offsets,
            contextual_seq_len=contextual_seq_len,
            kernel=HammerKernel.TRITON,
        )
        torch.testing.assert_close(ref_x, real_x)
        dout = dout.detach().clone()
        real_x.backward(dout)
        real_d_minus_l2_x = minus_l2_x.grad.clone()
        real_d_l2_x = l2_x.grad.clone()
        torch.testing.assert_close(ref_d_minus_l2_x, real_d_minus_l2_x)
        torch.testing.assert_close(ref_d_l2_x, real_d_l2_x)

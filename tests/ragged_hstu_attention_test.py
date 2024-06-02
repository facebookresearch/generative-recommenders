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

import unittest
from typing import Tuple

import torch
import fbgemm_gpu

from hypothesis import given, settings, strategies as st, Verbosity


class RaggedHSTUAttentionTest(unittest.TestCase):
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([100, 128, 256, 1300]),
        max_targets=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([128]),
        hidden_dim=st.sampled_from([128]),
        num_buckets=st.sampled_from([180]),
        bucket_settings=st.sampled_from(
            [
                ("log", 0.301, 1),
            ]
        ),
        dtype=st.sampled_from([torch.float32]),
        weights_dtype=st.sampled_from([torch.float32]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=10,
        deadline=None,
    )
    def test_fused_attn_triton(
        self,
        batch_size: int,
        heads: int,
        max_uih_len: int,
        max_targets: int,
        attn_dim: int,
        hidden_dim: int,
        num_buckets: int,
        bucket_settings: Tuple[str, float, float],
        dtype: torch.dtype,
        weights_dtype: torch.dtype,
    ) -> None:
        from ops.triton.triton_ragged_hstu_attention import triton_ragged_attention_relative_bias
        from modeling.sequential.hstu import _hstu_attention_maybe_from_cache, RelativeBucketedTimeAndPositionBasedBias

        alpha = 1.0 #  / (attn_dim**0.5)
        lengths = torch.randint(
            max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
        )
        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = lengths + num_targets
        max_seq_len = max_uih_len + max_targets
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)

        L = int(seq_offsets[-1].item())
        q = (
            torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        k = (
            torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        v = (
            torch.empty(
                (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        time_delta = 0.1
        timestamp_deltas: torch.Tensor = torch.randint(
            86400,
            size=(batch_size, max_seq_len),
            device="cuda",
        )
        timestamps = timestamp_deltas.cumsum(dim=1)
        timestamps_triton = torch.cat([timestamps, timestamps[:, max_seq_len-1:]], dim=1)

        ts_weights: torch.Tensor = (
            torch.empty(
                (num_buckets + 1,),
                device="cuda",
                dtype=weights_dtype,
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        pos_weights_size = 2 * max_seq_len - 1
        pos_weights: torch.Tensor = (
            torch.empty(
                (pos_weights_size,),
                device="cuda",
                dtype=weights_dtype,
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )

        # triton implementation
        triton_out = triton_ragged_attention_relative_bias(
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            timestamps=timestamps_triton,
            ts_weights=ts_weights,
            pos_weights=pos_weights,
            causal=True,
            num_buckets=num_buckets,
            time_bucket_fn=bucket_settings[0],
            time_bucket_incr=bucket_settings[1],
            time_bucket_div=bucket_settings[2],
            time_delta=time_delta,
            max_pos_ind=None,
            num_targets=None,
            attn_scale=None,
            relative_bias_type="ALL",
        )
        # torch implementation
        relative_bias = RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=max_seq_len,
            num_buckets=num_buckets,
            bucketization_fn=lambda x: (torch.log((x + time_delta).clamp(min=1e-6) / bucket_settings[1]) / bucket_settings[2]).clamp(min=0).long(),
        )
        relative_bias._ts_w = torch.nn.Parameter(ts_weights)
        relative_bias._pos_w = torch.nn.Parameter(pos_weights)
        attn_mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), dtype=torch.bool),
            diagonal=1,
        ).cuda()
        torch_out, _, _ = _hstu_attention_maybe_from_cache(
            num_heads=heads,
            attention_dim=attn_dim,
            linear_dim=hidden_dim,
            q=q.view(L, -1),
            k=k.view(L, -1),
            v=v.view(L, -1),
            cached_q=None,
            cached_k=None,
            delta_x_offsets=None,
            x_offsets=seq_offsets,
            all_timestamps=timestamps,
            invalid_attn_mask=1.0 - attn_mask.to(torch.float32),
            rel_attn_bias=relative_bias,
        )
        torch.testing.assert_close(
            triton_out.view(L, -1),
            torch_out,
        )

if __name__ == "__main__":
    unittest.main()

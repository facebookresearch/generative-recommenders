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
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.modules.content_encoder import ContentEncoder


class ContentEncoderTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_forward(self) -> None:
        device = torch.device("cuda")
        input_embedding_dim = 32
        encoder = ContentEncoder(
            input_embedding_dim=input_embedding_dim,
            additional_content_feature_names=["a0", "a1"],
            target_enrich_feature_names=["t0", "t1"],
            is_inference=False,
        ).to(device)
        seq_lengths = [6, 3]
        seq_offsets = [0, 6, 9]
        num_targets = [2, 1]
        seq_embeddings = torch.rand(
            (sum(seq_lengths), input_embedding_dim), device=device
        ).requires_grad_(True)
        seq_payloads = {
            "a0": torch.rand_like(seq_embeddings).requires_grad_(True),
            "a1": torch.rand_like(seq_embeddings).requires_grad_(True),
            "t0": torch.rand(
                sum(num_targets), input_embedding_dim, device=device
            ).requires_grad_(True),
            "t1": torch.rand(
                sum(num_targets), input_embedding_dim, device=device
            ).requires_grad_(True),
        }
        content_embeddings = encoder(
            max_seq_len=max(seq_lengths),
            seq_embeddings=seq_embeddings,
            seq_lengths=torch.tensor(seq_lengths, device=device),
            seq_offsets=torch.tensor(seq_offsets, device=device),
            seq_payloads=seq_payloads,
            num_targets=torch.tensor(num_targets, device=device),
        )
        content_embeddings.sum().backward()

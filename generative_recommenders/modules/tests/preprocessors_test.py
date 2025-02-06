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
from generative_recommenders.modules.preprocessors import ContextualPreprocessor


class PreprocessorTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_forward(self) -> None:
        device = torch.device("cuda")
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=64,
            output_embedding_dim=128,
            contextual_feature_to_max_length={"c_0": 1, "c_1": 2},
            contextual_feature_to_min_uih_length={"c_1": 4},
            is_inference=True,
        ).to(device)
        seq_embeddings = torch.rand((9, 64), device=device)
        seq_timestamps = torch.tensor(
            [1, 2, 3, 4, 5, 6, 10, 20, 30],
            device=device,
        )

        (
            output_max_seq_len,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            _,
        ) = preprocessor(
            max_seq_len=6,
            seq_lengths=torch.tensor([6, 3], device=device),
            seq_timestamps=seq_timestamps,
            seq_embeddings=seq_embeddings,
            seq_payloads={
                "c_0": torch.rand((9, 64), device=device),
                "c_0_offsets": torch.tensor([0, 6, 9], device=device),
                "c_1": torch.rand((9, 64), device=device),
                "c_1_offsets": torch.tensor([0, 6, 9], device=device),
            },
            num_targets=torch.tensor([2, 1], device=device),
        )
        self.assertEqual(output_max_seq_len, 9)
        self.assertEqual(output_seq_lengths.tolist(), [9, 6])
        self.assertEqual(output_seq_offsets.tolist(), [0, 9, 15])
        self.assertEqual(output_num_targets.tolist(), [2, 1])
        self.assertEqual(
            output_seq_embeddings.size(),
            (
                15,
                128,
            ),
        )
        self.assertEqual(
            output_seq_timestamps.size(),
            (15,),
        )
        self.assertEqual(
            output_seq_timestamps.tolist(),
            [0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 10, 20, 30],
        )

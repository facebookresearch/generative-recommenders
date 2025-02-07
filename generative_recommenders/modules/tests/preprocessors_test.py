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
from typing import Tuple

import torch
from generative_recommenders.common import gpu_unavailable
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from hypothesis import given, settings, strategies as st, Verbosity


class PreprocessorTest(unittest.TestCase):
    # pyre-ignore
    @given(
        interleave_options=st.sampled_from(
            [(True, True), (False, True), (False, False)]
        ),
        is_inference=st.booleans(),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=50, deadline=None)
    def test_forward(
        self,
        interleave_options: Tuple[bool, bool],
        is_inference: bool,
    ) -> None:
        device = torch.device("cuda")
        interleave_action_with_target, interleave_action_with_uih = interleave_options
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=64,
            output_embedding_dim=128,
            contextual_feature_to_max_length={"c_0": 1, "c_1": 2},
            contextual_feature_to_min_uih_length={"c_1": 4},
            uih_weight_name="w",
            action_weights=[1, 2, 4],
            is_inference=is_inference,
            interleave_action_with_target=interleave_action_with_target,
            interleave_action_with_uih=interleave_action_with_uih,
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
                "c_0": torch.rand((2, 64), device=device),
                "c_0_offsets": torch.tensor([0, 1, 2], device=device),
                "c_1": torch.rand((4, 64), device=device),
                "c_1_offsets": torch.tensor([0, 2, 4], device=device),
                "w": torch.tensor([1, 2, 4, 1, 2, 4, 1, 2, 4], device=device),
            },
            num_targets=torch.tensor([2, 1], device=device),
        )
        if interleave_action_with_uih:
            if interleave_action_with_target and not is_inference:
                expected_max_seq_len = 15
                expected_output_seq_lengths = [15, 9]
                expected_output_seq_offsets = [0, 15, 24]
                expected_output_num_targets = [4, 2]
                expected_seq_embedding_size = (24, 128)
                expected_seq_timestamps_size = (24,)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    5,
                    6,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                    30,
                ]
            else:
                expected_max_seq_len = 13
                expected_output_seq_lengths = [13, 8]
                expected_output_seq_offsets = [0, 13, 21]
                expected_output_num_targets = [2, 1]
                expected_seq_embedding_size = (21, 128)
                expected_seq_timestamps_size = (21,)
                expected_output_seq_timestamps = [
                    0,
                    0,
                    0,
                    1,
                    1,
                    2,
                    2,
                    3,
                    3,
                    4,
                    4,
                    5,
                    6,
                    0,
                    0,
                    0,
                    10,
                    10,
                    20,
                    20,
                    30,
                ]
        else:
            expected_max_seq_len = 9
            expected_output_seq_lengths = [9, 6]
            expected_output_seq_offsets = [0, 9, 15]
            expected_output_num_targets = [2, 1]
            expected_seq_embedding_size = (15, 128)
            expected_seq_timestamps_size = (15,)
            expected_output_seq_timestamps = [
                0,
                0,
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                0,
                0,
                0,
                10,
                20,
                30,
            ]

        self.assertEqual(output_max_seq_len, expected_max_seq_len)
        self.assertEqual(output_seq_lengths.tolist(), expected_output_seq_lengths)
        self.assertEqual(output_seq_offsets.tolist(), expected_output_seq_offsets)
        self.assertEqual(output_num_targets.tolist(), expected_output_num_targets)
        self.assertEqual(
            output_seq_embeddings.size(),
            expected_seq_embedding_size,
        )
        self.assertEqual(
            output_seq_timestamps.size(),
            expected_seq_timestamps_size,
        )
        self.assertEqual(
            output_seq_timestamps.tolist(),
            expected_output_seq_timestamps,
        )

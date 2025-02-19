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
from generative_recommenders.modules.action_encoder import ActionEncoder


class ActionEncoderTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    def test_forward(self) -> None:
        device = torch.device("cuda")
        action_embedding_dim = 32
        action_weights = [1, 2, 4, 8, 16]
        watchtime_to_action_thresholds_and_weights = [
            (30, 32),
            (60, 64),
            (100, 128),
        ]
        num_action_types = len(action_weights) + len(
            watchtime_to_action_thresholds_and_weights
        )
        combined_action_weights = action_weights + [
            x[1] for x in watchtime_to_action_thresholds_and_weights
        ]
        enabled_actions = [
            [0],
            [0, 1],
            [1, 3, 4],
            [1, 2, 3, 4],
            [1, 2],
            [2],
            [3],
            [1, 2],
            [3],
        ]
        watchtimes = [40, 20, 110, 31, 26, 55, 33, 71, 66]
        for i, wt in enumerate(watchtimes):
            for j, w in enumerate(watchtime_to_action_thresholds_and_weights):
                if wt > w[0]:
                    enabled_actions[i].append(j + len(action_weights))
        actions = [
            sum([combined_action_weights[t] for t in x]) for x in enabled_actions
        ]

        encoder = ActionEncoder(
            watchtime_feature_name="watchtimes",
            action_feature_name="actions",
            action_weights=action_weights,
            watchtime_to_action_thresholds_and_weights=watchtime_to_action_thresholds_and_weights,
            action_embedding_dim=action_embedding_dim,
            is_inference=False,
        ).to(device)

        seq_lengths = [6, 3]
        seq_offsets = [0, 6, 9]
        num_targets = [2, 1]
        action_embeddings = encoder(
            max_seq_len=max(seq_lengths),
            seq_lengths=torch.tensor(seq_lengths, device=device),
            seq_offsets=torch.tensor(seq_offsets, device=device),
            seq_payloads={
                "watchtimes": torch.tensor(watchtimes, device=device),
                "actions": torch.tensor(actions, device=device),
            },
            num_targets=torch.tensor(num_targets, device=device),
        )
        self.assertEqual(
            action_embeddings.shape, (9, action_embedding_dim * num_action_types)
        )
        for b in range(len(seq_lengths)):
            b_start = seq_offsets[b]
            b_end = seq_offsets[b + 1]
            for j in range(b_start, b_end):
                embedding = action_embeddings[j].view(num_action_types, -1)
                for atype in range(num_action_types):
                    if b_end - j <= num_targets[b]:
                        torch.testing.assert_allclose(
                            embedding[atype],
                            encoder._target_action_embedding_table[atype],
                        )
                    else:
                        if atype in enabled_actions[j]:
                            torch.testing.assert_allclose(
                                embedding[atype],
                                encoder._action_embedding_table[atype],
                            )
                        else:
                            torch.testing.assert_allclose(
                                embedding[atype], torch.zeros_like(embedding[atype])
                            )
        action_embeddings.sum().backward()

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

from typing import Dict, List, Optional, Tuple

import torch

from generative_recommenders.common import (
    dense_to_jagged,
    HammerModule,
    jagged_to_padded_dense,
)


class ActionEncoder(HammerModule):
    def __init__(
        self,
        action_embedding_dim: int,
        action_feature_name: str,
        action_weights: List[int],
        watchtime_feature_name: str = "",
        watchtime_to_action_thresholds_and_weights: Optional[
            List[Tuple[int, int]]
        ] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._watchtime_feature_name: str = watchtime_feature_name
        self._action_feature_name: str = action_feature_name
        self._watchtime_to_action_thresholds_and_weights: List[Tuple[int, int]] = (
            watchtime_to_action_thresholds_and_weights
            if watchtime_to_action_thresholds_and_weights is not None
            else []
        )
        self.register_buffer(
            "_combined_action_weights",
            torch.tensor(
                action_weights
                + [x[1] for x in self._watchtime_to_action_thresholds_and_weights]
            ),
        )
        self._num_action_types: int = len(action_weights) + len(
            self._watchtime_to_action_thresholds_and_weights
        )
        self._action_embedding_dim = action_embedding_dim
        self._action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )
        self._target_action_embedding_table: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty((self._num_action_types, action_embedding_dim)).normal_(
                mean=0, std=0.1
            ),
        )

    @property
    def output_embedding_dim(self) -> int:
        return self._action_embedding_dim * self._num_action_types

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        num_targets: torch.Tensor,
    ) -> torch.Tensor:
        seq_actions = seq_payloads[self._action_feature_name]
        if len(self._watchtime_to_action_thresholds_and_weights) > 0:
            watchtimes = seq_payloads[self._watchtime_feature_name]
            for threshold, weight in self._watchtime_to_action_thresholds_and_weights:
                seq_actions = torch.bitwise_or(
                    seq_actions, (watchtimes >= threshold).to(torch.int64) * weight
                )
        exploded_actions = (
            torch.bitwise_and(
                seq_actions.unsqueeze(-1), self._combined_action_weights.unsqueeze(0)
            )
            > 0
        )
        action_embeddings = (
            exploded_actions.unsqueeze(-1) * self._action_embedding_table.unsqueeze(0)
        ).view(-1, self._num_action_types * self._action_embedding_dim)

        padded_action_embeddings = jagged_to_padded_dense(
            values=action_embeddings,
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        mask = torch.arange(max_seq_len, device=seq_offsets.device).view(1, max_seq_len)
        mask = torch.logical_and(
            mask >= (seq_lengths - num_targets).unsqueeze(1),
            mask < seq_lengths.unsqueeze(1),
        )
        padded_action_embeddings[mask] = self._target_action_embedding_table.view(
            1, -1
        ).tile(
            int(torch.sum(num_targets).item()),
            1,
        )
        action_embeddings = dense_to_jagged(
            dense=padded_action_embeddings,
            x_offsets=[seq_offsets],
        )
        return action_embeddings

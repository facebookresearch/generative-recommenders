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

from math import sqrt
from typing import Optional

import torch
from generative_recommenders.common import HammerModule
from generative_recommenders.ops.position import add_timestamp_positional_embeddings


class HSTUPositionalEncoder(HammerModule):
    def __init__(
        self,
        num_position_buckets: int,
        num_time_buckets: int,
        embedding_dim: int,
        contextual_seq_len: int,
        is_inference: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._embedding_dim: int = embedding_dim
        self._contextual_seq_len: int = contextual_seq_len
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                -sqrt(1.0 / num_time_buckets),
                sqrt(1.0 / num_time_buckets),
            ),
        )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        seq_embeddings = add_timestamp_positional_embeddings(
            alpha=self._embedding_dim**0.5,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=self._contextual_seq_len,
            position_embeddings_weight=self._position_embeddings_weight,
            timestamp_embeddings_weight=self._timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=seq_lengths,
            seq_embeddings=seq_embeddings,
            timestamps=seq_timestamps,
            num_targets=num_targets,
            interleave_targets=False,
            kernel=self.hammer_kernel(),
        )
        return seq_embeddings

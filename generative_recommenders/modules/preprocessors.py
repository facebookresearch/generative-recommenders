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

import abc
from math import sqrt
from typing import Dict, List, Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_infer_max_len,
    fx_torch_ones,
    HammerModule,
    jagged_to_padded_dense,
)
from generative_recommenders.modules.utils import init_mlp_weights_optional_bias
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm


class InputPreprocessor(HammerModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """
        Args:
            max_seq_len: int
            seq_lengths: (B,)
            seq_embeddings: (L, D)
            seq_timestamps: (B, N)
            num_targets: (B,) Optional.
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            (max_seq_len, lengths, offsets, timestamps, embeddings, num_targets, payloads) updated based on input preprocessor.
        """
        pass


class ContextualPreprocessor(InputPreprocessor):
    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        uih_weight_name: str,
        action_weights: Optional[List[int]] = None,
        action_embedding_dim: int = 8,
        is_inference: bool = True,
        interleave_action_with_target: bool = False,
        interleave_action_with_uih: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._output_embedding_dim: int = output_embedding_dim
        self._input_embedding_dim = input_embedding_dim
        self._interleave_action_with_target = interleave_action_with_target
        self._interleave_action_with_uih = interleave_action_with_uih
        if self._interleave_action_with_target:
            assert (
                self._interleave_action_with_uih
            ), "interleave_action_with_target requires interleave_action_with_uih"

        hidden_dim = 256
        self._content_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self._input_embedding_dim,
                out_features=hidden_dim,
            ),
            SwishLayerNorm(hidden_dim),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=self._output_embedding_dim,
            ),
            LayerNorm(self._output_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

        self._uih_weight_name = uih_weight_name
        self._action_weights = action_weights
        if self._action_weights is not None:
            self._action_embedding_dim: int = action_embedding_dim
            self._num_actions: int = len(self._action_weights)
            self._action_embeddings: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty((self._num_actions, action_embedding_dim)).normal_(
                    mean=0, std=0.1
                ),
            )
            if not self._interleave_action_with_uih:
                self._candidate_action_embedding: torch.nn.Parameter = (
                    torch.nn.Parameter(
                        torch.empty(
                            (1, self._num_actions * self._action_embedding_dim)
                        ).normal_(mean=0, std=0.1),
                    )
                )
            self._action_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
                torch.nn.Linear(
                    in_features=self._action_embedding_dim * self._num_actions,
                    out_features=hidden_dim,
                ),
                SwishLayerNorm(hidden_dim),
                torch.nn.Linear(
                    in_features=hidden_dim,
                    out_features=self._output_embedding_dim,
                ),
                LayerNorm(self._output_embedding_dim),
            ).apply(init_mlp_weights_optional_bias)

        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        if self._max_contextual_seq_len > 0:
            std = 1.0 * sqrt(
                2.0 / float(input_embedding_dim + self._output_embedding_dim)
            )
            self._batched_contextual_linear_weights: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (
                            self._max_contextual_seq_len,
                            input_embedding_dim,
                            self._output_embedding_dim,
                        )
                    ).normal_(0.0, std)
                )
            )
            self._batched_contextual_linear_bias: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (self._max_contextual_seq_len, self._output_embedding_dim)
                    ).fill_(0.0)
                )
            )

    def _mask_candidate_action(
        self,
        action_embeddings: torch.Tensor,
        seq_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        max_seq_len: int,
        candidate_action_embedding: torch.Tensor,
    ) -> torch.Tensor:
        B = seq_lengths.size(0)
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        padded_action_embeddings = jagged_to_padded_dense(
            action_embeddings, [seq_offsets], [max_seq_len], padding_value=0.0
        )
        raw_mask = torch.arange(max_seq_len, device=seq_lengths.device).expand(
            B, max_seq_len
        )

        candidate_mask = torch.logical_and(
            raw_mask >= (seq_lengths - num_targets).unsqueeze(1),
            raw_mask < seq_lengths.unsqueeze(1),
        )
        padded_action_embeddings[candidate_mask] = candidate_action_embedding
        action_embeddings = torch.ops.fbgemm.dense_to_jagged(
            padded_action_embeddings,
            [seq_offsets],
        )[0]
        return action_embeddings

    def _get_action_embeddings(
        self,
        actions: torch.Tensor,
        seq_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        max_seq_len: int,
    ) -> torch.Tensor:
        mask = fx_torch_ones(
            [actions.shape[0], self._num_actions],
            device=actions.device,
            dtype=torch.int32,
        )
        mask *= torch.tensor(self._action_weights, device=actions.device)
        decoded_actions = torch.bitwise_and(actions.unsqueeze(-1), mask) > 0
        decoded_actions = decoded_actions.to(self._action_embeddings.dtype)
        action_embeddings = (
            decoded_actions.unsqueeze(-1) * self._action_embeddings.unsqueeze(0)
        ).view(-1, self._num_actions * self._action_embedding_dim)

        if not self._interleave_action_with_uih:
            action_embeddings = self._mask_candidate_action(
                action_embeddings=action_embeddings,
                seq_lengths=seq_lengths,
                num_targets=num_targets,
                max_seq_len=max_seq_len,
                candidate_action_embedding=self._candidate_action_embedding,
            )

        action_embeddings = self._action_embedding_mlp(action_embeddings)
        return action_embeddings

    def _remove_actions_from_target(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_lengths: torch.Tensor,
        num_targets: torch.Tensor,
        B: int,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = (
            torch.arange(max_seq_len * 2, device=seq_embeddings.device)
            .unsqueeze(0)
            .expand(B, -1)
        )
        valid_mask = torch.logical_and(
            mask < 2 * seq_lengths.unsqueeze(1),
            torch.logical_or(
                mask < 2 * (seq_lengths - num_targets).unsqueeze(1),
                torch.remainder(mask, 2) == 0,
            ),
        )
        valid_mask_jagged = (
            torch.ops.fbgemm.dense_to_jagged(
                valid_mask.int().unsqueeze(2),
                [torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths * 2)],
            )[0]
            .to(torch.bool)
            .squeeze(1)
        )
        seq_embeddings = seq_embeddings[valid_mask_jagged]
        seq_timestamps = seq_timestamps[valid_mask_jagged]

        return seq_embeddings, seq_timestamps

    def _add_action_embeddings(
        self,
        seq_embeddings: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        max_seq_len: int,
        num_targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        action_embeddings = self._get_action_embeddings(
            seq_payloads[self._uih_weight_name],
            seq_lengths,
            num_targets,
            max_seq_len,
        )
        if self._interleave_action_with_uih:
            B: int = seq_lengths.size()[0]
            seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
            dense_timestamps = torch.ops.fbgemm.jagged_to_padded_dense(
                values=seq_timestamps,
                offsets=[seq_offsets],
                max_lengths=[max_seq_len],
                padding_value=0.0,
            )
            dense_timestamps = (
                dense_timestamps.unsqueeze(2).expand(-1, -1, 2).reshape(B, -1)
            )
            seq_timestamps = torch.ops.fbgemm.dense_to_jagged(
                dense_timestamps.unsqueeze(-1),
                [torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths * 2)],
            )[0].squeeze(-1)

            seq_embeddings = torch.stack(
                [seq_embeddings, action_embeddings], dim=1
            ).reshape((-1, self._output_embedding_dim))

            if self.interleave_action_with_target:
                seq_lengths = seq_lengths * 2
                max_seq_len = max_seq_len * 2
                num_targets = num_targets * 2
            else:
                seq_embeddings, seq_timestamps = self._remove_actions_from_target(
                    seq_embeddings,
                    seq_timestamps,
                    seq_lengths,
                    num_targets,
                    B,
                    max_seq_len,
                )
                seq_lengths = seq_lengths * 2 - num_targets
                max_seq_len = fx_infer_max_len(seq_lengths)
                num_targets = num_targets

        else:
            seq_embeddings = seq_embeddings + action_embeddings
        return seq_embeddings, seq_lengths, seq_timestamps, max_seq_len, num_targets

    def _get_contextual_input_embeddings(
        self,
        seq_lengths: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        padded_values: List[torch.Tensor] = []
        for key, max_len in contextual_feature_to_max_length.items():
            v = torch.flatten(
                jagged_to_padded_dense(
                    values=seq_payloads[key].to(dtype),
                    offsets=[seq_payloads[key + "_offsets"]],
                    max_lengths=[max_len],
                    padding_value=0.0,
                ),
                1,
                2,
            )
            min_uih_length = contextual_feature_to_min_uih_length.get(key, 0)
            if min_uih_length > 0:
                v = v * (seq_lengths.view(-1, 1) >= min_uih_length)
            padded_values.append(v)
        return torch.cat(padded_values, dim=1)

    def _add_contextual_embeddings(
        self,
        seq_embeddings: torch.Tensor,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        contextual_embeddings = self._get_contextual_input_embeddings(
            seq_lengths=seq_lengths,
            seq_payloads=seq_payloads,
            contextual_feature_to_max_length=self._contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
            dtype=seq_embeddings.dtype,
        )
        contextual_embeddings = torch.baddbmm(
            self._batched_contextual_linear_bias.view(
                -1, 1, self._output_embedding_dim
            ).to(contextual_embeddings.dtype),
            contextual_embeddings.view(
                -1, self._max_contextual_seq_len, self._input_embedding_dim
            ).transpose(0, 1),
            self._batched_contextual_linear_weights.to(contextual_embeddings.dtype),
        ).transpose(0, 1)

        output_seq_embeddings = concat_2D_jagged(
            values_left=contextual_embeddings.reshape(-1, self._output_embedding_dim),
            values_right=seq_embeddings,
            max_len_left=self._max_contextual_seq_len,
            max_len_right=max_seq_len,
            offsets_left=None,
            offsets_right=seq_offsets,
        )
        output_seq_timestamps = concat_2D_jagged(
            values_left=torch.zeros(
                (seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                dtype=seq_timestamps.dtype,
                device=seq_timestamps.device,
            ),
            values_right=seq_timestamps.unsqueeze(-1),
            max_len_left=self._max_contextual_seq_len,
            max_len_right=max_seq_len,
            offsets_left=None,
            offsets_right=seq_offsets,
        ).squeeze(-1)
        output_max_seq_len = max_seq_len + self._max_contextual_seq_len
        output_seq_lengths = seq_lengths + self._max_contextual_seq_len

        return (
            output_seq_embeddings,
            output_seq_lengths,
            output_seq_timestamps,
            output_max_seq_len,
        )

    def forward(  # noqa C901
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        seq_embeddings = self._content_embedding_mlp(seq_embeddings)
        if self._action_weights is not None:
            seq_embeddings, seq_lengths, seq_timestamps, max_seq_len, num_targets = (
                self._add_action_embeddings(
                    seq_embeddings=seq_embeddings,
                    seq_lengths=seq_lengths,
                    seq_timestamps=seq_timestamps,
                    seq_payloads=seq_payloads,
                    max_seq_len=max_seq_len,
                    num_targets=num_targets,
                )
            )

        if self._max_contextual_seq_len > 0:
            (
                seq_embeddings,
                seq_lengths,
                seq_timestamps,
                max_seq_len,
            ) = self._add_contextual_embeddings(
                seq_embeddings=seq_embeddings,
                seq_lengths=seq_lengths,
                seq_timestamps=seq_timestamps,
                seq_payloads=seq_payloads,
                max_seq_len=max_seq_len,
            )

        return (
            max_seq_len,
            seq_lengths,
            torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths),
            seq_timestamps,
            seq_embeddings,
            num_targets,
            seq_payloads,
        )

    @property
    def interleave_action_with_target(self) -> bool:
        return self._interleave_action_with_target and self.is_train

    @property
    def interleave_action_with_uih(self) -> bool:
        return self._interleave_action_with_uih

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
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from generative_recommenders.common import HammerModule


class MultitaskTaskType(IntEnum):
    BINARY_CLASSIFICATION = 0
    REGRESSION = 1


@dataclass
class TaskConfig:
    task_name: str
    task_weight: int
    task_type: MultitaskTaskType


class MultitaskModule(HammerModule):
    @abc.abstractmethod
    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        payload_features: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        pass


def _compute_pred_and_logits(
    prediction_module: torch.nn.Module,
    encoded_user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    task_offsets: List[int],
    has_multiple_task_types: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    candidates_logits = prediction_module(
        encoded_user_embeddings * item_embeddings
    ).transpose(0, 1)
    candidates_preds_list: List[torch.Tensor] = []
    for task_type in MultitaskTaskType:
        if task_offsets[task_type + 1] - task_offsets[task_type] > 0:
            if task_type == MultitaskTaskType.REGRESSION:
                candidates_preds_list.append(
                    candidates_logits[
                        task_offsets[task_type] : task_offsets[task_type + 1],
                        :,
                    ]
                )
            else:
                candidates_preds_list.append(
                    F.sigmoid(
                        candidates_logits[
                            task_offsets[task_type] : task_offsets[task_type + 1],
                            :,
                        ]
                    )
                )
    if has_multiple_task_types:
        candidates_preds: torch.Tensor = torch.concat(candidates_preds_list, dim=0)
    else:
        candidates_preds: torch.Tensor = candidates_preds_list[0]

    return candidates_preds, candidates_logits


def _compute_labels_and_weights(
    supervision_bitmasks: torch.Tensor,
    watchtime_sequence: torch.Tensor,
    task_configs: List[TaskConfig],
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    default_supervision_mask = torch.ones_like(
        supervision_bitmasks,
        dtype=dtype,
        device=supervision_bitmasks.device,
    )
    candidates_lables_list: List[torch.Tensor] = []
    candidates_weights_list: List[torch.Tensor] = []
    for task in task_configs:
        if task.task_type == MultitaskTaskType.REGRESSION:
            candidates_lables_list.append(watchtime_sequence.unsqueeze(0))
            candidates_weights_list.append(default_supervision_mask.unsqueeze(0))
        else:
            candidates_lables_list.append(
                (torch.bitwise_and(supervision_bitmasks, task.task_weight) > 0)
                .to(dtype)
                .unsqueeze(0)
            )
            candidates_weights_list.append(default_supervision_mask.unsqueeze(0))
    if len(task_configs) > 1:
        candidates_labels = torch.concat(candidates_lables_list, dim=0)
        candidates_weights = torch.concat(candidates_weights_list, dim=0)
    else:
        candidates_labels = candidates_lables_list[0]
        candidates_weights = candidates_weights_list[0]

    return candidates_labels, candidates_weights


def _compute_loss(
    task_offsets: List[int],
    causal_multitask_weights: float,
    candidates_logits: torch.Tensor,
    candidates_labels: torch.Tensor,
    candidates_weights: torch.Tensor,
    has_multiple_task_types: bool,
) -> torch.Tensor:
    candidates_losses_list: List[torch.Tensor] = []
    for task_type in MultitaskTaskType:
        if task_offsets[task_type + 1] - task_offsets[task_type] > 0:
            if task_type == MultitaskTaskType.REGRESSION:
                candidates_losses_list.append(
                    F.mse_loss(
                        candidates_logits[
                            task_offsets[task_type] : task_offsets[task_type + 1],
                            :,
                        ],
                        candidates_labels[
                            task_offsets[task_type] : task_offsets[task_type + 1],
                            :,
                        ],
                        reduction="none",
                    )
                    * candidates_weights[
                        task_offsets[task_type] : task_offsets[task_type + 1],
                        :,
                    ]
                )
            else:
                candidates_losses_list.append(
                    F.binary_cross_entropy_with_logits(
                        input=candidates_logits[
                            task_offsets[task_type] : task_offsets[task_type + 1],
                            :,
                        ],
                        target=candidates_labels[
                            task_offsets[task_type] : task_offsets[task_type + 1],
                            :,
                        ],
                        reduction="none",
                    )
                    * candidates_weights[
                        task_offsets[task_type] : task_offsets[task_type + 1],
                        :,
                    ]
                )

    if has_multiple_task_types:
        candidates_losses = torch.concat(candidates_losses_list, dim=0)
    else:
        candidates_losses = candidates_losses_list[0]
    candidates_losses = (
        candidates_losses.sum(-1)
        / candidates_weights.sum(-1).clamp(min=1.0)
        * causal_multitask_weights
    )
    return candidates_losses


class DefaultMultitaskModule(MultitaskModule):
    def __init__(
        self,
        task_configs: List[TaskConfig],
        embedding_dim: int,
        prediction_fn: Callable[[int, int], torch.nn.Module],
        candidates_weight_feature_name: str,
        candidates_watchtime_feature_name: str,
        causal_multitask_weights: float,
        is_inference: bool,
    ) -> None:
        super().__init__(is_inference)
        self._prediction_module: torch.nn.Module = prediction_fn(
            embedding_dim, len(task_configs)
        )
        self._task_configs: List[TaskConfig] = task_configs
        self._task_offsets: List[int] = [0] * (len(MultitaskTaskType) + 1)
        assert len(task_configs) > 0
        for task in self._task_configs:
            self._task_offsets[task.task_type + 1] += 1
        self._has_multiple_task_types: bool = self._task_offsets.count(0) < len(
            MultitaskTaskType
        )
        self._task_offsets[1:] = np.cumsum(self._task_offsets[1:]).tolist()
        self._candidates_weight_feature_name: str = candidates_weight_feature_name
        self._candidates_watchtime_feature_name: str = candidates_watchtime_feature_name
        self._causal_multitask_weights: float = causal_multitask_weights

    def forward(
        self,
        encoded_user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        payload_features: Dict[str, torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """
        Computes multi-task predictions.

        Args:
            encoded_user_embeddings: (L, D) x float.
            item_embeddings: (L, D) x float.

        Returns:
            (T, L) x float, predictions, labels, weights, losses
        """

        candidates_preds, candidates_logits = _compute_pred_and_logits(
            prediction_module=self._prediction_module,
            encoded_user_embeddings=encoded_user_embeddings,
            item_embeddings=item_embeddings,
            task_offsets=self._task_offsets,
            has_multiple_task_types=self._has_multiple_task_types,
        )
        candidates_labels: Optional[torch.Tensor] = None
        candidates_weights: Optional[torch.Tensor] = None
        candidates_losses: Optional[torch.Tensor] = None
        if not self._is_inference:
            candidates_labels, candidates_weights = _compute_labels_and_weights(
                supervision_bitmasks=payload_features[
                    self._candidates_weight_feature_name
                ].to(torch.int64),
                watchtime_sequence=payload_features[
                    self._candidates_watchtime_feature_name
                ],
                task_configs=self._task_configs,
                dtype=encoded_user_embeddings.dtype,
            )
            if self.training:
                candidates_losses = _compute_loss(
                    self._task_offsets,
                    self._causal_multitask_weights,
                    candidates_logits,
                    candidates_labels,
                    candidates_weights,
                    self._has_multiple_task_types,
                )

        return (
            candidates_preds,
            candidates_labels,
            candidates_weights,
            candidates_losses,
        )

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

import abc
from typing import Optional, Tuple

import torch

from modeling.ndp_module import NDPModule


class InteractionModule(torch.nn.Module):

    @abc.abstractmethod
    def get_item_embeddings(
        self,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def get_item_sideinfo(
        self,
        item_ids: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        pass

    @abc.abstractmethod
    def interaction(
        self,
        input_embeddings: torch.Tensor,  # [B, D]
        target_ids: torch.Tensor,  # [1, X] or [B, X]
        target_embeddings: Optional[torch.Tensor] = None,   # [1, X, D'] or [B, X, D']
    ) -> torch.Tensor:
        pass


class GeneralizedInteractionModule(InteractionModule):
    def __init__(
        self,
        ndp_module: NDPModule,
    ) -> None:
        super().__init__()

        self._ndp_module: NDPModule = ndp_module

    @abc.abstractmethod
    def debug_str(
        self,
    ) -> str:
        pass

    def interaction(
        self,
        input_embeddings: torch.Tensor,
        target_ids: torch.Tensor,
        target_embeddings: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        torch._assert(len(input_embeddings.size()) == 2, "len(input_embeddings.size()) must be 2")
        torch._assert(len(target_ids.size()) == 2, "len(target_ids.size()) must be 2")
        if target_embeddings is None:
            target_embeddings = self.get_item_embeddings(target_ids)
        torch._assert(len(target_embeddings.size()) == 3, "len(target_embeddings.size()) must be 3")

        with torch.autocast(enabled=True, dtype=torch.bfloat16, device_type="cuda"):
            return self._ndp_module(
                input_embeddings=input_embeddings,  # [B, self._input_embedding_dim]
                item_embeddings=target_embeddings,  # [1/B, X, self._item_embedding_dim]
                item_sideinfo=self.get_item_sideinfo(item_ids=target_ids),  # [1/B, X, self._item_sideinfo_dim]
                item_ids=target_ids,
                precomputed_logits=None,
            )
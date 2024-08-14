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

from typing import Dict, Callable, Optional, Tuple

import torch

from modeling.ndp_module import NDPModule


class DotProductSimilarity(NDPModule):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def debug_str(self) -> str:
        return "dp"

    def forward(
        self,
        input_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_sideinfo: Optional[torch.Tensor],
        item_ids: torch.Tensor,
        precomputed_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_embeddings: (B, D,) or (B * r, D) x float.
            item_embeddings: (1, X, D) or (B, X, D) x float.

        Returns:
            (B, X) x float (or (B * r, X) x float).
        """
        del item_ids

        if item_embeddings.size(0) == 1:
            # [B, D] x ([1, X, D] -> [D, X]) => [B, X]
            return torch.mm(input_embeddings, item_embeddings.squeeze(0).t()), {}  # [B, X]
        elif input_embeddings.size(0) != item_embeddings.size(0):
            # (B * r, D) x (B, X, D).
            B, X, D = item_embeddings.size()
            return torch.bmm(input_embeddings.view(B, -1, D), item_embeddings.permute(0, 2, 1)).view(-1, X)
        else:
            # assert input_embeddings.size(0) == item_embeddings.size(0)
            # [B, X, D] x ([B, D] -> [B, D, 1]) => [B, X, 1] -> [B, X]
            return torch.bmm(item_embeddings, input_embeddings.unsqueeze(2)).squeeze(2)

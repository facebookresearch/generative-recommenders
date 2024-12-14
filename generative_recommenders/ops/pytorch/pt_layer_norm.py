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


# pyre-strict


from typing import List

import torch


def pytorch_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return torch.nn.functional.layer_norm(
        x,
        normalized_shape,
        weight.to(x.dtype),
        bias.to(x.dtype),
        eps,
    )


def pytorch_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    return x * torch.sigmoid(
        torch.nn.functional.layer_norm(
            x,
            normalized_shape,
            weight.to(x.dtype),
            bias.to(x.dtype),
            eps,
        )
    )

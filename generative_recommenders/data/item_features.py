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

# pyre-unsafe

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class ItemFeatures:
    num_items: int
    max_jagged_dimension: int
    max_ind_range: List[int]  # [(,)] x num_features
    lengths: List[torch.Tensor]  # [(num_items,)] x num_features
    values: List[torch.Tensor]  # [(num_items, max_jagged_dimension)] x num_features

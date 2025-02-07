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
import contextlib
from typing import Any, Generator, Optional, Tuple

import torch
from generative_recommenders.common import fx_infer_max_len
from generative_recommenders.modules.stu import STU
from generative_recommenders.ops.jagged_tensors import (
    hstu_concat_l2_embeddings,
    hstu_split_l2_embeddings,
)


try:
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops")
    torch.ops.load_library("//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu")
except OSError:
    pass


@contextlib.contextmanager
# pyre-ignore[3]
def _freeze_rng_state() -> Generator[Any, None, None]:
    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            # pyre-ignore[61]
            torch.cuda.set_rng_state(cuda_rng_state)
        torch.set_rng_state(rng_state)


class DynamicSTU(STU):
    def __init__(self, stu: STU, is_inference: bool) -> None:
        super().__init__(is_inference)
        self._stu = stu

    @abc.abstractmethod
    def _preprocess(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
    ]:
        pass

    @abc.abstractmethod
    def _postprocess(
        self,
        stu_output: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            x,
            x_lengths,
            x_offsets,
            max_seq_len,
            num_targets,
            max_kv_caching_len,
            kv_caching_lengths,
        ) = self._preprocess(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        stu_output = self._stu(
            x=x,
            x_lengths=x_lengths,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
            max_kv_caching_len=max_kv_caching_len,
            kv_caching_lengths=kv_caching_lengths,
        )

        return self._postprocess(
            stu_output=stu_output,
        )


class SDSTU(DynamicSTU):
    def __init__(
        self,
        stu: STU,
        is_inference: bool,
        dropout_ratio: float = 0.5,
        seed: int = 0,
    ) -> None:
        """
        Stochastic Depth STU
        """
        super().__init__(stu=stu, is_inference=is_inference)
        self._dropout_ratio: float = dropout_ratio
        self._iter: int = 0
        self._seed: int = seed
        self._skip_x: Optional[torch.Tensor] = None

    def _preprocess(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
    ]:
        if self.training:
            with _freeze_rng_state():
                torch.manual_seed(self._iter + self._seed)
                prob = torch.rand(1)
                if prob.item() <= self._dropout_ratio:
                    new_x = torch.empty(size=(0, x.shape[1]), device=x.device)
                    self._skip_x = x
                    new_x_lengths = torch.zeros_like(x_lengths)
                    new_x_offsets = torch.zeros_like(x_offsets)
                    new_max_seq_len = 1
                else:
                    new_x = x
                    new_x_lengths = x_lengths
                    new_x_offsets = x_offsets
                    new_max_seq_len = max_seq_len
            self._iter += 1
        else:
            new_x = x
            new_x_lengths = x_lengths
            new_x_offsets = x_offsets
            new_max_seq_len = max_seq_len
        return (
            new_x,
            new_x_lengths,
            new_x_offsets,
            new_max_seq_len,
            num_targets,
            max_kv_caching_len,
            kv_caching_lengths,
        )

    def _postprocess(
        self,
        stu_output: torch.Tensor,
    ) -> torch.Tensor:
        if self.training and self._skip_x is not None:
            ret = self._skip_x
            self._skip_x = None
            return ret
        else:
            return stu_output


@torch.fx.wrap
def _fx_unwrap_optional_tuple_tensor(
    optional: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert optional is not None, "Expected optional to be non-None"
    return optional


class L2STU(DynamicSTU):
    def __init__(
        self,
        stu: STU,
        max_l2_len: int,
        is_inference: bool,
        contextual_seq_len: int = 0,
    ) -> None:
        """
        Stochastic Depth STU
        """
        super().__init__(stu=stu, is_inference=is_inference)
        self._max_l2_len: int = max_l2_len
        self._contextual_seq_len: int = contextual_seq_len
        self._saved_tensors: Optional[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ] = None
        self._runtime_max_l2_len: int = 0
        self._runtime_minus_l2_len: int = 0

    def _preprocess(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        x_offsets: torch.Tensor,
        max_seq_len: int,
        num_targets: torch.Tensor,
        max_kv_caching_len: int = 0,
        kv_caching_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        int,
        Optional[torch.Tensor],
    ]:
        minus_l2_lengths = (
            x_lengths - self._max_l2_len - num_targets - self._contextual_seq_len
        )
        minus_l2_lengths = torch.clamp(minus_l2_lengths, min=0)
        minus_l2_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            minus_l2_lengths
        )
        l2_lengths = x_lengths - minus_l2_lengths
        l2_offsets = x_offsets - minus_l2_offsets
        self._runtime_max_l2_len = fx_infer_max_len(l2_lengths)
        self._runtime_minus_l2_len = fx_infer_max_len(minus_l2_lengths)
        minus_l2_x, l2_x = hstu_split_l2_embeddings(
            max_seq_len=max_seq_len,
            x=x,
            minus_l2_offsets=minus_l2_offsets,
            l2_offsets=l2_offsets,
            contextual_seq_len=self._contextual_seq_len,
        )
        self._saved_tensors = (
            minus_l2_offsets,
            minus_l2_x,
            l2_offsets,
        )
        return (
            l2_x,
            l2_lengths,
            l2_offsets,
            self._runtime_max_l2_len,
            num_targets,
            max_kv_caching_len,
            kv_caching_lengths,
        )

    def _postprocess(
        self,
        stu_output: torch.Tensor,
    ) -> torch.Tensor:
        (
            minus_l2_offsets,
            minus_l2_x,
            l2_offsets,
        ) = _fx_unwrap_optional_tuple_tensor(self._saved_tensors)
        self._saved_tensors = None
        return hstu_concat_l2_embeddings(
            max_minus_l2_len=self._runtime_minus_l2_len,
            minus_l2_x=minus_l2_x,
            minus_l2_offsets=minus_l2_offsets,
            max_l2_len=self._runtime_max_l2_len,
            l2_x=stu_output,
            l2_offsets=l2_offsets,
            contextual_seq_len=self._contextual_seq_len,
            kernel=self.hammer_kernel(),
        )

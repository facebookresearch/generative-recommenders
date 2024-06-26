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

import torch


def batch_gather_embeddings(
    rowwise_indices: torch.Tensor,
    embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        rowwise_indices: (B, N) x int, where each entry is in [0, X).
        embeddings: (B, X, D,) x float.

    Returns:
        (B, N, D,) x float, embeddings corresponding to rowwise_indices.
    """
    _, N = rowwise_indices.size()
    B, X, D = embeddings.size()
    flattened_indices = (
        rowwise_indices
        + torch.arange(
            start=0,
            end=B,
            step=1,
            dtype=rowwise_indices.dtype,
            device=rowwise_indices.device,
        )
        .unsqueeze(1)
        .expand(-1, N)
        * X
    )
    return embeddings.view(-1, D)[flattened_indices, :].reshape(
        rowwise_indices.size() + (D,)
    )


def batch_scatter_embeddings(
    dst_embeddings: torch.Tensor,
    rowwise_indices: torch.Tensor,
    src_embeddings: torch.Tensor,
) -> None:
    """
    Args:
        dst_embeddings: (B, N, D,) x float.
        rowwise_indices: (B,) x int, where each entry is in [0, N - 1).
        source_embeddings: (B, D,) x float.
    """
    B, N, D = dst_embeddings.size()
    flattened_indices = rowwise_indices + torch.arange(
        start=0,
        end=B * N,
        step=N,
        dtype=rowwise_indices.dtype,
        device=rowwise_indices.device,
    )
    dst_embeddings.view(B * N, D)[flattened_indices, :] = src_embeddings


def get_current_embeddings(
    lengths: torch.Tensor,
    encoded_embeddings: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        lengths: (B,) x int
        seq_embeddings: (B, N, D,) x float

    Returns:
        (B, D,) x float, where [i, :] == encoded_embeddings[i, lengths[i] - 1, :]
    """
    B, N, D = encoded_embeddings.size()
    flattened_offsets = (lengths - 1) + torch.arange(
        start=0, end=B, step=1, dtype=lengths.dtype, device=lengths.device
    ) * N
    return encoded_embeddings.reshape(-1, D)[flattened_offsets, :].reshape(B, D)


def jagged_or_dense_repeat_interleave_dim0(
    x: torch.Tensor, lengths: torch.Tensor, repeats: int
) -> torch.Tensor:
    if len(x.size()) == 3:
        return x.repeat_interleave(repeats, dim=0)
    else:
        assert len(x.size()) == 2, f"x.size() = {x.size()}"
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
            max_lengths=[lengths.max()],
            padding_value=0.0,
        )
        lengths = lengths.repeat_interleave(repeats, dim=0)
        return torch.ops.fbgemm.dense_to_jagged(
            padded_x.repeat_interleave(repeats, dim=0),
            [torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
        )[0]


def jagged_or_dense_index_select_dim0(
    x: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    if len(x.size()) == 3:
        return x[indices, :, :]
    else:
        assert len(x.size()) == 2, f"x.size() = {x.size()}"
        padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
            values=x,
            offsets=[torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)],
            max_lengths=[lengths.max()],
            padding_value=0.0,
        )
        return torch.ops.fbgemm.dense_to_jagged(
            padded_x[indices, :],
            [torch.ops.fbgemm.asynchronous_complete_cumsum(lengths[indices])],
        )[0]

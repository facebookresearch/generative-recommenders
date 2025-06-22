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

import os
from datetime import datetime
from typing import Any, Dict, Optional, Set

import gin

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.optim.optimizer import Optimizer
from torchrec.distributed.types import ShardedTensor


class SparseState(Stateful):
    def __init__(self, model: torch.nn.Module, sparse_tensor_keys: Set[str]) -> None:
        self.model = model
        self.sparse_tensor_keys = sparse_tensor_keys

    def state_dict(self) -> Dict[str, torch.Tensor]:
        out_dict: Dict[str, torch.Tensor] = {}
        is_sharded_tensor: Optional[bool] = None
        for k, v in self.model.state_dict().items():
            if k in self.sparse_tensor_keys:
                if is_sharded_tensor is None:
                    is_sharded_tensor = isinstance(v, ShardedTensor)
                assert is_sharded_tensor == isinstance(v, ShardedTensor)
                out_dict[k] = v
        return out_dict

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        incompatible_keys = self.model.load_state_dict(state_dict, strict=False)
        assert not incompatible_keys.unexpected_keys


def is_sparse_key(k: str, v: torch.Tensor) -> bool:
    return isinstance(v, ShardedTensor) or "embedding_collection" in k


def load_dense_state_dict(model: torch.nn.Module, state_dict: Dict[str, Any]) -> None:
    own_state = model.state_dict()
    own_state_dense_keys = {k for k, v in own_state.items() if not is_sparse_key(k, v)}
    state_dict_dense_keys = {
        k for k, v in state_dict.items() if not is_sparse_key(k, v)
    }
    assert (
        own_state_dense_keys == state_dict_dense_keys
    ), f"expects {own_state_dense_keys} but gets {state_dict_dense_keys}"
    for name in state_dict_dense_keys:
        param = state_dict[name]
        if isinstance(param, torch.nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)


@gin.configurable
def save_dmp_checkpoint_every_n_steps(
    model: torch.nn.Module,
    optimizer: Optimizer,
    rank: int,
    path: str,
    batch_idx: int,
    save_every_n_steps: Optional[int],
) -> None:
    if (
        rank == 0
        and save_every_n_steps is not None
        and batch_idx != 0
        and batch_idx % save_every_n_steps == 0
    ):
        return save_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            rank=rank,
            path=path,
        )
    else:
        return None


def save_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    rank: int,
    path: str = "",
) -> None:
    if path == "":
        return
    now = datetime.now()
    formatted_datetime = now.strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{path}/{formatted_datetime}"
    if not os.path.exists(path) and rank == 0:
        os.makedirs(path)
    sparse_path = f"{path}/sparse/"
    if not os.path.exists(sparse_path) and rank == 0:
        os.makedirs(sparse_path)
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if isinstance(v, ShardedTensor)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.save(
        sparse_dict,
        storage_writer=torch.distributed.checkpoint.FileSystemWriter(sparse_path),
    )
    if rank == 0:
        dense_state_dict = {
            k: v
            for k, v in model.state_dict().items()
            if not isinstance(v, ShardedTensor)
        }
        torch.save(
            {
                "dense_dict": dense_state_dict,
                "optimizer_dict": optimizer.state_dict(),
                "sparse_tensor_keys": sparse_tensor_keys,
            },
            non_sparse_ckpt,
        )
    print("checkpoint successfully saved")


@gin.configurable
def load_sparse_checkpoint(
    model: torch.nn.Module,
    path: str = "",
) -> None:
    if path == "":
        return
    sparse_path = f"{path}/sparse/"

    sparse_tensor_keys = {
        k for k, v in model.state_dict().items() if is_sparse_key(k, v)
    }
    sparse_dict = {"sparse_dict": SparseState(model, sparse_tensor_keys)}
    torch.distributed.checkpoint.load(
        sparse_dict,
        storage_reader=torch.distributed.checkpoint.FileSystemReader(sparse_path),
    )
    print("sparse checkpoint successfully loaded")


@gin.configurable
def load_nonsparse_checkpoint(
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    path: str = "",
) -> None:
    if path == "":
        return
    non_sparse_ckpt = f"{path}/non_sparse.ckpt"

    non_sparse_state_dict = torch.load(non_sparse_ckpt)
    load_dense_state_dict(model, non_sparse_state_dict["dense_dict"])
    print("dense checkpoint successfully loaded")
    if optimizer is not None:
        optimizer.load_state_dict(non_sparse_state_dict["optimizer_dict"])
        print("optimizer checkpoint successfully loaded")


@gin.configurable
def load_dmp_checkpoint(
    model: torch.nn.Module,
    optimizer: Optimizer,
    path: str = "",
) -> None:
    load_sparse_checkpoint(model=model, path=path)
    load_nonsparse_checkpoint(model=model, optimizer=optimizer, path=path)

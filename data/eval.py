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

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import logging
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from indexing.candidate_index import CandidateIndex, TopKModule
from modeling.ndp_module import NDPModule
from modeling.sequential.features import SequentialFeatures, movielens_seq_features_from_row
from modeling.sequential.utils import get_current_embeddings


logging.basicConfig(stream=sys.stdout, level=logging.INFO)


@dataclass
class EvalState:
    candidate_index: CandidateIndex
    top_k_module: TopKModule


@dataclass
class EvalState:
    all_item_ids: Set[int]
    candidate_index: CandidateIndex
    top_k_module: TopKModule


def get_eval_state(
    model: NDPModule,
    all_item_ids: List[int],  # [X]
    negatives_sampler,
    top_k_module_fn: Callable[[torch.Tensor, torch.Tensor], TopKModule],
    device: torch.device,
    float_dtype: Optional[torch.dtype] = None,
) -> EvalState:
    # Exhaustively eval all items (incl. seen ids).
    eval_negatives_ids = torch.as_tensor(all_item_ids).to(device).unsqueeze(0)  # [1, X]
    eval_negative_embeddings = negatives_sampler.normalize_embeddings(
        model.get_item_embeddings(eval_negatives_ids)
    )
    if float_dtype is not None:
        eval_negative_embeddings = eval_negative_embeddings.to(float_dtype)
    candidates = CandidateIndex(
        ids=eval_negatives_ids,
        embeddings=eval_negative_embeddings,
    )
    return EvalState(
        all_item_ids=set(all_item_ids),
        candidate_index=candidates,
        top_k_module=top_k_module_fn(eval_negative_embeddings, eval_negatives_ids)
    )


@torch.inference_mode
def eval_metrics_v2_from_tensors(
    eval_state: EvalState,
    model: NDPModule,
    seq_features: SequentialFeatures,
    target_ids: torch.Tensor,  # [B, 1]
    min_positive_rating: int = 4,
    target_ratings: Optional[torch.Tensor] = None,  # [B, 1]
    epoch: Optional[str] = None,
    include_full_matrices: bool = False,
    filter_invalid_ids: bool = True,
    user_max_batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[float]]:
    """
    Args:
        eval_negatives_ids: Optional[Tensor]. If not present, defaults to eval over
            the entire corpus (`num_items`) excluding all the items that users have
            seen in the past (historical_ids, target_ids). This is consistent with
            papers like SASRec and TDM but may not be fair in practice as retrieval
            modules don't have access to read state during the initial fetch stage.
        item_max_batch_size: int. maximum number of items (*not* users - i.e., M/R
            not B) to eval per batch.
        filter_invalid_ids: bool. If true, filters seen ids by default.
    Returns:
        keyed metric -> list of values for each example.
    """
    B, _ = target_ids.shape
    device = target_ids.device

    for target_id in target_ids:
        target_id = int(target_id)
        if target_id not in eval_state.all_item_ids:
            print(f"missing target_id {target_id}")

    start_time = time.time()
    # computes ro- part exactly once.
    shared_input_embeddings = model.encode(
        past_lengths=seq_features.past_lengths,
        past_ids=seq_features.past_ids,
        past_embeddings=model.get_item_embeddings(seq_features.past_ids),
        past_payloads=seq_features.past_payloads,
    )
    if dtype is not None:
        shared_input_embeddings = shared_input_embeddings.to(dtype)

    MAX_K = 2500
    k = min(MAX_K, eval_state.candidate_index.ids.size(1))
    user_max_batch_size = user_max_batch_size or shared_input_embeddings.size(0)
    num_batches = (
        (shared_input_embeddings.size(0) + user_max_batch_size - 1) // user_max_batch_size
    )
    eval_top_k_ids_all = []
    eval_top_k_prs_all = []
    for mb in range(num_batches):
        eval_top_k_ids, eval_top_k_prs, _ = eval_state.candidate_index.get_top_k_outputs(
            query_embeddings=shared_input_embeddings[mb * user_max_batch_size: (mb + 1) * user_max_batch_size, ...],
            top_k_module=eval_state.top_k_module,
            k=k,
            #policy_fn=lambda ids, embeddings: model.interaction(
            #    input_embeddings=shared_input_embeddings,
            #    target_ids=ids,
            #    target_embeddings=embeddings,
            #),
            invalid_ids=seq_features.past_ids[
                mb * user_max_batch_size: (mb + 1) * user_max_batch_size, :
            ] if filter_invalid_ids else None,
            return_embeddings=False,
        )
        eval_top_k_ids_all.append(eval_top_k_ids)
        eval_top_k_prs_all.append(eval_top_k_prs)

    if num_batches == 1:
        eval_top_k_ids = eval_top_k_ids_all[0]
        eval_top_k_prs = eval_top_k_prs_all[0]
    else:
        eval_top_k_ids = torch.cat(eval_top_k_ids_all, dim=0)
        eval_top_k_prs = torch.cat(eval_top_k_prs_all, dim=0)

    assert eval_top_k_ids.size(1) == k
    _, eval_rank_indices = torch.max(
        torch.cat(
            [eval_top_k_ids, target_ids],
            dim=1,
        ) == target_ids,
        dim=1,
    )
    eval_ranks = torch.where(eval_rank_indices == k, MAX_K + 1, eval_rank_indices + 1)
    # print(f"eval_ranks.min()={eval_ranks.min()}, eval_ranks.max()={eval_ranks.max()}")

    output = {
        "ndcg@1": torch.where(
            eval_ranks <= 1,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).tolist(),
        "ndcg@10": torch.where(
            eval_ranks <= 10,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).tolist(),
        "ndcg@50": torch.where(
            eval_ranks <= 50,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).tolist(),
        "ndcg@200": torch.where(
            eval_ranks <= 200,
            1.0 / torch.log2(eval_ranks + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).tolist(),
        "hr@1": (eval_ranks <= 1).tolist(),
        "hr@10": (eval_ranks <= 10).tolist(),
        "hr@50": (eval_ranks <= 50).tolist(),
        "hr@200": (eval_ranks <= 200).tolist(),
        "hr@500": (eval_ranks <= 500).tolist(),
        "hr@1000": (eval_ranks <= 1000).tolist(),
        "hr@5000": (eval_ranks <= 5000).tolist(),
        "mrr": (1.0 / eval_ranks).tolist(),
    }
    if target_ratings is not None:
        target_ratings = target_ratings.squeeze(1)  # [B]
        output["ndcg@10_>=4"] = torch.where(
            eval_ranks[target_ratings >= 4] <= 10,
            1.0 / torch.log2(eval_ranks[target_ratings >= 4] + 1),
            torch.zeros(1, dtype=torch.float32, device=device),
        ).tolist()
        output[f"hr@10_>={min_positive_rating}"] = (
            eval_ranks[target_ratings >= min_positive_rating] <= 10
        ).tolist()
        output[f"hr@50_>={min_positive_rating}"] = (
            eval_ranks[target_ratings >= min_positive_rating] <= 50
        ).tolist()
        output[f"mrr_>={min_positive_rating}"] = (
            1.0 / eval_ranks[target_ratings >= min_positive_rating]
        ).tolist()

    if include_full_matrices:
        output["raw_eval_logits"] = raw_eval_logits - 1
        # print(output["raw_eval_logits"].sum(-1))
    return output


def eval_recall_metrics_from_tensors(
    eval_state: EvalState,
    model: NDPModule,
    seq_features: SequentialFeatures,
    include_full_matrices: bool = False,
    user_max_batch_size: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
) -> Dict[str, List[float]]:
    target_ids = seq_features.past_ids[:, -1].unsqueeze(1)
    filtered_past_ids = seq_features.past_ids.detach().clone()
    filtered_past_ids[:, -1] = torch.zeros_like(target_ids.squeeze(1))
    return eval_metrics_v2_from_tensors(
        eval_state=eval_state,
        model=model,
        seq_features=SequentialFeatures(
            past_lengths=seq_features.past_lengths - 1,
            past_ids=filtered_past_ids,
            past_embeddings=seq_features.past_embeddings,
            past_payloads=seq_features.past_payloads,
        ),
        target_ids=target_ids,
        user_max_batch_size=user_max_batch_size,
        dtype=dtype,
    )


def add_to_summary_writer(writer: SummaryWriter, batch_id: int, metrics: Dict[str, List[float]], prefix: str) -> None:
    if writer is None:
        logging.warn("Writer is None. Skipping logging.")
        return
    for key, values in metrics.items():
        avg_value = sum(values) / len(values)
        writer.add_scalar(f"{prefix}/{key}", avg_value, batch_id)



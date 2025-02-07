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


import logging
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Tuple

import torch
from generative_recommenders.common import (
    fx_infer_max_len,
    fx_mark_length_features,
    HammerKernel,
    HammerModule,
)
from generative_recommenders.modules.hstu_transducer import HSTUTransducer
from generative_recommenders.modules.multitask_module import (
    DefaultMultitaskModule,
    TaskConfig,
)
from generative_recommenders.modules.positional_encoder import HSTUPositionalEncoder
from generative_recommenders.modules.postprocessors import (
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)
from generative_recommenders.modules.preprocessors import ContextualPreprocessor
from generative_recommenders.modules.stu import STU, STULayer, STULayerConfig, STUStack
from generative_recommenders.modules.utils import init_mlp_weights_optional_bias
from generative_recommenders.ops.jagged_tensors import concat_2D_jagged
from generative_recommenders.ops.layer_norm import LayerNorm, SwishLayerNorm
from torch.autograd.profiler import record_function
from torchrec import KeyedJaggedTensor
from torchrec.modules.embedding_configs import EmbeddingConfig
from torchrec.modules.embedding_modules import EmbeddingCollection

logger: logging.Logger = logging.getLogger(__name__)

torch.ops.load_library("//hammer/oss/generative_recommenders/ops/cpp:cpp_ops")

torch.fx.wrap("fx_infer_max_len")
torch.fx.wrap("len")


class SequenceEmbedding(NamedTuple):
    lengths: torch.Tensor
    embedding: torch.Tensor


@dataclass
class DlrmHSTUConfig:
    hstu_num_heads: int = 1
    hstu_attn_linear_dim: int = 256
    hstu_attn_qk_dim: int = 128
    hstu_attn_num_layers: int = 12
    hstu_embedding_table_dim: int = 192
    hstu_transducer_embedding_dim: int = 0
    hstu_group_norm: bool = False
    hstu_input_dropout_rate: float = 0.2
    hstu_linear_dropout_rate: float = 0.2
    contextual_feature_to_max_length: Dict[str, int] = field(default_factory=dict)
    contextual_feature_to_min_uih_length: Dict[str, int] = field(default_factory=dict)
    candidates_weight_feature_name: str = ""
    candidates_watchtime_feature_name: str = ""
    causal_multitask_weights: float = 0.2
    multitask_configs: List[TaskConfig] = field(default_factory=list)
    user_embedding_feature_names: List[str] = field(default_factory=list)
    item_embedding_feature_names: List[str] = field(default_factory=list)
    uih_post_id_feature_name: str = ""
    uih_action_time_feature_name: str = ""
    uih_weight_feature_name: str = ""
    hstu_uih_feature_names: List[str] = field(default_factory=list)
    hstu_candidate_feature_names: List[str] = field(default_factory=list)
    merge_uih_candidate_feature_mapping: List[Tuple[str, str]] = field(
        default_factory=list
    )
    action_weights: Optional[List[int]] = None
    enable_postprocessor: bool = True
    use_layer_norm_postprocessor: bool = False


class DlrmHSTU(HammerModule):
    def __init__(  # noqa C901
        self,
        hstu_configs: DlrmHSTUConfig,
        embedding_tables: Dict[str, EmbeddingConfig],
        is_inference: bool,
    ) -> None:
        super().__init__(is_inference=is_inference)
        logger.info(f"Initialize HSTU module with configs {hstu_configs}")
        self._hstu_configs = hstu_configs
        self._embedding_collection = EmbeddingCollection(
            tables=list(embedding_tables.values()),
            need_indices=False,
            device=torch.device("meta"),
        )

        # multitask configs, must sort by task types
        self._multitask_configs: List[TaskConfig] = hstu_configs.multitask_configs
        self._multitask_configs.sort(key=lambda x: x.task_type)
        self._multitask_module = DefaultMultitaskModule(
            task_configs=self._multitask_configs,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            prediction_fn=lambda in_dim, num_tasks: torch.nn.Sequential(
                torch.nn.Linear(in_features=in_dim, out_features=512),
                SwishLayerNorm(512),
                torch.nn.Linear(in_features=512, out_features=num_tasks),
            ).apply(init_mlp_weights_optional_bias),
            candidates_weight_feature_name=self._hstu_configs.candidates_weight_feature_name,
            candidates_watchtime_feature_name=self._hstu_configs.candidates_watchtime_feature_name,
            causal_multitask_weights=hstu_configs.causal_multitask_weights,
            is_inference=self._is_inference,
        )

        # Preprocessor setup
        preprocessor = ContextualPreprocessor(
            input_embedding_dim=hstu_configs.hstu_embedding_table_dim,
            output_embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            contextual_feature_to_max_length=hstu_configs.contextual_feature_to_max_length,
            contextual_feature_to_min_uih_length=hstu_configs.contextual_feature_to_min_uih_length,
            uih_weight_name=hstu_configs.uih_weight_feature_name,
            action_weights=hstu_configs.action_weights,
            is_inference=self._is_inference,
            interleave_action_with_target=True,
            interleave_action_with_uih=True,
        )

        positional_encoder = HSTUPositionalEncoder(
            num_position_buckets=8192,
            num_time_buckets=2048,
            embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
            is_inference=self._is_inference,
            use_time_encoding=True,
        )

        if hstu_configs.enable_postprocessor:
            if hstu_configs.use_layer_norm_postprocessor:
                postprocessor = LayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
            else:
                postprocessor = TimestampLayerNormPostprocessor(
                    embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                    time_duration_features=[
                        (60 * 60, 24),  # hour of day
                        (24 * 60 * 60, 7),  # day of week
                        # (24 * 60 * 60, 365), # time of year (approximate)
                    ],
                    eps=1e-5,
                    is_inference=self._is_inference,
                )
        else:
            postprocessor = None

        stu_module: STU = STUStack(
            stu_list=[
                STULayer(
                    config=STULayerConfig(
                        embedding_dim=hstu_configs.hstu_transducer_embedding_dim,
                        num_heads=hstu_configs.hstu_num_heads,
                        hidden_dim=hstu_configs.hstu_attn_linear_dim,
                        attention_dim=hstu_configs.hstu_attn_qk_dim,
                        output_dropout_ratio=hstu_configs.hstu_linear_dropout_rate,
                        use_group_norm=hstu_configs.hstu_group_norm,
                        causal=True,
                        target_aware=True,
                        max_attn_len=None,
                        attn_alpha=None,
                        recompute_normed_x=True,
                        recompute_uvqk=True,
                        recompute_y=True,
                        sort_by_length=True,
                        contextual_seq_len=0,
                    ),
                    is_inference=is_inference,
                )
                for _ in range(hstu_configs.hstu_attn_num_layers)
            ],
            is_inference=is_inference,
        )
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            stu_module=stu_module,
            input_preprocessor=preprocessor,
            output_postprocessor=postprocessor,
            input_dropout_ratio=hstu_configs.hstu_input_dropout_rate,
            positional_encoder=positional_encoder,
            is_inference=self._is_inference,
            return_full_embeddings=False,
            listwise=False,
        )

        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hstu_configs.hstu_embedding_table_dim
                * len(self._hstu_configs.item_embedding_feature_names),
                out_features=512,
            ),
            SwishLayerNorm(512),
            torch.nn.Linear(
                in_features=512,
                out_features=hstu_configs.hstu_transducer_embedding_dim,
            ),
            LayerNorm(hstu_configs.hstu_transducer_embedding_dim),
        ).apply(init_mlp_weights_optional_bias)

    def _construct_payload(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> Dict[str, torch.Tensor]:
        if len(self._hstu_configs.contextual_feature_to_max_length) > 0:
            contextual_lengths = torch.stack(
                [
                    seq_embeddings[x].lengths
                    for x in self._hstu_configs.contextual_feature_to_max_length.keys()
                ],
                dim=0,
            )
            contextual_offsets = torch.ops.gr.batched_complete_cumsum(
                contextual_lengths
            )
        else:
            # Dummy, offsets are unused
            contextual_offsets = torch.empty((0, 0))
        return {
            **payload_features,
            **{
                x: seq_embeddings[x].embedding
                for x in self._hstu_configs.contextual_feature_to_max_length.keys()
            },
            **{
                x + "_offsets": contextual_offsets[i]
                for i, x in enumerate(
                    list(self._hstu_configs.contextual_feature_to_max_length.keys())
                )
            },
        }

    def _user_forward(
        self,
        payload_features: Dict[str, torch.Tensor],
        seq_embeddings: Dict[str, SequenceEmbedding],
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        source_lengths = seq_embeddings[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths
        runtime_max_seq_len = fx_infer_max_len(source_lengths)
        source_timestamps = payload_features[
            self._hstu_configs.uih_action_time_feature_name
        ]
        candidates_user_embeddings, _ = self._hstu_transducer(
            max_seq_len=runtime_max_seq_len,
            seq_embeddings=seq_embeddings[
                self._hstu_configs.uih_post_id_feature_name
            ].embedding,
            seq_lengths=source_lengths,
            seq_timestamps=source_timestamps,
            seq_payloads=self._construct_payload(
                payload_features=payload_features,
                seq_embeddings=seq_embeddings,
            ),
            num_targets=num_candidates,
        )

        return candidates_user_embeddings

    def _item_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
    ) -> torch.Tensor:  # [L, D]
        all_embeddings = [
            torch.cat(
                [
                    seq_embeddings[name].embedding
                    for name in self._hstu_configs.item_embedding_feature_names
                ],
                dim=-1,
            )
        ]
        item_embeddings = self._item_embedding_mlp(torch.cat(all_embeddings, dim=-1))
        return item_embeddings

    def preprocess(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        Dict[str, SequenceEmbedding],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        # embedding lookup for uih and candidates
        merged_sparse_features = KeyedJaggedTensor.from_lengths_sync(
            keys=uih_features.keys() + candidates_features.keys(),
            values=torch.cat(
                [uih_features.values(), candidates_features.values()],
                dim=0,
            ),
            lengths=torch.cat(
                [uih_features.lengths(), candidates_features.lengths()],
                dim=0,
            ),
        )
        seq_embeddings_dict = self._embedding_collection(merged_sparse_features)
        num_candidates = fx_mark_length_features(
            candidates_features.lengths().view(len(candidates_features.keys()), -1)
        )[0]
        max_num_candidates = fx_infer_max_len(num_candidates)
        uih_seq_lengths = uih_features[
            self._hstu_configs.uih_post_id_feature_name
        ].lengths()
        max_uih_len = fx_infer_max_len(uih_seq_lengths)

        # prepare payload features
        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if (
                candidate_feature_name
                not in self._hstu_configs.item_embedding_feature_names
                and uih_feature_name
                not in self._hstu_configs.user_embedding_feature_names
            ):
                values_left = uih_features[uih_feature_name].values()
                if self._is_inference and (
                    candidate_feature_name
                    == self._hstu_configs.candidates_weight_feature_name
                    or candidate_feature_name
                    == self._hstu_configs.candidates_watchtime_feature_name
                ):
                    total_candidates = torch.sum(num_candidates).item()
                    values_right = torch.zeros(
                        total_candidates,  # pyre-ignore
                        dtype=torch.int64,
                        device=values_left.device,
                    )
                else:
                    values_right = candidates_features[candidate_feature_name].values()
                merged_values = concat_2D_jagged(
                    max_len_left=max_uih_len,
                    offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                        uih_seq_lengths
                    ),
                    values_left=values_left.unsqueeze(-1),
                    max_len_right=max_num_candidates,
                    offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                        num_candidates
                    ),
                    values_right=values_right.unsqueeze(-1),
                    kernel=HammerKernel.PYTORCH
                    if self._is_inference
                    else self.hammer_kernel(),
                ).squeeze(-1)
                payload_features[uih_feature_name] = merged_values
                payload_features[candidate_feature_name] = values_right
        payload_features["offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(
            uih_seq_lengths + num_candidates
        )

        seq_embeddings = {
            k: SequenceEmbedding(
                lengths=seq_embeddings_dict[k].lengths(),
                embedding=seq_embeddings_dict[k].values(),
            )
            for k in self._hstu_configs.user_embedding_feature_names
            + self._hstu_configs.item_embedding_feature_names
        }

        return (
            seq_embeddings,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    def main_forward(
        self,
        seq_embeddings: Dict[str, SequenceEmbedding],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        # merge uih and candidates embeddings
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._hstu_configs.merge_uih_candidate_feature_mapping:
            if uih_feature_name in seq_embeddings:
                seq_embeddings[uih_feature_name] = SequenceEmbedding(
                    lengths=uih_seq_lengths + num_candidates,
                    embedding=concat_2D_jagged(
                        max_len_left=max_uih_len,
                        offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            uih_seq_lengths
                        ),
                        values_left=seq_embeddings[uih_feature_name].embedding,
                        max_len_right=max_num_candidates,
                        offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                            num_candidates
                        ),
                        values_right=seq_embeddings[candidate_feature_name].embedding,
                        kernel=self.hammer_kernel(),
                    ),
                )

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(
                seq_embeddings,
            )
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                payload_features,
                seq_embeddings,
                num_candidates=num_candidates,
            )
        with record_function("## multitask_module ##"):
            mt_target_preds, mt_target_labels, mt_target_weights, mt_losses = (
                self._multitask_module(
                    encoded_user_embeddings=candidates_user_embeddings,
                    item_embeddings=candidates_item_embeddings,
                    payload_features=payload_features,
                )
            )

        aux_losses: Dict[str, torch.Tensor] = {}
        if not self._is_inference and self.training:
            for i, task in enumerate(self._multitask_configs):
                aux_losses[task.task_name] = mt_losses[i]

        return (
            candidates_user_embeddings,
            candidates_item_embeddings,
            aux_losses,
            mt_target_preds,
            mt_target_labels,
            mt_target_weights,
        )

    def forward(
        self,
        uih_features: KeyedJaggedTensor,
        candidates_features: KeyedJaggedTensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        with record_function("## preprocess ##"):
            (
                seq_embeddings,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(
                uih_features=uih_features,
                candidates_features=candidates_features,
            )

        with record_function("## main_forward ##"):
            return self.main_forward(
                seq_embeddings=seq_embeddings,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )
